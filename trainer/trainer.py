from typing import Any, Dict, Optional
from config.training_config import TrainerState, TrainerConfig
import torch 
from deepspeed import DeepSpeedEngine
from torch.amp import autocast
from tqdm import tqdm
import os
import math
import time 

class LLMTrainer:
    def __init__(self, engine, train_dataloader, val_dataloader, trainer_cfg, logger=None):
        self.engine = engine
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.cfg = trainer_cfg
        self.logger = logger
        self.state = TrainerState(global_step=0, best_metric=float("inf"), should_stop=False)
        self.global_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        self.world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        self.output_dir = trainer_cfg.output_dir
        self.max_steps = trainer_cfg.max_steps
        self.eval_interval = trainer_cfg.eval_interval
        self.save_interval = trainer_cfg.save_interval
        self.log_interval = trainer_cfg.log_interval
        self.start_time = None
        self.total_tokens = 0
    
    def train(self) -> TrainerState:
        self.engine.train()
        if self.global_rank == 0:
            os.makedirs(self.cfg.output_dir, exist_ok=True)

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        self.start_time = time.time()
        step = self.state.global_step
        pbar = tqdm(range(self.cfg.max_steps), desc="Training", disable=self.global_rank != 0, initial=step)

        def get_train_iter():
            while True:
                for batch in self.train_dataloader:
                    yield batch
        
        train_iter = get_train_iter()

        while step < self.cfg.max_steps:
            batch = next(train_iter)
            loss = self.training_step(batch)
            self.state.global_step = step
            current_tokens = batch["input_ids"].numel() * self.world_size
            self.total_tokens += current_tokens

            if step % self.cfg.log_interval == 0 and self.global_rank == 0:
                elapsed = max(time.time() - self.start_time, 1e-9)
                tokens_per_sec = self.total_tokens / elapsed
                ppl = math.exp(loss) if loss < 20 else float("inf")
                pbar.set_postfix({"loss": f"{loss:.4f}", "ppl": f"{ppl:.2f}", "tps": f"{tokens_per_sec:.0f}"})
                pbar.update(1)

                if self.logger:
                    self.logger.report_scalar("Train", "Loss", value=loss, iteration=step)
                    self.logger.report_scalar("Train", "Perplexity", value=ppl, iteration=step)
                    self.logger.report_scalar("Train", "LR", value=self.engine.get_lr()[0], iteration=step)
                    self.logger.report_scalar("Performance", "Tokens/Sec", value=tokens_per_sec, iteration=step)

            if step > 0 and step % self.cfg.eval_interval == 0:
                eval_metrics = self.evaluation_step()
                if self.global_rank == 0:
                    if self.logger:
                        self.logger.report_scalar("Eval", "Loss", value=eval_metrics["loss"], iteration=step)
                        self.logger.report_scalar("Eval", "Perplexity", value=eval_metrics["ppl"], iteration=step)

                best_metric = self.state.best_metric if self.state.best_metric is not None else float("inf")
                if eval_metrics["loss"] < best_metric:
                    self.state.best_metric = eval_metrics["loss"]
                    self.save_checkpoint("best")

            if step > 0 and step % self.cfg.save_interval == 0:
                self.save_checkpoint(f"step_{step}")

            step += 1

        self.save_checkpoint("final")
        return self.state

    def training_step(self, batch: Dict[str, Any]) -> float:
        input_ids = batch["input_ids"].to(self.engine.device, non_blocking=True)
        labels = batch["labels"].to(self.engine.device, non_blocking=True)
        with autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = self.engine(input_ids=input_ids, labels=labels)
            loss = outputs["loss"]
        self.engine.backward(loss)
        self.engine.step()
        
        return loss.item()

    @torch.no_grad()
    def evaluation_step(self) -> Dict[str, float]:
        self.engine.eval()
        torch.cuda.empty_cache() 
        total_loss = 0.0
        num_batches = 0
        max_eval_batches = 50
        val_iter = iter(self.val_dataloader)
        last_batch = None
        
        for _ in range(max_eval_batches):
            try:
                batch = next(val_iter)
                last_batch = batch
                is_dummy = False
            except StopIteration:
                if last_batch is None:
                    raise RuntimeError(f"Rank {self.global_rank} has no validation data. Increase val_take.")
                batch = last_batch
                is_dummy = True
            
            input_ids = batch["input_ids"].to(self.engine.device)
            labels = batch["labels"].to(self.engine.device)
            outputs = self.engine(input_ids=input_ids, labels=labels)
            
            if not is_dummy:
                total_loss += outputs["loss"].item()
                num_batches += 1
        
        stats = torch.tensor([total_loss, float(num_batches)], device=self.engine.device)
        torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
        actual_total_loss = stats[0].item()
        actual_total_batches = stats[1].item()
        avg_loss = actual_total_loss / max(1.0, actual_total_batches)
        self.engine.train()
        return {
            "loss": avg_loss,
            "ppl": math.exp(avg_loss) if avg_loss < 20 else float('inf')
        }

    def save_checkpoint(self, tag: str) -> None:
        save_dir = os.path.join(self.cfg.output_dir, tag)
        self.engine.save_checkpoint(save_dir, client_state={"state": self.state})
        if self.global_rank == 0:
            print(f"Checkpoint saved to {save_dir}")


