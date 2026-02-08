import os
import json
import torch
import torch.nn.functional as F
import deepspeed
from typing import Any, Dict, Optional, Tuple
from transformers import AutoTokenizer, HfArgumentParser
from tqdm import tqdm
from clearml import Task
from datetime import timedelta

from RL.rl_config import RLModelConfig, RLDataConfig, RLTrainerConfig
from RL.rl_dataset import RLDataModule
from RL.prepare_data_rl import prepare_rl_data, RLDataPrepConfig
from model.model import LLaMAForCausalLM
from config.training_config import LLaMAConfig


def _is_rank0() -> bool:
    return (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0


def _setup_local_cuda_device() -> None:
    if not torch.cuda.is_available():
        return
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is None:
        return
    try:
        torch.cuda.set_device(int(local_rank))
    except Exception:
        return


def _unwrap_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        return state_dict["state_dict"]
    if isinstance(state_dict, dict) and "module" in state_dict:
        return state_dict["module"]
    if isinstance(state_dict, dict) and "model" in state_dict:
        return state_dict["model"]
    return state_dict


def _load_state_dict_from_checkpoint(path: str) -> Optional[Dict[str, Any]]:
    # Reuse the loading logic from SFT3 for consistency
    if os.path.isfile(path):
        if path.endswith(".safetensors"):
            from safetensors.torch import load_file
            return load_file(path)
        return torch.load(path, map_location="cpu")

    if not os.path.isdir(path):
        return None

    safetensors_file = os.path.join(path, "model.safetensors")
    if os.path.exists(safetensors_file):
        from safetensors.torch import load_file
        return load_file(safetensors_file)
        
    bin_file = os.path.join(path, "pytorch_model.bin")
    if os.path.exists(bin_file):
        return torch.load(bin_file, map_location="cpu")
        
    # Check for DeepSpeed ZeRO consolidated bin
    if os.path.exists(os.path.join(path, "pytorch_model.bin")):
        return torch.load(os.path.join(path, "pytorch_model.bin"), map_location="cpu")

    return None


def build_tokenizer(model_cfg: RLModelConfig) -> Any:
    resolved_path = model_cfg.tokenizer_path
    if not os.path.exists(resolved_path):
        fallback_paths = ["/from_s3/sft3_model/", "/from_s3/sft1_model/", "/from_s3/sft2_model/"]
        resolved_path = None
        for p in fallback_paths:
            if os.path.exists(p):
                resolved_path = p
                break
        if not resolved_path:
            raise FileNotFoundError(f"Tokenizer path not found: {model_cfg.tokenizer_path}")

    if _is_rank0():
        print(f"Loading tokenizer from {resolved_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        resolved_path,
        trust_remote_code=model_cfg.trust_remote_code,
        use_fast=True,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_model(model_cfg: RLModelConfig, tokenizer: Any, ds_config: Dict[str, Any]) -> Any:
    config = LLaMAConfig(
        vocab_size=model_cfg.vocab_size or len(tokenizer),
        hidden_size=model_cfg.hidden_size,
        intermediate_size=model_cfg.intermediate_size,
        num_attention_heads=model_cfg.num_attention_heads,
        num_key_value_heads=model_cfg.num_key_value_heads,
        num_hidden_layers=model_cfg.num_hidden_layers,
        max_position_embeddings=model_cfg.max_position_embeddings,
        rope_theta=model_cfg.rope_theta,
        rms_norm_eps=model_cfg.rms_norm_eps,
    )

    zero_stage = ds_config.get("zero_optimization", {}).get("stage", 0)
    use_zero_init = zero_stage == 3

    if model_cfg.checkpoint_path and os.path.exists(model_cfg.checkpoint_path):
        if _is_rank0():
            print(f"Loading weights from {model_cfg.checkpoint_path}")
        model = LLaMAForCausalLM(config)
        state_dict = _load_state_dict_from_checkpoint(model_cfg.checkpoint_path)
        
        if state_dict is not None:
            state_dict = _unwrap_state_dict(state_dict)
            if any(k.startswith("module.") for k in state_dict.keys()):
                state_dict = {k[len("module.") :]: v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=False)
        else:
            if _is_rank0():
                print("Warning: No weights found. Training from scratch.")
        return model

    if use_zero_init:
        with deepspeed.zero.Init(config_dict_or_path=ds_config):
            model = LLaMAForCausalLM(config)
    else:
        model = LLaMAForCausalLM(config)
    return model


def get_log_probs(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate average log-probabilities for the completion portion.
    """
    # Shift so that tokens predict the next token
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_mask = mask[..., 1:].contiguous()
    
    # Calculate per-token log-probs
    # Using cross_entropy with reduction='none' is more memory efficient than log_softmax + gather
    # Since cross_entropy(input, target) = -log_probs(target)
    flat_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_labels = shift_labels.view(-1)
    
    per_token_log_probs = -F.cross_entropy(flat_logits, flat_labels, reduction="none")
    per_token_log_probs = per_token_log_probs.view(shift_labels.shape)
    
    # Mask out tokens we don't care about (prompt and padding)
    masked_log_probs = per_token_log_probs * shift_mask
    
    return masked_log_probs.sum(dim=-1), shift_mask.sum(dim=-1)


def simpo_loss(
    model: Any, 
    batch: Dict[str, torch.Tensor], 
    beta: float, 
    gamma: float, 
    device: torch.device
) -> torch.Tensor:
    # We can run chosen and rejected in one forward pass by concatenating them
    c_ids = batch["chosen_input_ids"].to(device)
    c_mask = batch["chosen_attention_mask"].to(device)
    c_loss_mask = batch["chosen_loss_mask"].to(device)
    
    r_ids = batch["rejected_input_ids"].to(device)
    r_mask = batch["rejected_attention_mask"].to(device)
    r_loss_mask = batch["rejected_loss_mask"].to(device)
    
    # Concatenate for single forward pass
    all_ids = torch.cat([c_ids, r_ids], dim=0)
    all_mask = torch.cat([c_mask, r_mask], dim=0)
    
    outputs = model(input_ids=all_ids, attention_mask=all_mask)
    logits = outputs["logits"]
    
    # Split back
    logits_c, logits_r = logits.chunk(2, dim=0)
    
    log_probs_c, len_c = get_log_probs(logits_c, c_ids, c_loss_mask)
    log_probs_r, len_r = get_log_probs(logits_r, r_ids, r_loss_mask)
    
    # SimPO objective: length-normalized log-probs
    p_w = log_probs_c / len_c
    p_l = log_probs_r / len_r
    
    # SimPO margin loss
    simpo_logits = beta * (p_w - p_l) - gamma
    loss = -F.logsigmoid(simpo_logits).mean()
    
    return loss


@torch.no_grad()
def run_eval(model_engine: Any, val_loader: Any, beta: float, gamma: float, device: torch.device, max_batches: int) -> float:
    model_engine.eval()
    total_loss = torch.tensor(0.0, device=device)
    count = torch.tensor(0.0, device=device)

    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=device.type == "cuda"):
            loss = simpo_loss(model_engine, batch, beta, gamma, device)
        
        total_loss += loss
        count += 1

    torch.distributed.all_reduce(total_loss, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(count, op=torch.distributed.ReduceOp.SUM)
    return (total_loss / count).item() if count.item() > 0 else 0.0


def train_loop(
    model: Any, 
    data: RLDataModule, 
    train_cfg: RLTrainerConfig, 
    ds_config: Dict[str, Any], 
    logger: Optional[Any]
) -> None:
    train_loader = data.train_dataloader()
    val_loader = data.val_dataloader()

    model_engine, _, _, _ = deepspeed.initialize(model=model, config=ds_config)
    device = model_engine.device
    
    if hasattr(model_engine.module.model, "gradient_checkpointing"):
        model_engine.module.model.gradient_checkpointing = True

    global_step = 0
    best_val_loss = float("inf")
    
    model_engine.train()

    for epoch in range(train_cfg.max_epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", disable=not _is_rank0())
        for batch in pbar:
            if global_step >= train_cfg.max_steps:
                break
                
            loss = simpo_loss(model_engine, batch, train_cfg.beta, train_cfg.gamma, device)
            
            model_engine.backward(loss)
            model_engine.step()

            if not model_engine.is_gradient_accumulation_boundary():
                continue
                
            global_step += 1
            
            if _is_rank0():
                pbar.set_postfix({"loss": loss.item(), "step": global_step})
                if logger and global_step % train_cfg.log_interval == 0:
                    logger.report_scalar("Train", "SimPO Loss", value=loss.item(), iteration=global_step)
                    logger.report_scalar("Train", "LR", value=model_engine.get_lr()[0], iteration=global_step)

            if global_step % train_cfg.eval_interval == 0:
                val_loss = run_eval(model_engine, val_loader, train_cfg.beta, train_cfg.gamma, device, train_cfg.max_eval_batches)
                if _is_rank0():
                    print(f"Step {global_step}: Val Loss = {val_loss}")
                    if logger:
                        logger.report_scalar("Eval", "SimPO Loss", value=val_loss, iteration=global_step)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    if train_cfg.save_best_only:
                        model_engine.save_checkpoint(train_cfg.output_dir, tag="best")
                
                model_engine.train()

    if train_cfg.save_final:
        model_engine.save_checkpoint(train_cfg.output_dir, tag="final")


def main() -> None:
    parser = HfArgumentParser((RLModelConfig, RLDataConfig, RLTrainerConfig))
    model_cfg, data_cfg, train_cfg = parser.parse_args_into_dataclasses()

    _setup_local_cuda_device()
    if not torch.distributed.is_initialized():
        deepspeed.init_distributed(timeout=timedelta(hours=2))

    logger = None
    if _is_rank0():
        task = Task.init(project_name="LLaMA_RL", task_name="SimPO_RLHF", reuse_last_task_id=False)
        logger = task.get_logger()

    with open(train_cfg.ds_config_path, "r") as f:
        ds_config = json.load(f)

    # Manual overrides from configs
    if "optimizer" in ds_config and "params" in ds_config["optimizer"]:
        ds_config["optimizer"]["params"]["lr"] = train_cfg.learning_rate
        
    tokenizer = build_tokenizer(model_cfg)
    model = build_model(model_cfg, tokenizer, ds_config)
    
    # Data prep if needed
    status_path = os.path.join(data_cfg.data_root, ".prep_status_rl.json")
    if not os.path.exists(os.path.join(data_cfg.data_root, "train.jsonl")) and _is_rank0():
        print("Preparing RL data...")
        prep_cfg = RLDataPrepConfig(
            output_dir=data_cfg.data_root,
            seed=data_cfg.seed,
            max_length=data_cfg.max_length,
            max_prompt_length=data_cfg.max_prompt_length,
            tokenizer_path=model_cfg.tokenizer_path,
            ultrafeedback_ratio=data_cfg.ultrafeedback_ratio,
            hh_rlhf_ratio=data_cfg.hh_rlhf_ratio,
        )
        prepare_rl_data(prep_cfg)
        with open(status_path, "w") as f:
            json.dump({"ok": True}, f)
            
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        
    data = RLDataModule(
        train_files=[os.path.join(data_cfg.data_root, "train.jsonl")],
        val_files=[os.path.join(data_cfg.data_root, "val.jsonl")],
        tokenizer=tokenizer,
        max_length=data_cfg.max_length,
        max_prompt_length=data_cfg.max_prompt_length,
        batch_size=data_cfg.batch_size,
        num_workers=data_cfg.num_workers,
        seed=data_cfg.seed,
        add_eos=data_cfg.add_eos,
    )
    data.prepare()

    train_loop(model, data, train_cfg, ds_config, logger)


if __name__ == "__main__":
    main()
