import os
import json
import torch
import deepspeed
from typing import Any, Dict, List, Optional
from transformers import AutoTokenizer, HfArgumentParser
import argparse
from tqdm import tqdm
from clearml import Task
from datetime import timedelta

from sft.sft_1.sft_config import SFTModelConfig, SFTDataConfig, SFTTrainerConfig
from sft.sft_1.sft_dataset import SFTDataModule
from model.model import LLaMAForCausalLM
from config.training_config import LLaMAConfig


def build_tokenizer(model_cfg: SFTModelConfig) -> Any:
    tokenizer_path = model_cfg.tokenizer_path
    if tokenizer_path and os.path.exists(tokenizer_path):
        resolved_path = tokenizer_path
    elif model_cfg.checkpoint_path and os.path.exists(model_cfg.checkpoint_path):
        resolved_path = model_cfg.checkpoint_path
        if (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0:
            print(
                f"Tokenizer path not found at '{tokenizer_path}'. "
                f"Falling back to model directory: {resolved_path}"
            )
    else:
        raise FileNotFoundError(
            f"Tokenizer path not found: {tokenizer_path} and "
            f"model path not found: {model_cfg.checkpoint_path}"
        )

    if (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0:
        print(f"Loading tokenizer from {resolved_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            resolved_path,
            trust_remote_code=model_cfg.trust_remote_code,
            use_fast=True,
            local_files_only=True,
        )
    except Exception as e:
        # Fallback for incompatible/invalid fast tokenizer.json
        if (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0:
            print(f"Warning: Fast tokenizer failed to load ({e}). Retrying with use_fast=False.")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                resolved_path,
                trust_remote_code=model_cfg.trust_remote_code,
                use_fast=False,
                local_files_only=True,
            )
        except Exception as e2:
            raise RuntimeError(
                "Failed to load tokenizer. The tokenizer files may be corrupted or "
                "require a newer tokenizers/transformers version. "
                "Please re-sync the tokenizer directory or update dependencies."
            ) from e2
    if tokenizer.pad_token is None:
        if tokenizer.unk_token:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_model(model_cfg: SFTModelConfig, tokenizer: Any, ds_config: Dict[str, Any]) -> Any:
    # Use architecture parameters from SFTModelConfig
    config = LLaMAConfig(
        vocab_size=model_cfg.vocab_size or len(tokenizer),
        hidden_size=model_cfg.hidden_size,
        intermediate_size=model_cfg.intermediate_size,
        num_attention_heads=model_cfg.num_attention_heads,
        num_key_value_heads=model_cfg.num_key_value_heads,
        num_hidden_layers=model_cfg.num_hidden_layers,
        max_position_embeddings=model_cfg.max_position_embeddings,
        rope_theta=model_cfg.rope_theta,
    )

    zero_stage = ds_config.get("zero_optimization", {}).get("stage", 0)
    use_zero_init = zero_stage == 3

    if model_cfg.checkpoint_path and os.path.exists(model_cfg.checkpoint_path):
        if torch.distributed.get_rank() == 0:
            print(f"Loading model weights from {model_cfg.checkpoint_path}")
        model = LLaMAForCausalLM(config)
        if os.path.isdir(model_cfg.checkpoint_path):
            from safetensors.torch import load_file as load_safetensors
            for f in os.listdir(model_cfg.checkpoint_path):
                if f.endswith(".safetensors"):
                    state_dict = load_safetensors(os.path.join(model_cfg.checkpoint_path, f))
                    model.load_state_dict(state_dict, strict=False)
                elif f.endswith(".bin") or f.endswith(".pt"):
                    model.load_state_dict(torch.load(os.path.join(model_cfg.checkpoint_path, f), map_location="cpu"), strict=False)
        else:
            model.load_state_dict(torch.load(model_cfg.checkpoint_path, map_location="cpu"), strict=False)
        return model

    if model_cfg.checkpoint_path and not os.path.exists(model_cfg.checkpoint_path):
        print(f"Warning: Checkpoint path not found: {model_cfg.checkpoint_path}. Starting from scratch.")

    if use_zero_init:
        with deepspeed.zero.Init(config_dict_or_path=ds_config):
            model = LLaMAForCausalLM(config)
    else:
        model = LLaMAForCausalLM(config)

    return model


def build_datamodule(data_cfg: SFTDataConfig, tokenizer: Any) -> SFTDataModule:
    import glob
    from sft.sft_1.prepare_data import prepare_sft_data, DataPrepConfig
    
    if not os.path.exists(data_cfg.data_root) and torch.distributed.get_rank() == 0:
        os.makedirs(data_cfg.data_root, exist_ok=True)
        
    train_files = glob.glob(os.path.join(data_cfg.data_root, data_cfg.train_glob))
    val_files = glob.glob(os.path.join(data_cfg.data_root, data_cfg.val_glob))
    
    if not train_files and torch.distributed.get_rank() == 0:
        print("Data not found. Running data preparation...")
        prep_cfg = DataPrepConfig(
            output_dir=data_cfg.data_root,
            chat_template=data_cfg.chat_template,
            seed=data_cfg.seed,
            val_ratio=data_cfg.val_ratio,
        )
        prepare_sft_data(prep_cfg)
        train_files = glob.glob(os.path.join(data_cfg.data_root, data_cfg.train_glob))
        val_files = glob.glob(os.path.join(data_cfg.data_root, data_cfg.val_glob))

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        train_files = glob.glob(os.path.join(data_cfg.data_root, data_cfg.train_glob))
        val_files = glob.glob(os.path.join(data_cfg.data_root, data_cfg.val_glob)) # MUST FIX 1

    # MUST FIX 4: Handle cases where val_glob matches nothing
    if not val_files and train_files:
        print("Warning: No validation files found. Using a slice of training data for validation.")
        val_files = train_files[:1]

    if not train_files:
        raise FileNotFoundError(f"No training files found at {os.path.join(data_cfg.data_root, data_cfg.train_glob)}")
    
    dm = SFTDataModule(
        train_files=train_files,
        val_files=val_files,
        tokenizer=tokenizer,
        max_length=data_cfg.max_length,
        batch_size=data_cfg.batch_size,
        num_workers=data_cfg.num_workers,
        seed=data_cfg.seed,
        text_field=data_cfg.text_field,
        add_eos=data_cfg.add_eos,
        mask_user=data_cfg.mask_user,
        chat_template=data_cfg.chat_template,
        pack=True
    )
    dm.prepare()
    return dm


def train_loop(
    model: Any,
    data: SFTDataModule,
    train_cfg: SFTTrainerConfig,
    ds_config: Dict[str, Any],
    logger: Optional[Any] = None,
) -> None:
    train_loader = data.train_dataloader()
    val_loader = data.val_dataloader()
    
    model_engine, _, _, _ = deepspeed.initialize(
        model=model,
        config=ds_config,
    )
    
    device = model_engine.device
    global_step = 0
    best_val_loss = float("inf")
    
    if hasattr(model_engine.module.model, "gradient_checkpointing"):
        model_engine.module.model.gradient_checkpointing = True
    
    model_engine.train()
    
    max_steps = train_cfg.max_steps
    if "scheduler" in ds_config and "params" in ds_config["scheduler"]:
        ds_total_steps = ds_config["scheduler"]["params"].get("total_num_steps")
        if isinstance(ds_total_steps, int):
            max_steps = ds_total_steps

    for epoch in range(20): 
        if hasattr(data, "set_epoch"):
            data.set_epoch(epoch)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", disable=torch.distributed.get_rank() != 0)
        for batch in pbar:
            if global_step >= max_steps:
                break
                
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model_engine(input_ids=input_ids, labels=labels)
            loss = outputs["loss"]
            
            model_engine.backward(loss)
            model_engine.step()

            # P2: Track steps only on optimizer boundaries
            if not model_engine.is_gradient_accumulation_boundary():
                continue

            global_step += 1

            if torch.distributed.get_rank() == 0:
                pbar.set_postfix({"loss": loss.item(), "step": global_step})
                if logger and global_step % train_cfg.log_interval == 0:
                    import math
                    ppl = math.exp(min(loss.item(), 20)) # SHOULD FIX 6
                    logger.report_scalar("Train", "Loss", value=loss.item(), iteration=global_step)
                    logger.report_scalar("Train", "Perplexity", value=ppl, iteration=global_step)
                    logger.report_scalar("Train", "LR", value=model_engine.get_lr()[0], iteration=global_step)

            if not train_cfg.save_best_only and train_cfg.save_interval > 0:
                if global_step % train_cfg.save_interval == 0:
                    model_engine.save_checkpoint(train_cfg.output_dir, tag=f"step-{global_step}")

            if global_step % train_cfg.eval_interval == 0:
                val_loss = run_eval(model_engine, val_loader, device)
                if torch.distributed.get_rank() == 0:
                    print(f"Step {global_step}: Val Loss = {val_loss}")
                    if logger:
                        logger.report_scalar("Eval", "Loss", value=val_loss, iteration=global_step)
                        import math
                        val_ppl = math.exp(min(val_loss, 20))
                        logger.report_scalar("Eval", "Perplexity", value=val_ppl, iteration=global_step)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    if train_cfg.save_best_only:
                        # Use fixed tag "best" to overwrite and save disk space
                        model_engine.save_checkpoint(train_cfg.output_dir, tag="best")
                        if torch.distributed.get_rank() == 0:
                            convert_to_hf(train_cfg.output_dir, "best", model_engine.module.config, data.tokenizer)

                model_engine.train()

    if train_cfg.save_final:
        model_engine.save_checkpoint(train_cfg.output_dir, tag="final")
        if torch.distributed.get_rank() == 0:
            convert_to_hf(train_cfg.output_dir, "final", model_engine.module.config, data.tokenizer)


def convert_to_hf(output_dir: str, tag: str, config: Any, tokenizer: Any):
    """SHOULD FIX 5: Conversion logic to make checkpoints compatible with Transformers."""
    from safetensors.torch import save_file
    import subprocess
    import sys
    
    save_dir = os.path.join(output_dir, tag)
    bin_path = None
    if not os.path.exists(save_dir):
        return

    try:
        # For ZeRO-3: consolidate if needed
        zero_to_fp32 = os.path.join(save_dir, "zero_to_fp32.py")
        fp32_path = os.path.join(save_dir, "pytorch_model.bin")
        if os.path.exists(zero_to_fp32) and not os.path.exists(fp32_path):
            subprocess.run([sys.executable, zero_to_fp32, save_dir, fp32_path], check=True)

        # Prefer consolidated model weights if present
        preferred = ["pytorch_model.bin", "model.bin", "model.pt"]
        for name in preferred:
            cand = os.path.join(save_dir, name)
            if os.path.exists(cand):
                bin_path = cand
                break
        if not bin_path:
            for f in os.listdir(save_dir):
                if f.endswith(".bin") or f.endswith(".pt"):
                    bin_path = os.path.join(save_dir, f)
                    break

        if not bin_path:
            print(f"No consolidated weight file found in {save_dir}. Skipping HF conversion.")
            return

        print(f"Converting {tag} checkpoint to HF format...")
        state_dict = torch.load(bin_path, map_location="cpu")

        # DeepSpeed checkpoints may wrap weights in a nested dict
        if isinstance(state_dict, dict):
            if "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
                state_dict = state_dict["state_dict"]
            elif "module" in state_dict and isinstance(state_dict["module"], dict):
                state_dict = state_dict["module"]
            elif "model" in state_dict and isinstance(state_dict["model"], dict):
                state_dict = state_dict["model"]

        hf_state_dict = {}
        for key, tensor in state_dict.items():
            if not torch.is_tensor(tensor):
                # Skip non-parameter entries (e.g., optimizer state)
                continue
            new_key = key
            if new_key.startswith("module."):
                new_key = new_key[len("module.") :]
            if new_key.startswith("model."):
                new_key = new_key[6:]

            new_key = new_key.replace("attention.q_proj", "self_attn.q_proj")
            new_key = new_key.replace("attention.k_proj", "self_attn.k_proj")
            new_key = new_key.replace("attention.v_proj", "self_attn.v_proj")
            new_key = new_key.replace("attention.o_proj", "self_attn.o_proj")
            new_key = new_key.replace("attention_norm", "input_layernorm")
            new_key = new_key.replace("ffn_norm", "post_attention_layernorm")

            if "feed_forward.gate_up_proj" in new_key:
                intermediate_size = tensor.shape[0] // 2
                gate_key = new_key.replace("feed_forward.gate_up_proj", "mlp.gate_proj")
                up_key = new_key.replace("feed_forward.gate_up_proj", "mlp.up_proj")
                hf_state_dict[gate_key] = tensor[:intermediate_size].clone()
                hf_state_dict[up_key] = tensor[intermediate_size:].clone()
            elif "feed_forward.down_proj" in new_key:
                new_key = new_key.replace("feed_forward.down_proj", "mlp.down_proj")
                hf_state_dict[new_key] = tensor
            else:
                hf_state_dict[new_key] = tensor

        save_file(hf_state_dict, os.path.join(save_dir, "model.safetensors"))

        hf_config = {
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "hidden_size": config.hidden_size,
            "intermediate_size": config.intermediate_size,
            "num_attention_heads": config.num_attention_heads,
            "num_hidden_layers": config.num_hidden_layers,
            "num_key_value_heads": config.num_key_value_heads,
            "vocab_size": config.vocab_size,
            "rms_norm_eps": config.rms_norm_eps,
            "max_position_embeddings": config.max_position_embeddings,
            "rope_theta": config.rope_theta,
            "torch_dtype": "bf16"
        }
        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump(hf_config, f, indent=2)

        tokenizer.save_pretrained(save_dir)
        print(f"HF checkpoint saved to {save_dir}")
    except Exception as e:
        print(f"Warning: HF conversion failed for tag '{tag}': {e}")


@torch.no_grad()
def run_eval(model_engine: Any, val_loader: Any, device: torch.device) -> float:
    """Robust evaluation averaging by total loss and total tokens/samples."""
    model_engine.eval()
    total_loss = torch.tensor(0.0, device=device)
    total_tokens = torch.tensor(0.0, device=device)
    
    for i, batch in enumerate(val_loader):
        if i >= 50: # Limit eval for speed
            break
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=device.type == "cuda"):
            outputs = model_engine(input_ids=input_ids, labels=labels)
        loss = outputs["loss"]
        token_count = (labels[..., 1:] != -100).sum()
        total_loss += loss * token_count
        total_tokens += token_count
    
    # Collective sum across all ranks
    torch.distributed.all_reduce(total_loss, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(total_tokens, op=torch.distributed.ReduceOp.SUM)
    
    if total_tokens.item() == 0:
        return 0.0
    return (total_loss / total_tokens).item()


def main() -> None:
    parser = HfArgumentParser((SFTModelConfig, SFTDataConfig, SFTTrainerConfig))
    model_cfg, data_cfg, train_cfg = parser.parse_args_into_dataclasses()
    
    if not torch.distributed.is_initialized():
        deepspeed.init_distributed(timeout=timedelta(hours=2))
    
    global_rank = torch.distributed.get_rank()
    
    logger = None
    if global_rank == 0:
        task = Task.init(
            project_name="LLaMA_SFT",
            task_name="SFT_Stage_1_Qwen_ChatML",
            reuse_last_task_id=False,
        )
        task.connect(model_cfg, name="ModelConfig")
        task.connect(data_cfg, name="DataConfig")
        task.connect(train_cfg, name="TrainerConfig")
        logger = task.get_logger()

    if not os.path.exists(train_cfg.ds_config_path):
        raise FileNotFoundError(f"DeepSpeed config not found: {train_cfg.ds_config_path}")
        
    with open(train_cfg.ds_config_path, "r") as f:
        ds_config = json.load(f)
    
    if "train_micro_batch_size_per_gpu" in ds_config:
        data_cfg.batch_size = ds_config["train_micro_batch_size_per_gpu"]
    
    tokenizer = build_tokenizer(model_cfg)
    model = build_model(model_cfg, tokenizer, ds_config)
    data = build_datamodule(data_cfg, tokenizer)
    
    train_loop(model, data, train_cfg, ds_config, logger)


if __name__ == "__main__":
    main()
