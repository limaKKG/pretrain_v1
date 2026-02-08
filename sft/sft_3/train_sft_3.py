import os
import json
import torch
import deepspeed
from typing import Any, Dict, Optional
from transformers import AutoTokenizer, HfArgumentParser
from tqdm import tqdm
from clearml import Task
from datetime import timedelta

from sft.sft_3.sft_config_3 import SFTModelConfig, SFTDataConfig, SFTTrainerConfig
from sft.sft_1.sft_dataset import SFTDataModule
from sft.sft_3.prepare_data_sft3 import prepare_sft3_data, SFT3DataPrepConfig
from model.model import LLaMAForCausalLM
from config.training_config import LLaMAConfig


def _is_rank0() -> bool:
    return (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0


def _setup_local_cuda_device() -> None:
    """
    Fix NCCL 'Duplicate GPU detected' by binding each rank to its local GPU.
    Must run BEFORE any NCCL collectives.
    """
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


def _load_sharded_safetensors(index_path: str) -> Dict[str, Any]:
    from safetensors.torch import load_file
    with open(index_path, "r") as f:
        index = json.load(f)
    weight_map = index.get("weight_map", {})
    shard_files = sorted(set(weight_map.values()))
    state_dict: Dict[str, Any] = {}
    for shard in shard_files:
        shard_path = os.path.join(os.path.dirname(index_path), shard)
        if not os.path.exists(shard_path):
            continue
        state_dict.update(load_file(shard_path))
    return state_dict


def _load_sharded_bin(index_path: str) -> Dict[str, Any]:
    with open(index_path, "r") as f:
        index = json.load(f)
    weight_map = index.get("weight_map", {})
    shard_files = sorted(set(weight_map.values()))
    state_dict: Dict[str, Any] = {}
    for shard in shard_files:
        shard_path = os.path.join(os.path.dirname(index_path), shard)
        if not os.path.exists(shard_path):
            continue
        state_dict.update(torch.load(shard_path, map_location="cpu"))
    return state_dict


def _load_state_dict_from_checkpoint(path: str) -> Optional[Dict[str, Any]]:
    if os.path.isfile(path):
        if path.endswith(".safetensors"):
            from safetensors.torch import load_file
            return load_file(path)
        return torch.load(path, map_location="cpu")

    if not os.path.isdir(path):
        return None

    candidate_dirs = [path, os.path.join(path, "best")]
    for d in candidate_dirs:
        if not os.path.isdir(d):
            continue
        safetensors_index = os.path.join(d, "model.safetensors.index.json")
        if os.path.exists(safetensors_index):
            return _load_sharded_safetensors(safetensors_index)
        safetensors_file = os.path.join(d, "model.safetensors")
        if os.path.exists(safetensors_file):
            from safetensors.torch import load_file
            return load_file(safetensors_file)
        bin_index = os.path.join(d, "pytorch_model.bin.index.json")
        if os.path.exists(bin_index):
            return _load_sharded_bin(bin_index)
        bin_file = os.path.join(d, "pytorch_model.bin")
        if os.path.exists(bin_file):
            return torch.load(bin_file, map_location="cpu")
    return None


def build_tokenizer(model_cfg: SFTModelConfig) -> Any:
    tokenizer_path = model_cfg.tokenizer_path
    if tokenizer_path and os.path.exists(tokenizer_path):
        resolved_path = tokenizer_path
    elif model_cfg.checkpoint_path and os.path.exists(model_cfg.checkpoint_path):
        resolved_path = model_cfg.checkpoint_path
    else:
        fallback_paths = ["/from_s3/sft2_model/", "/from_s3/sft1_model/"]
        resolved_path = None
        for p in fallback_paths:
            if os.path.exists(p):
                resolved_path = p
                break
        if not resolved_path:
            raise FileNotFoundError(
                f"Tokenizer path not found: {tokenizer_path}, {model_cfg.checkpoint_path}, or fallback paths."
            )

    if _is_rank0():
        print(f"Loading tokenizer from {resolved_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            resolved_path,
            trust_remote_code=model_cfg.trust_remote_code,
            use_fast=True,
            local_files_only=True,
        )
    except Exception as e:
        if _is_rank0():
            print(f"Warning: Fast tokenizer failed to load ({e}). Retrying with use_fast=False.")
        tokenizer = AutoTokenizer.from_pretrained(
            resolved_path,
            trust_remote_code=model_cfg.trust_remote_code,
            use_fast=False,
            local_files_only=True,
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token or tokenizer.eos_token
    return tokenizer


def build_model(model_cfg: SFTModelConfig, tokenizer: Any, ds_config: Dict[str, Any]) -> Any:
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
            print(f"Loading SFT Stage 2 weights from {model_cfg.checkpoint_path}")
        model = LLaMAForCausalLM(config)
        state_dict = _load_state_dict_from_checkpoint(model_cfg.checkpoint_path)

        # Support DeepSpeed ZeRO checkpoint directories that contain only
        # `zero_pp_rank_*_model_states.pt` shards (no consolidated safetensors/bin).
        if state_dict is None and os.path.isdir(model_cfg.checkpoint_path):
            try:
                from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
                if _is_rank0():
                    print("Detected non-HF DeepSpeed checkpoint. Consolidating ZeRO shards to FP32 state_dict...")
                # Deepspeed expects a 'latest' file in checkpoint root. Some pipelines
                # upload only the tag directory contents to S3 (no latest). In that case,
                # create a temporary wrapper with symlinks and a latest file.
                ckpt_root = model_cfg.checkpoint_path
                latest_path = os.path.join(ckpt_root, "latest")
                if not os.path.exists(latest_path):
                    import tempfile
                    wrapper = tempfile.mkdtemp(prefix="ds_ckpt_wrap_")
                    tag_dir = os.path.join(wrapper, "best")
                    os.makedirs(tag_dir, exist_ok=True)
                    # link all shard files into wrapper/best/
                    for fname in os.listdir(ckpt_root):
                        if fname.endswith(".pt") or fname.endswith(".bin") or fname.endswith(".safetensors"):
                            src = os.path.join(ckpt_root, fname)
                            dst = os.path.join(tag_dir, fname)
                            try:
                                os.symlink(src, dst)
                            except FileExistsError:
                                pass
                    with open(os.path.join(wrapper, "latest"), "w") as f:
                        f.write("best")
                    ckpt_root = wrapper
                state_dict = get_fp32_state_dict_from_zero_checkpoint(ckpt_root)
            except Exception as e:
                if _is_rank0():
                    print(f"Warning: Failed to consolidate DeepSpeed checkpoint: {e}")
                state_dict = None

        if state_dict is None:
            if _is_rank0():
                print("Warning: No weights found in checkpoint_path. Starting from initial state.")
            return model

        state_dict = _unwrap_state_dict(state_dict)
        # Deepspeed fp32 state dict can be prefixed with "module."
        if isinstance(state_dict, dict) and any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k[len("module.") :]: v for k, v in state_dict.items()}

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if _is_rank0():
            print(f"Loaded weights. Missing keys: {len(missing)}, unexpected: {len(unexpected)}")
        return model

    if use_zero_init:
        with deepspeed.zero.Init(config_dict_or_path=ds_config):
            model = LLaMAForCausalLM(config)
    else:
        model = LLaMAForCausalLM(config)
    return model


def build_datamodule(data_cfg: SFTDataConfig, tokenizer: Any, tokenizer_path: str) -> SFTDataModule:
    import glob

    if not os.path.exists(data_cfg.data_root) and _is_rank0():
        os.makedirs(data_cfg.data_root, exist_ok=True)

    train_files = glob.glob(os.path.join(data_cfg.data_root, data_cfg.train_glob))
    val_files = glob.glob(os.path.join(data_cfg.data_root, data_cfg.val_glob))

    status_path = os.path.join(data_cfg.data_root, ".prep_status_sft3.json")
    prep_timeout_s = int(os.environ.get("SFT_DATA_PREP_TIMEOUT_S", "21600"))  # 6h
    poll_s = float(os.environ.get("SFT_DATA_PREP_POLL_S", "5"))

    if not train_files and _is_rank0():
        print("SFT Stage 3 data not found. Running calibration data preparation...")
        try:
            prep_cfg = SFT3DataPrepConfig(
                output_dir=data_cfg.data_root,
                seed=data_cfg.seed,
                val_ratio=data_cfg.val_ratio,
                max_length=data_cfg.max_length,
                add_eos=data_cfg.add_eos,
                tokenizer_path=tokenizer_path,
                no_robots_ratio=0.60,
                lima_ratio=0.15,
                stage2_replay_ratio=0.25,
                stage2_data_root="data/sft2",
            )
            prepare_sft3_data(prep_cfg)
            with open(status_path, "w") as f:
                json.dump({"ok": True, "error": None}, f)
        except Exception as e:
            print(f"CRITICAL: Stage3 data preparation failed on rank 0: {e}")
            try:
                with open(status_path, "w") as f:
                    json.dump({"ok": False, "error": str(e)}, f)
            finally:
                pass

    # Avoid NCCL collectives while rank0 is doing long data prep; poll a shared status file.
    if not train_files:
        import time
        start = time.time()
        while True:
            if os.path.exists(status_path):
                with open(status_path, "r") as f:
                    status = json.load(f)
                if not status.get("ok", False):
                    raise RuntimeError(f"Stage3 data preparation failed on rank 0: {status.get('error')}")
                break
            if time.time() - start > prep_timeout_s:
                raise TimeoutError(f"Timed out waiting for stage3 data prep status file: {status_path}")
            time.sleep(poll_s)

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        train_files = glob.glob(os.path.join(data_cfg.data_root, data_cfg.train_glob))
        val_files = glob.glob(os.path.join(data_cfg.data_root, data_cfg.val_glob))

    if not val_files and train_files:
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
        pack=True,
    )
    dm.prepare()
    return dm


@torch.no_grad()
def run_eval(model_engine: Any, val_loader: Any, device: torch.device, max_batches: int) -> float:
    model_engine.eval()
    total_loss = torch.tensor(0.0, device=device)
    total_tokens = torch.tensor(0.0, device=device)

    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=device.type == "cuda"):
            outputs = model_engine(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
        loss = outputs["loss"]
        token_count = (labels[..., 1:] != -100).sum()
        total_loss += loss * token_count
        total_tokens += token_count

    torch.distributed.all_reduce(total_loss, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(total_tokens, op=torch.distributed.ReduceOp.SUM)
    return (total_loss / total_tokens).item() if total_tokens.item() > 0 else 0.0


def convert_to_hf(output_dir: str, tag: str, config: Any, tokenizer: Any):
    from safetensors.torch import save_file
    import subprocess
    import sys

    save_dir = os.path.join(output_dir, tag)
    if not os.path.exists(save_dir):
        return

    try:
        zero_to_fp32 = os.path.join(save_dir, "zero_to_fp32.py")
        fp32_path = os.path.join(save_dir, "pytorch_model.bin")
        if os.path.exists(zero_to_fp32) and not os.path.exists(fp32_path):
            subprocess.run([sys.executable, zero_to_fp32, save_dir, fp32_path], check=True)

        bin_path = None
        for name in ["pytorch_model.bin", "model.bin", "model.pt"]:
            cand = os.path.join(save_dir, name)
            if os.path.exists(cand):
                bin_path = cand
                break
        if not bin_path:
            return

        state_dict = torch.load(bin_path, map_location="cpu")
        state_dict = _unwrap_state_dict(state_dict)

        hf_state_dict = {}
        for key, tensor in state_dict.items():
            if not torch.is_tensor(tensor):
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
                hf_state_dict[new_key.replace("feed_forward.gate_up_proj", "mlp.gate_proj")] = tensor[:intermediate_size].clone()
                hf_state_dict[new_key.replace("feed_forward.gate_up_proj", "mlp.up_proj")] = tensor[intermediate_size:].clone()
            elif "feed_forward.down_proj" in new_key:
                hf_state_dict[new_key.replace("feed_forward.down_proj", "mlp.down_proj")] = tensor
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
            "rms_norm_eps": getattr(config, "rms_norm_eps", 1e-5),
            "max_position_embeddings": config.max_position_embeddings,
            "rope_theta": config.rope_theta,
            "torch_dtype": "bf16",
        }
        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump(hf_config, f, indent=2)

        tokenizer.save_pretrained(save_dir)
        print(f"HF checkpoint saved to {save_dir}")
    except Exception as e:
        print(f"Warning: HF conversion failed: {e}")


def train_loop(model: Any, data: SFTDataModule, train_cfg: SFTTrainerConfig, ds_config: Dict[str, Any], logger: Optional[Any]) -> None:
    train_loader = data.train_dataloader()
    val_loader = data.val_dataloader()

    model_engine, _, _, _ = deepspeed.initialize(model=model, config=ds_config)

    device = model_engine.device
    global_step = 0
    best_val_loss = float("inf")

    if hasattr(model_engine.module.model, "gradient_checkpointing"):
        model_engine.module.model.gradient_checkpointing = True

    model_engine.train()
    max_steps = train_cfg.max_steps

    for epoch in range(train_cfg.max_epochs):
        if hasattr(data, "set_epoch"):
            data.set_epoch(epoch)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", disable=not _is_rank0())
        for batch in pbar:
            if global_step >= max_steps:
                break
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model_engine(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
            loss = outputs["loss"]

            model_engine.backward(loss)
            model_engine.step()

            if not model_engine.is_gradient_accumulation_boundary():
                continue
            global_step += 1

            if _is_rank0():
                pbar.set_postfix({"loss": loss.item(), "step": global_step})
                if logger and global_step % train_cfg.log_interval == 0:
                    import math
                    logger.report_scalar("Train", "Loss", value=loss.item(), iteration=global_step)
                    logger.report_scalar("Train", "Perplexity", value=math.exp(min(loss.item(), 20)), iteration=global_step)
                    logger.report_scalar("Train", "LR", value=model_engine.get_lr()[0], iteration=global_step)

            if global_step % train_cfg.eval_interval == 0:
                val_loss = run_eval(model_engine, val_loader, device, train_cfg.max_eval_batches)
                if _is_rank0():
                    print(f"Step {global_step}: Val Loss = {val_loss}")
                    if logger:
                        import math
                        logger.report_scalar("Eval", "Loss", value=val_loss, iteration=global_step)
                        logger.report_scalar("Eval", "Perplexity", value=math.exp(min(val_loss, 20)), iteration=global_step)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    if train_cfg.save_best_only:
                        model_engine.save_checkpoint(train_cfg.output_dir, tag="best")
                        if _is_rank0():
                            convert_to_hf(train_cfg.output_dir, "best", model_engine.module.config, data.tokenizer)

                model_engine.train()

    if train_cfg.save_final:
        model_engine.save_checkpoint(train_cfg.output_dir, tag="final")
        if _is_rank0():
            convert_to_hf(train_cfg.output_dir, "final", model_engine.module.config, data.tokenizer)


def main() -> None:
    parser = HfArgumentParser((SFTModelConfig, SFTDataConfig, SFTTrainerConfig))
    model_cfg, data_cfg, train_cfg = parser.parse_args_into_dataclasses()

    if data_cfg.max_length > model_cfg.max_position_embeddings:
        raise ValueError(
            f"data_cfg.max_length ({data_cfg.max_length}) exceeds model_cfg.max_position_embeddings ({model_cfg.max_position_embeddings})."
        )

    # Bind CUDA device per-rank before initializing distributed/NCCL.
    _setup_local_cuda_device()

    if not torch.distributed.is_initialized():
        deepspeed.init_distributed(timeout=timedelta(hours=2))

    logger = None
    if _is_rank0():
        task = Task.init(project_name="LLaMA_SFT", task_name="SFT_Stage_3_Calibration", reuse_last_task_id=False)
        task.connect(model_cfg, name="ModelConfig")
        task.connect(data_cfg, name="DataConfig")
        task.connect(train_cfg, name="TrainerConfig")
        logger = task.get_logger()

    if not os.path.exists(train_cfg.ds_config_path):
        raise FileNotFoundError(f"DeepSpeed config not found: {train_cfg.ds_config_path}")
    with open(train_cfg.ds_config_path, "r") as f:
        ds_config = json.load(f)

    # Ensure scheduler matches the requested max_steps
    if isinstance(ds_config.get("scheduler", {}).get("params", {}).get("total_num_steps"), int):
        ds_config["scheduler"]["params"]["total_num_steps"] = int(train_cfg.max_steps)

    if "train_micro_batch_size_per_gpu" in ds_config:
        data_cfg.batch_size = ds_config["train_micro_batch_size_per_gpu"]

    tokenizer = build_tokenizer(model_cfg)
    model = build_model(model_cfg, tokenizer, ds_config)
    data = build_datamodule(data_cfg, tokenizer, tokenizer_path=model_cfg.tokenizer_path)

    train_loop(model, data, train_cfg, ds_config, logger)


if __name__ == "__main__":
    main()

