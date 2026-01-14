from __future__ import annotations
import os
import json
import torch
import deepspeed
from typing import Any, Optional
from transformers import AutoTokenizer
from config.training_config import (
    DataConfig,
    LLaMAConfig,
    TrainerConfig
)
from data.dataset import DataModule
from model.model import LLaMAForCausalLM, LLaMAModelWrapper
from trainer.trainer import LLMTrainer
from clearml import Task


def build_model(model_cfg: LLaMAConfig, tokenizer: Any) -> LLaMAModelWrapper:
    llama_config = LLaMAConfig(
        vocab_size=len(tokenizer),
        hidden_size=model_cfg.hidden_size,
        intermediate_size=model_cfg.intermediate_size,
        num_attention_heads=model_cfg.num_attention_heads,
        num_key_value_heads=model_cfg.num_key_value_heads,
        num_hidden_layers=model_cfg.num_hidden_layers,
        max_position_embeddings=model_cfg.max_position_embeddings,
        rope_theta=model_cfg.rope_theta,
        dropout=model_cfg.dropout,
        initializer_range=model_cfg.initializer_range,
    )
    wrapper = LLaMAModelWrapper(llama_config, tokenizer)
    wrapper.build_model()
    wrapper.model.model.gradient_checkpointing = True
    
    if model_cfg.checkpoint_path:
        wrapper.load_pretrained(model_cfg.checkpoint_path)
        
    return wrapper

def main():
    deepspeed.init_distributed()
    global_rank = torch.distributed.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    logger = None
    task = None
    model_cfg = LLaMAConfig()
    data_cfg = DataConfig()
    train_cfg = TrainerConfig()
    torch.manual_seed(train_cfg.seed)

    if global_rank == 0:
        task = Task.init(
            project_name="llm_pretrain",
            task_name="pretrain_advanced_llama_32b",
            reuse_last_task_id=False,
        )
        logger = task.get_logger()
        os.makedirs(train_cfg.output_dir, exist_ok=True)

    tokenizer_path = data_cfg.tokenizer_name
    if not os.path.isdir(tokenizer_path):
        raise FileNotFoundError(
            f"Local tokenizer path not found: {tokenizer_path}. "
            "Ensure S3 sync placed files there."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer_vocab_size = len(tokenizer)
    if model_cfg.vocab_size != tokenizer_vocab_size:
        if global_rank == 0:
            print(
                f"[WARN] model_cfg.vocab_size ({model_cfg.vocab_size}) != len(tokenizer) ({tokenizer_vocab_size}). "
                "Overriding model vocab_size to match tokenizer."
            )
        model_cfg.vocab_size = tokenizer_vocab_size

    model = LLaMAForCausalLM(model_cfg)
    model.model.gradient_checkpointing = True

    dm = DataModule(
        dataset_name=data_cfg.dataset_name,
        data_root=data_cfg.data_root,
        sources=data_cfg.sources,
        file_glob=data_cfg.file_glob,
        dataset_config=getattr(data_cfg, "dataset_config", None),
        trust_remote_code=getattr(data_cfg, "trust_remote_code", True),
        tokenizer=tokenizer,
        block_size=data_cfg.block_size,
        train_split=data_cfg.train_split,
        val_split=data_cfg.val_split,
        batch_size=data_cfg.batch_size,
        num_workers=data_cfg.num_workers,
        shuffle_buffer=data_cfg.shuffle_buffer,
        prefetch_factor=data_cfg.prefetch_factor,
        val_take=data_cfg.val_take,
        text_field=data_cfg.text_field,
    )
    dm.prepare()

    first_batch = next(iter(dm.train_dataloader()))
    local_max_id = int(first_batch["input_ids"].max().item())
    max_id = torch.tensor(local_max_id, device=torch.device(f"cuda:{local_rank}"))
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(max_id, op=torch.distributed.ReduceOp.MAX)
    max_id_val = int(max_id.item())
    if max_id_val >= model_cfg.vocab_size:
        raise RuntimeError(
            f"Found token id {max_id_val} >= model vocab_size {model_cfg.vocab_size}. "
            "Fix: ensure model_cfg.vocab_size == len(tokenizer) and you are using the intended tokenizer."
        )

    here = os.path.dirname(os.path.abspath(__file__))
    ds_config_path = os.path.join(here, train_cfg.ds_config_path)

    with open(ds_config_path, "r") as f:
        ds_config = json.load(f)

    ds_config["gradient_accumulation_steps"] = train_cfg.grad_accum_steps
    ds_config["train_micro_batch_size_per_gpu"] = data_cfg.batch_size
    if "scheduler" in ds_config and "params" in ds_config["scheduler"]:
        ds_config["scheduler"]["params"]["total_num_steps"] = train_cfg.max_steps
        warmup = ds_config["scheduler"]["params"].get("warmup_num_steps", train_cfg.max_steps // 10)
        ds_config["scheduler"]["params"]["warmup_num_steps"] = min(warmup, train_cfg.max_steps)

    engine, _, _, _ = deepspeed.initialize(
        config=ds_config,
        model=model,
        model_parameters=model.parameters(),
    )

    trainer = LLMTrainer(
        engine=engine,
        train_dataloader=dm.train_dataloader(),
        val_dataloader=dm.val_dataloader(),
        trainer_cfg=train_cfg,
        logger=logger,
    )

    if global_rank == 0 and task is not None:
        task.connect(model_cfg, name="Model Config")
        task.connect(train_cfg, name="Trainer Config")
        task.connect(data_cfg, name="Data Config")

    trainer.train()



if __name__ == "__main__":
    main()