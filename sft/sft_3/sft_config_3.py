from typing import Optional
from dataclasses import dataclass, field


@dataclass
class SFTModelConfig:
    checkpoint_path: str = field(
        default="/from_s3/sft2_model/",
        metadata={"help": "Path to the SFT Stage 2 model weights (HF format directory)."},
    )
    tokenizer_path: str = field(
        default="/from_s3/sft2_model/",
        metadata={"help": "Path to the tokenizer (usually included in model weights)."},
    )
    max_position_embeddings: int = field(
        default=4096,
        metadata={"help": "Maximum sequence length for the model."},
    )
    rope_theta: float = field(
        default=500000.0,
        metadata={"help": "RoPE theta value."},
    )
    hidden_size: int = field(
        default=4096,
        metadata={"help": "Model hidden size."},
    )
    intermediate_size: int = field(
        default=11008,
        metadata={"help": "Model intermediate size."},
    )
    num_attention_heads: int = field(
        default=32,
        metadata={"help": "Number of attention heads."},
    )
    num_key_value_heads: int = field(
        default=8,
        metadata={"help": "Number of KV heads (for GQA)."},
    )
    num_hidden_layers: int = field(
        default=32,
        metadata={"help": "Number of hidden layers."},
    )
    vocab_size: Optional[int] = field(
        default=None,
        metadata={"help": "Override vocab size if needed."},
    )
    rms_norm_eps: float = field(
        default=1e-5,
        metadata={"help": "RMSNorm epsilon."},
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Whether to trust remote code from HuggingFace."},
    )


@dataclass
class SFTDataConfig:
    data_root: str = field(
        default="data/sft3",
        metadata={"help": "Root directory for the dataset."},
    )
    train_glob: str = field(
        default="train.jsonl",
        metadata={"help": "Glob pattern for training files."},
    )
    val_glob: str = field(
        default="val.jsonl",
        metadata={"help": "Glob pattern for validation files."},
    )
    text_field: str = field(
        default="text",
        metadata={"help": "Field in JSONL containing the conversation text."},
    )
    max_length: int = field(
        default=2048,
        metadata={"help": "Maximum length for tokenization."},
    )
    batch_size: int = field(
        default=4,
        metadata={"help": "Micro batch size per GPU. Overwritten by ds_config if present."},
    )
    num_workers: int = field(
        default=8,
        metadata={"help": "Number of workers for data loading."},
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed."},
    )
    val_ratio: float = field(
        default=0.05,
        metadata={"help": "Ratio of data to use for validation."},
    )
    add_eos: bool = field(
        default=True,
        metadata={"help": "Whether to add an EOS token to the end of conversations."},
    )
    mask_user: bool = field(
        default=True,
        metadata={"help": "Whether to mask user turns in the loss calculation."},
    )
    chat_template: str = field(
        default="chatml",
        metadata={"help": "Chat template name (chatml, llama3, etc.)."},
    )


@dataclass
class SFTTrainerConfig:
    # Calibration is usually short; adjust as needed.
    max_steps: int = field(
        default=170,
        metadata={"help": "Maximum number of optimizer steps."},
    )
    max_epochs: int = field(
        default=10,
        metadata={"help": "Maximum number of epochs (will stop early at max_steps)."},
    )
    max_eval_batches: int = field(
        default=50,
        metadata={"help": "Maximum number of batches to evaluate."},
    )
    eval_interval: int = field(
        default=10,
        metadata={"help": "Steps between evaluations."},
    )
    log_interval: int = field(
        default=10,
        metadata={"help": "Steps between logging."},
    )
    output_dir: str = field(
        default="checkpoints/sft_stage3",
        metadata={"help": "Directory to save checkpoints."},
    )
    ds_config_path: str = field(
        default="sft/sft_3/ds_config_zero3.json",
        metadata={"help": "Path to the DeepSpeed config file."},
    )
    save_best_only: bool = field(
        default=True,
        metadata={"help": "Whether to only save the best checkpoint."},
    )
    save_final: bool = field(
        default=False,
        metadata={"help": "Whether to save a final checkpoint."},
    )

