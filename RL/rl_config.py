from typing import Optional
from dataclasses import dataclass, field


@dataclass
class RLModelConfig:
    checkpoint_path: str = field(
        default="checkpoints/sft_stage3/best",
        metadata={"help": "Path to the SFT Stage 3 model weights (HF format directory)."},
    )
    tokenizer_path: str = field(
        default="checkpoints/sft_stage3/best",
        metadata={"help": "Path to the tokenizer."},
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
class RLDataConfig:
    data_root: str = field(
        default="data/rl",
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
    max_length: int = field(
        default=2048,
        metadata={"help": "Maximum length for tokenization."},
    )
    max_prompt_length: int = field(
        default=1024,
        metadata={"help": "Maximum length for the prompt portion."},
    )
    batch_size: int = field(
        default=1,
        metadata={"help": "Micro batch size per GPU. SimPO can be memory intensive."},
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
        metadata={"help": "Whether to add an EOS token to the end of responses."},
    )
    chat_template: str = field(
        default="chatml",
        metadata={"help": "Chat template name (chatml, llama3, etc.)."},
    )
    ultrafeedback_ratio: float = field(
        default=0.85,
        metadata={"help": "Ratio of UltraFeedback samples."},
    )
    hh_rlhf_ratio: float = field(
        default=0.15,
        metadata={"help": "Ratio of Anthropic HH-RLHF samples."},
    )


@dataclass
class RLTrainerConfig:
    beta: float = field(
        default=2.0,
        metadata={"help": "SimPO beta parameter (scale for log-probs)."},
    )
    gamma: float = field(
        default=0.5,
        metadata={"help": "SimPO gamma parameter (target margin)."},
    )
    max_steps: int = field(
        default=1000,
        metadata={"help": "Maximum number of optimizer steps."},
    )
    max_epochs: int = field(
        default=1,
        metadata={"help": "Maximum number of epochs."},
    )
    max_eval_batches: int = field(
        default=50,
        metadata={"help": "Maximum number of batches to evaluate."},
    )
    eval_interval: int = field(
        default=50,
        metadata={"help": "Steps between evaluations."},
    )
    log_interval: int = field(
        default=5,
        metadata={"help": "Steps between logging."},
    )
    output_dir: str = field(
        default="checkpoints/rl_simpo",
        metadata={"help": "Directory to save checkpoints."},
    )
    ds_config_path: str = field(
        default="RL/ds_config_zero3.json",
        metadata={"help": "Path to the DeepSpeed config file."},
    )
    save_best_only: bool = field(
        default=True,
        metadata={"help": "Whether to only save the best checkpoint."},
    )
    save_final: bool = field(
        default=True,
        metadata={"help": "Whether to save a final checkpoint."},
    )
    learning_rate: float = field(
        default=5e-7,
        metadata={"help": "Learning rate (usually very small for RL)."},
    )
