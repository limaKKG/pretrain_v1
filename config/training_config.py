from dataclasses import dataclass
from typing import Optional, List

@dataclass
class LLaMAConfig:
    vocab_size: int = 128256  
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    num_hidden_layers: int = 32
    max_position_embeddings: int = 4096
    rope_theta: float = 500000.0
    rms_norm_eps: float = 1e-5
    dropout: float = 0.0
    attention_dropout: float = 0.0
    initializer_range: float = 0.02
    tie_word_embeddings: bool = False
    checkpoint_path: Optional[str] = None

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads
    
    def __post_init__(self) -> None:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")

@dataclass
class TrainerState:
    global_step: int = 0
    best_metric: Optional[float] = None
    should_stop: bool = False

@dataclass
class TrainerConfig:
    max_steps: int = 100000
    grad_accum_steps: int = 8 
    eval_interval: int = 1000
    save_interval: int = 100000
    log_interval: int = 1
    output_dir: str = "checkpoints/llama_8b_pretrain"
    seed: int = 42
    fp16: bool = False
    bf16: bool = True 
    ds_config_path: str = "config/ds_config.json"
    save_best_only: bool = True
    save_final: bool = False

@dataclass
class DataConfig:
    dataset_name: str = "togethercomputer/RedPajama-Data-1T"
    dataset_config: str = "default"    
    trust_remote_code: bool = True  
    tokenizer_name: str = "/from_s3/model"
    block_size: int = 4096
    train_split: str = "train"
    val_split: str = "train"
    batch_size: int = 2
    num_workers: int = 0
    shuffle_buffer: int = 10000
    prefetch_factor: int = 2
    data_root: Optional[str] = "/from_s3"
    sources: Optional[List[str]] = None
    file_glob: str = "part-*.jsonl.gz"
    val_take: int = 5000
    text_field: str = "text"