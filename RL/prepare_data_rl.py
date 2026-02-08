import json
import os
import random
import hashlib
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class RLDataPrepConfig:
    def __init__(
        self,
        output_dir: str,
        chat_template: str = "chatml",
        default_system_prompt: str = "You are a helpful assistant.",
        seed: int = 42,
        val_ratio: float = 0.05,
        max_length: int = 2048,
        max_prompt_length: int = 1024,
        add_eos: bool = True,
        tokenizer_path: str = "checkpoints/sft_stage3/best",
        trust_remote_code: bool = True,
        use_fast_tokenizer: bool = True,
        ultrafeedback_ratio: float = 0.85,
        hh_rlhf_ratio: float = 0.15,
    ) -> None:
        self.output_dir = output_dir
        self.chat_template = chat_template
        self.default_system_prompt = default_system_prompt
        self.seed = seed
        self.val_ratio = val_ratio
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
        self.add_eos = add_eos
        self.tokenizer_path = tokenizer_path
        self.trust_remote_code = trust_remote_code
        self.use_fast_tokenizer = use_fast_tokenizer
        self.ultrafeedback_ratio = ultrafeedback_ratio
        self.hh_rlhf_ratio = hh_rlhf_ratio


def _load_tokenizer(cfg: RLDataPrepConfig) -> Any:
    resolved_path = cfg.tokenizer_path
    if not os.path.exists(resolved_path):
        fallback_paths = ["/from_s3/sft1_model/", "/from_s3/sft3_model/", "/from_s3/sft2_model/"]
        resolved_path = None
        for p in fallback_paths:
            if os.path.exists(p):
                resolved_path = p
                break
        if not resolved_path:
            # If nothing found, we might be in a local env without the model mounted.
            # In that case, we can't really prepare data properly without tokenizer.
            # But let's try to use a generic one if possible or just fail.
            raise FileNotFoundError(f"Tokenizer path not found: {cfg.tokenizer_path}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            resolved_path,
            trust_remote_code=cfg.trust_remote_code,
            use_fast=cfg.use_fast_tokenizer,
            local_files_only=True,
        )
    except Exception as e:
        logger.warning(f"Fast tokenizer failed to load ({e}). Retrying with use_fast=False.")
        tokenizer = AutoTokenizer.from_pretrained(
            resolved_path,
            trust_remote_code=cfg.trust_remote_code,
            use_fast=False,
            local_files_only=True,
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _get_hf_token() -> Optional[str]:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def format_chatml_prompt(messages: List[Dict[str, str]], system_prompt: Optional[str]) -> str:
    text = ""
    if system_prompt:
        text += f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
    for m in messages:
        if m["role"] == "assistant":
            # We don't want the last assistant message in the prompt
            continue
        text += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
    text += "<|im_start|>assistant\n"
    return text


def _process_ultrafeedback(
    dataset: Any,
    cfg: RLDataPrepConfig,
    seen_hashes: Set[str],
) -> List[Dict[str, Any]]:
    records = []
    for ex in tqdm(dataset, desc="Processing UltraFeedback"):
        # UltraFeedback binarized format: 'prompt', 'chosen', 'rejected'
        prompt = ex["prompt"]
        chosen = ex["chosen"] # list of messages
        rejected = ex["rejected"] # list of messages
        
        # In UltraFeedback binarized, chosen/rejected usually contain the whole conversation.
        # The last message is the response.
        if not chosen or chosen[-1]["role"] != "assistant":
            continue
        if not rejected or rejected[-1]["role"] != "assistant":
            continue
            
        # Extract prompt messages (all except last)
        prompt_msgs = chosen[:-1]
        chosen_content = chosen[-1]["content"]
        rejected_content = rejected[-1]["content"]
        
        # Check if we should add system prompt
        has_system = any(m["role"] == "system" for m in prompt_msgs)
        system = cfg.default_system_prompt if not has_system else None
        
        prompt_text = format_chatml_prompt(prompt_msgs, system)
        
        h = hashlib.md5(prompt_text.encode("utf-8")).hexdigest()
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        
        records.append({
            "prompt": prompt_text,
            "chosen": chosen_content,
            "rejected": rejected_content,
            "source": "ultrafeedback"
        })
    return records


def _process_hh_rlhf(
    dataset: Any,
    cfg: RLDataPrepConfig,
    seen_hashes: Set[str],
) -> List[Dict[str, Any]]:
    records = []
    for ex in tqdm(dataset, desc="Processing HH-RLHF"):
        # HH-RLHF format: 'chosen' and 'rejected' strings with \n\nHuman: and \n\nAssistant:
        chosen_raw = ex["chosen"]
        rejected_raw = ex["rejected"]
        
        # Find common prefix (the prompt)
        # HH-RLHF strings look like: "\n\nHuman: ... \n\nAssistant: ..."
        # We need to split them.
        
        def parse_hh(text):
            parts = text.split("\n\n")
            msgs = []
            for p in parts:
                if p.startswith("Human: "):
                    msgs.append({"role": "user", "content": p[len("Human: "):]})
                elif p.startswith("Assistant: "):
                    msgs.append({"role": "assistant", "content": p[len("Assistant: "):]})
            return msgs

        chosen_msgs = parse_hh(chosen_raw)
        rejected_msgs = parse_hh(rejected_raw)
        
        if not chosen_msgs or chosen_msgs[-1]["role"] != "assistant":
            continue
        if not rejected_msgs or rejected_msgs[-1]["role"] != "assistant":
            continue
            
        prompt_msgs = chosen_msgs[:-1]
        chosen_content = chosen_msgs[-1]["content"]
        rejected_content = rejected_msgs[-1]["content"]
        
        prompt_text = format_chatml_prompt(prompt_msgs, cfg.default_system_prompt)
        
        h = hashlib.md5(prompt_text.encode("utf-8")).hexdigest()
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        
        records.append({
            "prompt": prompt_text,
            "chosen": chosen_content,
            "rejected": rejected_content,
            "source": "hh_rlhf"
        })
    return records


def prepare_rl_data(cfg: RLDataPrepConfig) -> None:
    rng = random.Random(cfg.seed)
    seen_hashes: Set[str] = set()
    hf_token = _get_hf_token()

    logger.info("Loading RL datasets...")
    
    # UltraFeedback
    uf_train = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs", token=hf_token, trust_remote_code=cfg.trust_remote_code)
    uf_test = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="test_prefs", token=hf_token, trust_remote_code=cfg.trust_remote_code)
    
    uf_records = _process_ultrafeedback(uf_train, cfg, seen_hashes)
    uf_val_records = _process_ultrafeedback(uf_test, cfg, seen_hashes)
    
    # HH-RLHF
    hh_train = load_dataset("Anthropic/hh-rlhf", split="train", token=hf_token, trust_remote_code=cfg.trust_remote_code)
    hh_test = load_dataset("Anthropic/hh-rlhf", split="test", token=hf_token, trust_remote_code=cfg.trust_remote_code)
    
    hh_records = _process_hh_rlhf(hh_train, cfg, seen_hashes)
    hh_val_records = _process_hh_rlhf(hh_test, cfg, seen_hashes)
    
    # Mixing
    # We want 85-90% UF, 10-15% HH.
    # Let's target the requested tokens/examples ratio.
    
    target_train_size = len(uf_records) + len(hh_records)
    # If we want 85% UF, then total = len(uf) / 0.85
    # If len(hh) is enough, we use that.
    
    uf_ratio = cfg.ultrafeedback_ratio
    hh_ratio = cfg.hh_rlhf_ratio
    
    # Total examples based on what we have
    total_train = min(len(uf_records) / uf_ratio, len(hh_records) / hh_ratio)
    
    num_uf = int(total_train * uf_ratio)
    num_hh = int(total_train * hh_ratio)
    
    rng.shuffle(uf_records)
    rng.shuffle(hh_records)
    
    train_mix = uf_records[:num_uf] + hh_records[:num_hh]
    rng.shuffle(train_mix)
    
    # Same for validation but smaller
    total_val = min(len(uf_val_records) / uf_ratio, len(hh_val_records) / hh_ratio)
    num_uf_val = int(total_val * uf_ratio)
    num_hh_val = int(total_val * hh_ratio)
    
    val_mix = uf_val_records[:num_uf_val] + hh_val_records[:num_hh_val]
    rng.shuffle(val_mix)
    
    logger.info(f"Final RL mix: train={len(train_mix)} val={len(val_mix)}")
    logger.info(f"Source breakdown: UF={num_uf}, HH={num_hh}")

    os.makedirs(cfg.output_dir, exist_ok=True)
    for name, data in [("train", train_mix), ("val", val_mix)]:
        path = os.path.join(cfg.output_dir, f"{name}.jsonl")
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
    logger.info(f"Saved RL data to {cfg.output_dir}")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/rl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tokenizer_path", type=str, default="checkpoints/sft_stage3/best")
    parser.add_argument("--ultrafeedback_ratio", type=float, default=0.85)
    parser.add_argument("--hh_rlhf_ratio", type=float, default=0.15)
    args = parser.parse_args()

    cfg = RLDataPrepConfig(
        output_dir=args.output_dir,
        seed=args.seed,
        tokenizer_path=args.tokenizer_path,
        ultrafeedback_ratio=args.ultrafeedback_ratio,
        hh_rlhf_ratio=args.hh_rlhf_ratio,
    )
    prepare_rl_data(cfg)


if __name__ == "__main__":
    main()
