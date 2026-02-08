import json
import os
import random
import hashlib
import itertools
from typing import Any, Dict, Iterable, List, Optional
from tqdm import tqdm
from datasets import load_dataset


class DataPrepConfig:
    def __init__(
        self,
        output_dir: str,
        ultrachat_split: str = "train_sft",
        oasst2_split: str = "train",
        chat_template: str = "chatml",
        system_prompt: str = "You are a helpful assistant.",
        max_samples: Optional[int] = None,
        seed: int = 42,
        val_ratio: float = 0.05,
        min_char_length: int = 50,
        max_char_length: int = 8192,
    ) -> None:
        self.output_dir = output_dir
        self.ultrachat_split = ultrachat_split
        self.oasst2_split = oasst2_split
        self.chat_template = chat_template
        self.system_prompt = system_prompt
        self.max_samples = max_samples
        self.seed = seed
        self.val_ratio = val_ratio
        self.min_char_length = min_char_length
        self.max_char_length = max_char_length


def load_ultrachat(split: str) -> Any:
    print(f"Loading UltraChat 200k ({split})...")
    return load_dataset("HuggingFaceH4/ultrachat_200k", split=split)


def load_oasst2(split: str) -> Any:
    print(f"Loading OASST2 ({split})...")
    return load_dataset("OpenAssistant/oasst2", split=split)


def get_content_hash(messages: List[Dict[str, str]]) -> str:
    full_text = "".join([m["content"] for m in messages])
    return hashlib.md5(full_text.encode("utf-8")).hexdigest()


def normalize_ultrachat(example: Dict[str, Any]) -> List[Dict[str, str]]:
    return example["messages"]


def extract_oasst2_conversations(dataset: Any) -> List[List[Dict[str, str]]]:
    if "messages" in dataset.column_names:
        return [ex["messages"] for i, ex in enumerate(dataset) if i < 10000] 

    conversations = []
    if "text" in dataset.column_names and "role" in dataset.column_names:
        pass
        
    return conversations


def format_chat(
    messages: List[Dict[str, str]],
    template_name: str,
    system_prompt: Optional[str],
) -> str:
    formatted_text = ""
    if template_name == "chatml":
        if system_prompt:
            formatted_text += f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            formatted_text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    elif template_name == "llama3":
        formatted_text += "<|begin_of_text|>"
        if system_prompt:
            formatted_text += f"<|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            formatted_text += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
    else:
        if system_prompt:
            formatted_text += f"System: {system_prompt}\n"
        for msg in messages:
            formatted_text += f"{msg['role'].capitalize()}: {msg['content']}\n"
    return formatted_text


def process_dataset(dataset: Iterable[Dict[str, Any]], source: str, cfg: DataPrepConfig) -> List[Dict[str, Any]]:
    records = []
    seen_hashes = set()
    
    for example in tqdm(dataset, desc=f"Processing {source}"):
        messages = []
        if source == "ultrachat":
            messages = example["messages"]
        elif source == "oasst2":
            if "messages" in example:
                messages = example["messages"]
            elif "text" in example and "role" in example:
                continue 

        if not messages or len(messages) < 2:
            continue
            
        if messages[-1]["role"] != "assistant":
            messages = messages[:-1]
            if len(messages) < 2: continue

        content_hash = get_content_hash(messages)
        if content_hash in seen_hashes:
            continue
        seen_hashes.add(content_hash)
        
        full_text = format_chat(messages, cfg.chat_template, cfg.system_prompt)
        if len(full_text) < cfg.min_char_length or len(full_text) > cfg.max_char_length:
            continue
            
        records.append({
            "text": full_text,
            "source": source,
            "messages": messages
        })
        
    return records


def prepare_sft_data(cfg: DataPrepConfig) -> None:
    ultrachat = load_ultrachat(cfg.ultrachat_split)
    oasst2 = load_oasst2(cfg.oasst2_split)
    all_records = []
    all_records.extend(process_dataset(ultrachat, "ultrachat", cfg))
    all_records.extend(process_dataset(oasst2, "oasst2", cfg))
    random.seed(cfg.seed)
    random.shuffle(all_records)
    if cfg.max_samples:
        all_records = all_records[:cfg.max_samples]
    
    val_size = int(len(all_records) * cfg.val_ratio)
    val_records = all_records[:val_size]
    train_records = all_records[val_size:]
    
    print(f"Final dataset: {len(train_records)} train, {len(val_records)} val")
    
    os.makedirs(cfg.output_dir, exist_ok=True)
    
    for name, data in [("train", train_records), ("val", val_records)]:
        path = os.path.join(cfg.output_dir, f"{name}.jsonl")
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
    
    print(f"Saved to {cfg.output_dir}")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/sft")
    parser.add_argument("--chat_template", type=str, default="chatml")
    parser.add_argument("--system_prompt", type=str, default="You are a helpful assistant.")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    args = parser.parse_args()
    
    cfg = DataPrepConfig(
        output_dir=args.output_dir,
        chat_template=args.chat_template,
        system_prompt=args.system_prompt,
        max_samples=args.max_samples,
        seed=args.seed,
        val_ratio=args.val_ratio,
    )
    prepare_sft_data(cfg)


if __name__ == "__main__":
    main()
