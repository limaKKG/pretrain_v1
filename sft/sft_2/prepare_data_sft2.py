import json
import os
import random
import hashlib
import logging
import time
from typing import Any, Dict, Iterable, List, Optional, Set
from tqdm import tqdm
from datasets import load_dataset
from datasets.download.download_config import DownloadConfig
from http.client import IncompleteRead
from transformers import AutoTokenizer
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class SFT2DataPrepConfig:
    def __init__(
        self,
        output_dir: str,
        chat_template: str = "chatml",
        system_prompt: str = "You are a helpful assistant.",
        seed: int = 42,
        val_ratio: float = 0.05,
        min_char_length: int = 50,
        max_char_length: int = 8192,
        max_length: int = 2048,
        add_eos: bool = True,
        tokenizer_path: str = "/from_s3/sft1_model/",
        trust_remote_code: bool = True,
        use_fast_tokenizer: bool = True,
        stage1_ratio: float = 0.20,
        flan_ratio: float = 0.45,
        natural_ratio: float = 0.15,
        math_ratio: float = 0.20,
        math_datasets: Optional[List[str]] = None,
        # Cap how much we materialize per source (important for huge streaming datasets like FLAN)
        max_tokens_per_pool: int = 50_000_000,
    ) -> None:
        self.output_dir = output_dir
        self.chat_template = chat_template
        self.system_prompt = system_prompt
        self.seed = seed
        self.val_ratio = val_ratio
        self.min_char_length = min_char_length
        self.max_char_length = max_char_length
        self.max_length = max_length
        self.add_eos = add_eos
        self.tokenizer_path = tokenizer_path
        self.trust_remote_code = trust_remote_code
        self.use_fast_tokenizer = use_fast_tokenizer
        self.stage1_ratio = stage1_ratio
        self.flan_ratio = flan_ratio
        self.natural_ratio = natural_ratio
        self.math_ratio = math_ratio
        self.math_datasets = math_datasets or [
            "nvidia/OpenMathInstruct-1",
            "meta-math/MetaMathQA",
        ]
        self.max_tokens_per_pool = max_tokens_per_pool


def get_content_hash(messages: List[Dict[str, str]]) -> str:
    full_text = "".join([f"{m.get('role','')}:{m.get('content','')}" for m in messages])
    return hashlib.md5(full_text.encode("utf-8")).hexdigest()


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
    else:
        if system_prompt:
            formatted_text += f"System: {system_prompt}\n"
        for msg in messages:
            formatted_text += f"{msg['role'].capitalize()}: {msg['content']}\n"
    return formatted_text


def _coerce_to_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join([str(v) for v in value if v is not None])
    return str(value)


def _get_first_nonempty(example: Dict[str, Any], keys: List[str]) -> Optional[str]:
    for key in keys:
        if key in example and example[key] not in (None, "", []):
            return _coerce_to_str(example[key])
    return None


def _load_tokenizer(cfg: SFT2DataPrepConfig) -> Any:
    resolved_path = cfg.tokenizer_path
    if not os.path.exists(resolved_path):
        fallback_paths = ["/from_s3/sft1_model/"]
        resolved_path = None
        for p in fallback_paths:
            if os.path.exists(p):
                resolved_path = p
                break
        if not resolved_path:
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


def _load_dataset_with_retries(
    name: str,
    *,
    split: str,
    trust_remote_code: bool,
    retries: int = 20,
    streaming_fallback: bool = True,
):
    """
    Robust HF datasets download with resume + retries.
    Fixes transient 'IncompleteRead' / broken connections on large shards.
    """
    last_err: Optional[Exception] = None
    dl_cfg = DownloadConfig(resume_download=True, max_retries=50)
    cache_dir = os.environ.get("HF_DATASETS_CACHE") or os.environ.get("HF_HOME") or None

    def _try(streaming: bool):
        return load_dataset(
            name,
            split=split,
            trust_remote_code=trust_remote_code,
            download_config=dl_cfg,
            cache_dir=cache_dir,
            streaming=streaming,
        )

    for attempt in range(1, retries + 1):
        try:
            return _try(streaming=False)
        except IncompleteRead as e:
            last_err = e
            logger.warning(f"Download IncompleteRead for {name} (attempt {attempt}/{retries}). Retrying...")
        except (OSError, ConnectionError, TimeoutError) as e:
            last_err = e
            logger.warning(f"Network error for {name} (attempt {attempt}/{retries}): {e}. Retrying...")
        except Exception as e:
            last_err = e
            logger.warning(f"Download failed for {name} (attempt {attempt}/{retries}): {e}. Retrying...")
        time.sleep(min(60, 2 * attempt))

    if streaming_fallback:
        logger.warning(f"Falling back to streaming load for {name} after {retries} failed attempts.")
        try:
            return _try(streaming=True)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Failed to load dataset {name} after {retries} attempts") from last_err


def _process_iterable_until_token_cap(
    dataset: Iterable[Dict[str, Any]],
    source: str,
    cfg: SFT2DataPrepConfig,
    message_extractor,
    tokenizer: Any,
    seen_hashes: Set[str],
    token_cap: int,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    total_tokens = 0
    # Reuse existing processing logic but stop early
    skipped_count = {"short_long": 0, "duplicate": 0, "invalid": 0, "over_max_tokens": 0}

    for example in tqdm(dataset, desc=f"Processing {source} (cap {token_cap} tokens)"):
        try:
            messages = message_extractor(example)
        except Exception:
            skipped_count["invalid"] += 1
            continue

        if not messages or len(messages) < 2:
            skipped_count["invalid"] += 1
            continue
        if messages[-1]["role"] != "assistant":
            messages = messages[:-1]
            if len(messages) < 2:
                skipped_count["invalid"] += 1
                continue

        content_hash = get_content_hash(messages)
        if content_hash in seen_hashes:
            skipped_count["duplicate"] += 1
            continue
        seen_hashes.add(content_hash)

        full_text = format_chat(messages, cfg.chat_template, cfg.system_prompt)
        if len(full_text) < cfg.min_char_length or len(full_text) > cfg.max_char_length:
            skipped_count["short_long"] += 1
            continue

        token_count = _count_tokens(full_text, tokenizer, cfg.max_length, cfg.add_eos)
        if token_count == 0:
            skipped_count["invalid"] += 1
            continue
        if cfg.max_length and token_count > cfg.max_length:
            skipped_count["over_max_tokens"] += 1
            continue

        records.append(
            {"text": full_text, "source": source, "messages": messages, "token_count": token_count}
        )
        total_tokens += int(token_count)
        if total_tokens >= token_cap:
            break

    logger.info(f"Source {source}: kept={len(records)} tokens={total_tokens} skipped={skipped_count}")
    return records


def _count_tokens(
    text: str,
    tokenizer: Any,
    max_length: int,
    add_eos: bool,
) -> int:
    if add_eos and tokenizer.eos_token and not text.endswith(tokenizer.eos_token):
        text = text + tokenizer.eos_token
    token_count = len(tokenizer.encode(text, add_special_tokens=False))
    return token_count


def process_generic_dataset(
    dataset: Iterable[Dict[str, Any]],
    source: str,
    cfg: SFT2DataPrepConfig,
    message_extractor,
    tokenizer: Any,
    seen_hashes: Set[str],
) -> List[Dict[str, Any]]:
    records = []
    skipped_count = {"short_long": 0, "duplicate": 0, "invalid": 0, "over_max_tokens": 0}
    
    for example in tqdm(dataset, desc=f"Processing {source}"):
        try:
            messages = message_extractor(example)
        except Exception:
            skipped_count["invalid"] += 1
            continue

        if not messages or len(messages) < 2:
            skipped_count["invalid"] += 1
            continue
            
        if messages[-1]["role"] != "assistant":
            messages = messages[:-1]
            if len(messages) < 2:
                skipped_count["invalid"] += 1
                continue

        content_hash = get_content_hash(messages)
        if content_hash in seen_hashes:
            skipped_count["duplicate"] += 1
            continue
        seen_hashes.add(content_hash)
        
        full_text = format_chat(messages, cfg.chat_template, cfg.system_prompt)
        if len(full_text) < cfg.min_char_length or len(full_text) > cfg.max_char_length:
            skipped_count["short_long"] += 1
            continue
        
        token_count = _count_tokens(full_text, tokenizer, cfg.max_length, cfg.add_eos)
        if token_count == 0:
            skipped_count["invalid"] += 1
            continue
        if cfg.max_length and token_count > cfg.max_length:
            skipped_count["over_max_tokens"] += 1
            continue
            
        records.append({
            "text": full_text,
            "source": source,
            "messages": messages,
            "token_count": token_count,
        })
        
    logger.info(f"Source {source}: {len(records)} records, skipped: {skipped_count}")
    return records

def extract_ultrachat(ex):
    return ex["messages"]


def extract_oasst2_tree(dataset: Any) -> List[List[Dict[str, str]]]:
    msg_dict = {}
    children = {}
    for msg in dataset:
        msg_id = msg.get("message_id")
        if not msg_id:
            continue
        msg_dict[msg_id] = msg
        parent_id = msg.get("parent_id")
        if parent_id:
            children.setdefault(parent_id, []).append(msg_id)

    leaf_ids = [
        msg_id for msg_id, msg in msg_dict.items()
        if msg.get("role") == "assistant" and msg_id not in children
    ]

    conversations = []
    for leaf_id in leaf_ids:
        path = []
        curr_id = leaf_id
        while curr_id:
            msg = msg_dict.get(curr_id)
            if not msg:
                break
            role = msg.get("role")
            text = msg.get("text")
            if not role or text is None:
                break
            path.append({"role": role, "content": text})
            curr_id = msg.get("parent_id")

        if len(path) >= 2:
            path.reverse()
            conversations.append(path)

    return conversations


def extract_natural_instructions(ex):
    definition = _coerce_to_str(ex.get("definition", ""))
    inputs = _coerce_to_str(ex.get("inputs", ""))
    targets = _coerce_to_str(ex.get("targets", ""))
    if not definition or not targets:
        raise ValueError("Missing definition or targets")
    prompt = definition
    if inputs:
        prompt = f"{definition}\n\nInput: {inputs}"
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": targets},
    ]


def extract_flan_v2(ex):
    inputs = _coerce_to_str(ex.get("inputs", ""))
    targets = _coerce_to_str(ex.get("targets", ""))
    if not inputs or not targets:
        raise ValueError("Missing inputs or targets")
    return [
        {"role": "user", "content": inputs},
        {"role": "assistant", "content": targets},
    ]


def extract_math_example(ex):
    if isinstance(ex, list):
        return ex
    if "messages" in ex:
        return ex["messages"]

    instruction = _get_first_nonempty(ex, ["instruction", "prompt", "query", "question", "problem", "input"])
    output = _get_first_nonempty(ex, ["output", "response", "answer", "solution", "completion", "final_answer"])
    extra_input = _get_first_nonempty(ex, ["input", "context"])

    if not instruction or not output:
        raise ValueError("Missing instruction or output")

    if extra_input and extra_input != instruction:
        user_text = f"{instruction}\n\nInput: {extra_input}"
    else:
        user_text = instruction

    return [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": output},
    ]


def _sum_tokens(records: List[Dict[str, Any]]) -> int:
    return sum(r.get("token_count", 0) for r in records)


def _select_by_token_budget(
    records: List[Dict[str, Any]],
    target_tokens: int,
    rng: random.Random,
) -> (List[Dict[str, Any]], int):
    rng.shuffle(records)
    selected = []
    total_tokens = 0
    for rec in records:
        selected.append(rec)
        total_tokens += rec["token_count"]
        if total_tokens >= target_tokens:
            break
    return selected, total_tokens


def prepare_sft2_data(cfg: SFT2DataPrepConfig, tokenizer: Optional[Any] = None) -> None:
    rng = random.Random(cfg.seed)
    seen_hashes = set()
    ratios = {
        "stage1": cfg.stage1_ratio,
        "flan": cfg.flan_ratio,
        "natural": cfg.natural_ratio,
        "math": cfg.math_ratio,
    }
    total_ratio = sum(ratios.values())
    if total_ratio <= 0:
        raise ValueError("Mix ratios must sum to a positive value.")
    if abs(total_ratio - 1.0) > 1e-6:
        logger.warning(f"Mix ratios sum to {total_ratio:.3f}; normalizing.")
        ratios = {k: v / total_ratio for k, v in ratios.items()}

    tokenizer = tokenizer or _load_tokenizer(cfg)

    logger.info("Loading Stage 2 datasets...")
    # FLAN is huge; prefer streaming to avoid Arrow generation of 168M rows.
    flan_v2 = load_dataset(
        "SirNeural/flan_v2",
        split="train",
        trust_remote_code=cfg.trust_remote_code,
        streaming=True,
    )
    natural_inst = _load_dataset_with_retries(
        "Muennighoff/natural-instructions", split="train", trust_remote_code=cfg.trust_remote_code
    )

    flan_records = _process_iterable_until_token_cap(
        flan_v2,
        "flan_v2",
        cfg,
        extract_flan_v2,
        tokenizer,
        seen_hashes,
        token_cap=cfg.max_tokens_per_pool,
    )
    natural_records = process_generic_dataset(
        natural_inst, "natural-instructions", cfg, extract_natural_instructions, tokenizer, seen_hashes
    )

    math_records: List[Dict[str, Any]] = []
    for dataset_name in cfg.math_datasets:
        logger.info(f"Loading math dataset: {dataset_name}")
        math_ds = _load_dataset_with_retries(
            dataset_name, split="train", trust_remote_code=cfg.trust_remote_code
        )
        # math can be large too; cap it
        math_records.extend(
            _process_iterable_until_token_cap(
                math_ds,
                dataset_name,
                cfg,
                extract_math_example,
                tokenizer,
                seen_hashes,
                token_cap=max(5_000_000, cfg.max_tokens_per_pool // 4),
            )
        )

    logger.info("Loading Stage 1 (replay) datasets...")
    ultrachat = _load_dataset_with_retries(
        "HuggingFaceH4/ultrachat_200k", split="train_sft", trust_remote_code=cfg.trust_remote_code
    )
    oasst2_raw = _load_dataset_with_retries(
        "OpenAssistant/oasst2", split="train", trust_remote_code=cfg.trust_remote_code
    )
    oasst2_conversations = extract_oasst2_tree(oasst2_raw)

    ultrachat_records = process_generic_dataset(
        ultrachat, "ultrachat", cfg, extract_ultrachat, tokenizer, seen_hashes
    )
    oasst2_records = process_generic_dataset(
        oasst2_conversations, "oasst2", cfg, lambda x: x, tokenizer, seen_hashes
    )

    stage1_records = ultrachat_records + oasst2_records

    pools = {
        "stage1": stage1_records,
        "flan": flan_records,
        "natural": natural_records,
        "math": math_records,
    }
    pool_tokens = {k: _sum_tokens(v) for k, v in pools.items()}

    for name, total in pool_tokens.items():
        if total <= 0:
            raise ValueError(f"No usable tokens for pool '{name}'. Check dataset extraction or filters.")

    total_possible = min(pool_tokens[name] / ratios[name] for name in pools.keys())
    target_tokens = {name: int(total_possible * ratios[name]) for name in pools.keys()}

    selected = {}
    selected_tokens = {}
    for name, records in pools.items():
        selected[name], selected_tokens[name] = _select_by_token_budget(
            records, target_tokens[name], rng
        )

    all_records = (
        selected["stage1"]
        + selected["flan"]
        + selected["natural"]
        + selected["math"]
    )
    rng.shuffle(all_records)

    val_size = int(len(all_records) * cfg.val_ratio)
    val_records = all_records[:val_size]
    train_records = all_records[val_size:]

    logger.info(f"Final dataset mix: {len(train_records)} train, {len(val_records)} val")
    logger.info(f"Token mix targets: {target_tokens}")
    logger.info(f"Token mix actuals: {selected_tokens}")

    os.makedirs(cfg.output_dir, exist_ok=True)

    for name, data in [("train", train_records), ("val", val_records)]:
        path = os.path.join(cfg.output_dir, f"{name}.jsonl")
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

    logger.info(f"Saved to {cfg.output_dir}")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/sft2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--max_char_length", type=int, default=8192)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--tokenizer_path", type=str, default="/from_s3/sft1_model/")
    parser.add_argument("--stage1_ratio", type=float, default=0.20)
    parser.add_argument("--flan_ratio", type=float, default=0.45)
    parser.add_argument("--natural_ratio", type=float, default=0.15)
    parser.add_argument("--math_ratio", type=float, default=0.20)
    parser.add_argument(
        "--math_datasets",
        type=str,
        default="nvidia/OpenMathInstruct-1,meta-math/MetaMathQA",
        help="Comma-separated list of math datasets.",
    )
    args = parser.parse_args()

    math_datasets = [d.strip() for d in args.math_datasets.split(",") if d.strip()]

    cfg = SFT2DataPrepConfig(
        output_dir=args.output_dir,
        seed=args.seed,
        val_ratio=args.val_ratio,
        max_char_length=args.max_char_length,
        max_length=args.max_length,
        tokenizer_path=args.tokenizer_path,
        stage1_ratio=args.stage1_ratio,
        flan_ratio=args.flan_ratio,
        natural_ratio=args.natural_ratio,
        math_ratio=args.math_ratio,
        math_datasets=math_datasets,
    )
    prepare_sft2_data(cfg)


if __name__ == "__main__":
    main()
