import json
import os
import random
import hashlib
import logging
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class SFT3DataPrepConfig:
    def __init__(
        self,
        output_dir: str,
        chat_template: str = "chatml",
        default_system_prompt: str = "You are a helpful assistant.",
        seed: int = 42,
        val_ratio: float = 0.05,
        min_char_length: int = 50,
        max_char_length: int = 8192,
        max_length: int = 2048,
        add_eos: bool = True,
        tokenizer_path: str = "/from_s3/sft2_model/",
        trust_remote_code: bool = True,
        use_fast_tokenizer: bool = True,
        # token-mix
        no_robots_ratio: float = 0.35,
        lima_ratio: float = 0.15,
        stage2_replay_ratio: float = 0.50,
        # stage2 replay source
        stage2_data_root: str = "data/sft2",
        stage2_train_file: str = "train.jsonl",
    ) -> None:
        self.output_dir = output_dir
        self.chat_template = chat_template
        self.default_system_prompt = default_system_prompt
        self.seed = seed
        self.val_ratio = val_ratio
        self.min_char_length = min_char_length
        self.max_char_length = max_char_length
        self.max_length = max_length
        self.add_eos = add_eos
        self.tokenizer_path = tokenizer_path
        self.trust_remote_code = trust_remote_code
        self.use_fast_tokenizer = use_fast_tokenizer
        self.no_robots_ratio = no_robots_ratio
        self.lima_ratio = lima_ratio
        self.stage2_replay_ratio = stage2_replay_ratio
        self.stage2_data_root = stage2_data_root
        self.stage2_train_file = stage2_train_file


def _load_tokenizer(cfg: SFT3DataPrepConfig) -> Any:
    resolved_path = cfg.tokenizer_path
    if not os.path.exists(resolved_path):
        fallback_paths = ["/from_s3/sft2_model/", "/from_s3/sft1_model/"]
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


def _get_hf_token() -> Optional[str]:
    # HF hub honors both; keep both for portability.
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def _normalize_role(role: str) -> str:
    r = (role or "").strip().lower()
    if r in {"user", "human", "prompt"}:
        return "user"
    if r in {"assistant", "gpt", "model"}:
        return "assistant"
    if r in {"system"}:
        return "system"
    return r or "user"


def _as_chat_messages_from_conversations(convs: Any) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    if not isinstance(convs, list):
        return messages
    for turn in convs:
        if not isinstance(turn, dict):
            continue
        role = _normalize_role(turn.get("role") or turn.get("from") or turn.get("speaker") or "")
        content = turn.get("content") or turn.get("value") or turn.get("text") or ""
        if content is None:
            continue
        content = str(content)
        if content.strip() == "":
            continue
        messages.append({"role": role, "content": content})
    return messages


def extract_messages_generic(ex: Dict[str, Any]) -> List[Dict[str, str]]:
    if "messages" in ex and isinstance(ex["messages"], list):
        # already in chat format
        msgs = []
        for m in ex["messages"]:
            if not isinstance(m, dict):
                continue
            role = _normalize_role(m.get("role", ""))
            content = m.get("content", "")
            if content is None:
                continue
            content = str(content)
            if content.strip() == "":
                continue
            msgs.append({"role": role, "content": content})
        return msgs

    if "conversations" in ex:
        return _as_chat_messages_from_conversations(ex["conversations"])

    # Common instruction formats
    prompt = ex.get("prompt") or ex.get("instruction") or ex.get("query") or ex.get("question")
    completion = ex.get("completion") or ex.get("output") or ex.get("response") or ex.get("answer")
    if prompt is not None and completion is not None:
        return [
            {"role": "user", "content": str(prompt)},
            {"role": "assistant", "content": str(completion)},
        ]

    raise ValueError("Unsupported example schema (no messages/conversations/prompt+completion).")


def format_chatml(messages: List[Dict[str, str]], system_prompt: Optional[str]) -> str:
    text = ""
    if system_prompt:
        text += f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
    for m in messages:
        text += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
    return text


def _split_system(messages: List[Dict[str, str]], default_system: str) -> Tuple[Optional[str], List[Dict[str, str]]]:
    if messages and messages[0]["role"] == "system":
        sys = messages[0]["content"]
        rest = messages[1:]
        return sys, rest
    return default_system, messages


def _hash_text(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _count_tokens(tokenizer: Any, text: str, add_eos: bool) -> int:
    if add_eos and tokenizer.eos_token and not text.endswith(tokenizer.eos_token):
        text = text + tokenizer.eos_token
    return len(tokenizer.encode(text, add_special_tokens=False))


def _process_hf_dataset(
    dataset: Iterable[Dict[str, Any]],
    source: str,
    cfg: SFT3DataPrepConfig,
    tokenizer: Any,
    seen_hashes: Set[str],
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    skipped = {"invalid": 0, "duplicate": 0, "short_long": 0, "over_max_tokens": 0}

    for ex in tqdm(dataset, desc=f"Processing {source}"):
        try:
            msgs = extract_messages_generic(ex)
        except Exception:
            skipped["invalid"] += 1
            continue

        # enforce ending with assistant (for supervised loss masking)
        if not msgs or len(msgs) < 2:
            skipped["invalid"] += 1
            continue
        if msgs[-1]["role"] != "assistant":
            # drop trailing non-assistant
            msgs = msgs[:-1]
            if len(msgs) < 2 or msgs[-1]["role"] != "assistant":
                skipped["invalid"] += 1
                continue

        sys_prompt, msgs_wo_sys = _split_system(msgs, cfg.default_system_prompt)
        text = format_chatml(msgs_wo_sys, sys_prompt) if cfg.chat_template == "chatml" else format_chatml(msgs_wo_sys, sys_prompt)
        if len(text) < cfg.min_char_length or len(text) > cfg.max_char_length:
            skipped["short_long"] += 1
            continue

        tok = _count_tokens(tokenizer, text, cfg.add_eos)
        if cfg.max_length and tok > cfg.max_length:
            skipped["over_max_tokens"] += 1
            continue

        h = _hash_text(text)
        if h in seen_hashes:
            skipped["duplicate"] += 1
            continue
        seen_hashes.add(h)

        records.append({"text": text, "source": source, "token_count": tok})

    logger.info(f"{source}: kept={len(records)} skipped={skipped}")
    return records


def _load_stage2_replay_records(
    cfg: SFT3DataPrepConfig,
    tokenizer: Any,
    seen_hashes: Set[str],
) -> List[Dict[str, Any]]:
    stage2_path = os.path.join(cfg.stage2_data_root, cfg.stage2_train_file)
    if not os.path.exists(stage2_path):
        # optional auto-build of stage2 data (heavy but keeps replay exact)
        try:
            from sft.sft_2.prepare_data_sft2 import prepare_sft2_data, SFT2DataPrepConfig
            logger.warning(f"Stage2 replay file missing at {stage2_path}. Rebuilding data/sft2 via stage2 mixer...")
            os.makedirs(cfg.stage2_data_root, exist_ok=True)
            s2cfg = SFT2DataPrepConfig(
                output_dir=cfg.stage2_data_root,
                seed=cfg.seed,
                val_ratio=0.05,
                max_length=cfg.max_length,
                add_eos=cfg.add_eos,
                tokenizer_path=cfg.tokenizer_path,
            )
            prepare_sft2_data(s2cfg, tokenizer=tokenizer)
        except Exception as e:
            raise FileNotFoundError(
                f"Stage2 replay data not found at {stage2_path} and auto-build failed: {e}"
            ) from e

    records: List[Dict[str, Any]] = []
    skipped = {"invalid": 0, "duplicate": 0, "short_long": 0, "over_max_tokens": 0}
    with open(stage2_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                text = obj.get("text")
                if not isinstance(text, str):
                    skipped["invalid"] += 1
                    continue
            except Exception:
                skipped["invalid"] += 1
                continue

            if len(text) < cfg.min_char_length or len(text) > cfg.max_char_length:
                skipped["short_long"] += 1
                continue

            tok = _count_tokens(tokenizer, text, cfg.add_eos)
            if cfg.max_length and tok > cfg.max_length:
                skipped["over_max_tokens"] += 1
                continue

            h = _hash_text(text)
            if h in seen_hashes:
                skipped["duplicate"] += 1
                continue
            seen_hashes.add(h)
            records.append({"text": text, "source": "stage2_replay", "token_count": tok})

    logger.info(f"stage2_replay: kept={len(records)} skipped={skipped}")
    return records


def _sum_tokens(records: List[Dict[str, Any]]) -> int:
    return sum(int(r.get("token_count", 0)) for r in records)


def _select_by_token_budget(records: List[Dict[str, Any]], target_tokens: int, rng: random.Random) -> Tuple[List[Dict[str, Any]], int]:
    rng.shuffle(records)
    out: List[Dict[str, Any]] = []
    total = 0
    for r in records:
        out.append(r)
        total += int(r["token_count"])
        if total >= target_tokens:
            break
    return out, total


def prepare_sft3_data(cfg: SFT3DataPrepConfig) -> None:
    rng = random.Random(cfg.seed)
    tokenizer = _load_tokenizer(cfg)
    seen_hashes: Set[str] = set()

    ratios = {
        "no_robots": cfg.no_robots_ratio,
        "lima": cfg.lima_ratio,
        "stage2_replay": cfg.stage2_replay_ratio,
    }
    total_ratio = sum(ratios.values())
    if abs(total_ratio - 1.0) > 1e-6:
        logger.warning(f"Ratios sum to {total_ratio:.3f}; normalizing to 1.0")
        ratios = {k: v / total_ratio for k, v in ratios.items()}

    logger.info("Loading calibration datasets...")
    hf_token = _get_hf_token()
    no_robots = load_dataset(
        "HuggingFaceH4/no_robots",
        split="train",
        trust_remote_code=cfg.trust_remote_code,
        token=hf_token,
    )
    lima_available = True
    try:
        lima = load_dataset(
            "GAIR/lima",
            split="train",
            trust_remote_code=cfg.trust_remote_code,
            token=hf_token,
        )
    except Exception as e:
        msg = str(e)
        if "401" in msg or "gated" in msg.lower() or "restricted" in msg.lower():
            logger.warning(
                "LIMA gated or no access; skipping 'GAIR/lima' and redistributing its ratio to no_robots. "
                "Accept access on HuggingFace and provide HF_TOKEN/HUGGINGFACE_HUB_TOKEN to include LIMA."
            )
            lima_available = False
            lima = None
        else:
            raise

    no_robots_records = _process_hf_dataset(no_robots, "no_robots", cfg, tokenizer, seen_hashes)
    lima_records = _process_hf_dataset(lima, "lima", cfg, tokenizer, seen_hashes) if lima_available else []
    stage2_records = _load_stage2_replay_records(cfg, tokenizer, seen_hashes)

    pools = {"no_robots": no_robots_records, "stage2_replay": stage2_records}
    ratios = {
        "no_robots": cfg.no_robots_ratio + (0 if lima_available else cfg.lima_ratio),
        "stage2_replay": cfg.stage2_replay_ratio,
    }
    if lima_available:
        pools["lima"] = lima_records
        ratios["lima"] = cfg.lima_ratio

    total_ratio = sum(ratios.values())
    if total_ratio <= 0:
        raise ValueError("No ratios available after filtering pools.")
    if abs(total_ratio - 1.0) > 1e-6:
        logger.warning(f"Ratios sum to {total_ratio:.3f}; normalizing to 1.0")
        ratios = {k: v / total_ratio for k, v in ratios.items()}

    pool_tokens = {k: _sum_tokens(v) for k, v in pools.items()}
    for k, t in pool_tokens.items():
        if t <= 0:
            raise ValueError(f"No usable tokens for pool '{k}'. Check extraction/filters.")

    # Strict ratio without oversampling: limit total by the smallest pool.
    total_possible = min(pool_tokens[k] / ratios[k] for k in pools.keys())
    target_tokens = {k: int(total_possible * ratios[k]) for k in pools.keys()}

    selected = {}
    selected_tokens = {}
    for k in pools.keys():
        selected[k], selected_tokens[k] = _select_by_token_budget(pools[k], target_tokens[k], rng)

    all_records = selected["no_robots"] + selected.get("lima", []) + selected["stage2_replay"]
    rng.shuffle(all_records)

    val_size = int(len(all_records) * cfg.val_ratio)
    val_records = all_records[:val_size]
    train_records = all_records[val_size:]

    logger.info(f"Final mix: train={len(train_records)} val={len(val_records)}")
    logger.info(f"Token targets: {target_tokens}")
    logger.info(f"Token actuals: {selected_tokens}")

    os.makedirs(cfg.output_dir, exist_ok=True)
    for name, data in [("train", train_records), ("val", val_records)]:
        path = os.path.join(cfg.output_dir, f"{name}.jsonl")
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
    logger.info(f"Saved SFT3 data to {cfg.output_dir}")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/sft3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--tokenizer_path", type=str, default="/from_s3/sft2_model/")
    parser.add_argument("--no_robots_ratio", type=float, default=0.60)
    parser.add_argument("--lima_ratio", type=float, default=0.15)
    parser.add_argument("--stage2_replay_ratio", type=float, default=0.25)
    parser.add_argument("--stage2_data_root", type=str, default="data/sft2")
    parser.add_argument("--stage2_train_file", type=str, default="train.jsonl")
    args = parser.parse_args()

    cfg = SFT3DataPrepConfig(
        output_dir=args.output_dir,
        seed=args.seed,
        val_ratio=args.val_ratio,
        max_length=args.max_length,
        tokenizer_path=args.tokenizer_path,
        no_robots_ratio=args.no_robots_ratio,
        lima_ratio=args.lima_ratio,
        stage2_replay_ratio=args.stage2_replay_ratio,
        stage2_data_root=args.stage2_data_root,
        stage2_train_file=args.stage2_train_file,
    )
    prepare_sft3_data(cfg)


if __name__ == "__main__":
    main()

