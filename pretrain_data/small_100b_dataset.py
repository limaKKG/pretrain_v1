import os
import json
import time
import random
import gzip
import shutil
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed, Future, wait, FIRST_COMPLETED
os.environ.setdefault("AWS_REQUEST_CHECKSUM_CALCULATION", "when_required")
os.environ.setdefault("AWS_RESPONSE_CHECKSUM_VALIDATION", "when_required")
import boto3
from botocore.config import Config as BotoConfig
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

TARGET_TOTAL_TOKENS = int(os.environ.get("TARGET_TOTAL_TOKENS", "100000000000"))
TOKENS_PER_SHARD = int(os.environ.get("TOKENS_PER_SHARD", str(200_000_000))) 
TOKENIZE_BATCH_SIZE = int(os.environ.get("TOKENIZE_BATCH_SIZE", "1024")) 
COMPRESS = os.environ.get("COMPRESS", "gzip").lower() 
DISK_SOFT_LIMIT_GB = float(os.environ.get("DISK_SOFT_LIMIT_GB", "350"))
MAX_PENDING_UPLOADS = int(os.environ.get("MAX_PENDING_UPLOADS", "3"))
DATASET_NAME = os.environ.get("DATASET_NAME", "togethercomputer/RedPajama-Data-1T")
DATASET_CONFIG = os.environ.get("DATASET_CONFIG", "default")
DATASET_SPLIT = os.environ.get("DATASET_SPLIT", "train")
DATASET_TRUST_REMOTE_CODE = os.environ.get("DATASET_TRUST_REMOTE_CODE", "1") in ("1", "true", "True", "yes", "YES")
S3_BUCKET = os.environ["S3_BUCKET"]
S3_PREFIX = os.environ.get("S3_PREFIX", "data/pretrain_sample").strip("/")
S3_ENDPOINT = os.environ.get("S3_ENDPOINT")
UPLOAD_FILE_WORKERS = int(os.environ.get("UPLOAD_FILE_WORKERS", "8"))         
MULTIPART_CHUNK_MB = int(os.environ.get("MULTIPART_CHUNK_MB", "128")) 
DELETE_LOCAL_AFTER_UPLOAD = os.environ.get("DELETE_LOCAL_AFTER_UPLOAD", "1") == "1"
S3_FORCE_MANUAL_MULTIPART = os.environ.get("S3_FORCE_MANUAL_MULTIPART", "1") in ("1", "true", "True", "yes", "YES")
S3_SINGLE_PUT_MAX_BYTES = int(os.environ.get("S3_SINGLE_PUT_MAX_BYTES", str(int(4.9 * 1024**3))) )
LOCAL_OUT = Path(os.environ.get("LOCAL_OUT", "./redpajama_100b"))
LOCAL_OUT.mkdir(parents=True, exist_ok=True)
TOKENIZER_NAME = os.environ.get("TOKENIZER_NAME", "meta-llama/Llama-2-7b-hf")
TOKENIZER_S3_BUCKET = os.environ.get("TOKENIZER_S3_BUCKET")
TOKENIZER_S3_PREFIX = os.environ.get("TOKENIZER_S3_PREFIX") 
TOKENIZER_S3_ENDPOINT = os.environ.get("S3_ENDPOINT")
TOKENIZER_LOCAL_DIR = Path(os.environ.get("TOKENIZER_LOCAL_DIR", "/from_s3/model"))

def _download_s3_prefix(bucket: str, prefix: str, local_dir: Path, endpoint_url: Optional[str] = None) -> None:
    session = boto3.session.Session()
    s3 = session.client("s3", endpoint_url=endpoint_url) if endpoint_url else session.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    local_dir.mkdir(parents=True, exist_ok=True)
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            rel_path = key[len(prefix):].lstrip("/")
            target_path = local_dir / rel_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, key, str(target_path))
            print(f"Downloaded s3://{bucket}/{key} -> {target_path}")


def _load_tokenizer() -> AutoTokenizer:
    name_path = Path(TOKENIZER_NAME)
    if name_path.exists():
        print(f"Loading tokenizer from local path: {name_path}")
        return AutoTokenizer.from_pretrained(name_path, use_fast=True, trust_remote_code=True)

    if TOKENIZER_LOCAL_DIR.exists():
        print(f"Loading tokenizer from local dir: {TOKENIZER_LOCAL_DIR}")
        return AutoTokenizer.from_pretrained(TOKENIZER_LOCAL_DIR, use_fast=True, trust_remote_code=True)

    if TOKENIZER_S3_BUCKET and TOKENIZER_S3_PREFIX:
        print(f"Downloading tokenizer from s3://{TOKENIZER_S3_BUCKET}/{TOKENIZER_S3_PREFIX} to {TOKENIZER_LOCAL_DIR}")
        _download_s3_prefix(
            bucket=TOKENIZER_S3_BUCKET,
            prefix=TOKENIZER_S3_PREFIX,
            local_dir=TOKENIZER_LOCAL_DIR,
            endpoint_url=TOKENIZER_S3_ENDPOINT,
        )
        print(f"Loading tokenizer from {TOKENIZER_LOCAL_DIR}")
        return AutoTokenizer.from_pretrained(TOKENIZER_LOCAL_DIR, use_fast=True, trust_remote_code=True)

    print(f"Loading tokenizer from HF ID: {TOKENIZER_NAME}")
    return AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True, trust_remote_code=True)


tokenizer = _load_tokenizer()
try:
    tokenizer.model_max_length = int(1e30)
except Exception:
    pass

def _shard_suffix() -> str:
    return ".jsonl.gz" if COMPRESS == "gzip" else ".jsonl"

def _open_shard(path: Path):
    if COMPRESS == "gzip":
        return gzip.open(path, "wt", encoding="utf-8", compresslevel=1)
    return path.open("w", encoding="utf-8", buffering=1024 * 1024)

SOURCE_PROPS = {
    "common_crawl": 0.67, 
    "c4": 0.15,         
    "github": 0.045,     
    "wikipedia": 0.045,  
    "arxiv": 0.025,      
    "stackexchange": 0.02 
}

SOURCE_ALIASES: Dict[str, List[str]] = {
    "common_crawl": ["common_crawl", "cc"],
    "c4": ["c4"],
    "github": ["github"],
    "wikipedia": ["wikipedia", "wiki"],
    "book": ["book", "books"],
    "arxiv": ["arxiv"],
    "stackexchange": ["stackexchange", "stack_exchange"],
}

def _make_s3_client():
    session = boto3.session.Session()
    return session.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        config=BotoConfig(
            retries={"max_attempts": 20, "mode": "adaptive"},
        ),
    )


class S3Uploader:
    def __init__(self, bucket: str, prefix: str, local_root: Path) -> None:
        self.bucket = bucket
        self.prefix = prefix
        self.local_root = local_root
        self._s3 = _make_s3_client()
        self._pool = ThreadPoolExecutor(max_workers=UPLOAD_FILE_WORKERS)
        self._futures: List[Future] = []

    def _upload_manual(self, path: Path, key: str) -> None:
        size = path.stat().st_size
        chunk_size = MULTIPART_CHUNK_MB * 1024 * 1024

        if size == 0:
            self._s3.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=b"",
                ContentLength=0,
            )
            return

        if (not S3_FORCE_MANUAL_MULTIPART) and size <= S3_SINGLE_PUT_MAX_BYTES:
            with path.open("rb") as f:
                self._s3.put_object(
                    Bucket=self.bucket,
                    Key=key,
                    Body=f,
                    ContentLength=size,
                )
            return

        mp = self._s3.create_multipart_upload(Bucket=self.bucket, Key=key)
        upload_id = mp["UploadId"]
        parts: List[Dict[str, object]] = []
        part_number = 1

        try:
            with path.open("rb") as f:
                while True:
                    data = f.read(chunk_size)
                    if not data:
                        break
                    resp = self._s3.upload_part(
                        Bucket=self.bucket,
                        Key=key,
                        UploadId=upload_id,
                        PartNumber=part_number,
                        Body=data,
                        ContentLength=len(data),
                    )
                    parts.append({"ETag": resp["ETag"], "PartNumber": part_number})
                    part_number += 1

            self._s3.complete_multipart_upload(
                Bucket=self.bucket,
                Key=key,
                UploadId=upload_id,
                MultipartUpload={"Parts": parts},
            )
        except Exception:
            try:
                self._s3.abort_multipart_upload(Bucket=self.bucket, Key=key, UploadId=upload_id)
            except Exception:
                pass
            raise

    def _disk_used_gb(self) -> float:
        return shutil.disk_usage(self.local_root).used / (1024 ** 3)

    def _drain_completed(self) -> None:
        keep: List[Future] = []
        for f in self._futures:
            if f.done():
                key = f.result() 
                print(f"Uploaded -> s3://{self.bucket}/{key}")
            else:
                keep.append(f)
        self._futures = keep

    def maybe_backpressure(self) -> None:
        self._drain_completed()
        while self._futures and (
            len(self._futures) >= MAX_PENDING_UPLOADS or self._disk_used_gb() >= DISK_SOFT_LIMIT_GB
        ):
            wait(self._futures, return_when=FIRST_COMPLETED)
            self._drain_completed()

    def submit(self, path: Path) -> None:
        rel = path.relative_to(self.local_root)
        key = f"{self.prefix}/{rel.as_posix()}"

        def _do_upload():
            self._upload_manual(path, key)
            if DELETE_LOCAL_AFTER_UPLOAD:
                path.unlink(missing_ok=True)
            return key

        self._futures.append(self._pool.submit(_do_upload))
        self.maybe_backpressure()

    def wait(self) -> None:
        while self._futures:
            wait(self._futures, return_when=FIRST_COMPLETED)
            self._drain_completed()
        self._pool.shutdown(wait=True)

def _is_retryable_error(e: Exception) -> bool:
    """
    Расширенный список ошибок, которые считаем временными:
    429 (rate limit), 500/502/503/504 (gateway / server).
    """
    msg = str(e).lower()
    retryable_codes = ["429", "500", "502", "503", "504"]
    retryable_msgs = [
        "too many requests",
        "bad gateway",
        "gateway timeout",
        "service unavailable",
        "internal server error",
    ]
    return any(code in msg for code in retryable_codes) or any(m in msg for m in retryable_msgs)


def _get_existing_shards(source: str) -> List[int]:
    """
    Возвращает список shard_id, которые уже есть в S3. Нужно, чтобы:
    1) не перезаливать то, что уже там лежит;
    2) не терять прогресс, если джоба упала на середине.
    """
    client = _make_s3_client()
    prefix = f"{S3_PREFIX}/{source}/part-"
    ids: List[int] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            try:
                part_str = key.split("part-")[-1].split(".")[0]
                ids.append(int(part_str))
            except Exception:
                continue
    return sorted(set(ids))


def _batch_token_lengths(texts: List[str]) -> List[int]:
    enc = tokenizer(
        texts,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
        return_length=True,
    )
    if "length" in enc:
        return list(enc["length"])
    return [len(x) for x in enc["input_ids"]]

def _load_rpj_stream(source: str):
    trust = True if DATASET_TRUST_REMOTE_CODE else False
    last_err: Optional[Exception] = None
    candidates = SOURCE_ALIASES.get(source, [source])
    for cfg in candidates:
        try:
            return load_dataset(
                DATASET_NAME,
                cfg,
                split=DATASET_SPLIT,
                streaming=True,
                trust_remote_code=trust,
            )
        except Exception as e:
            last_err = e

    for cfg in candidates:
        try:
            return load_dataset(
                DATASET_NAME,
                DATASET_CONFIG,
                data_dir=cfg,
                split=DATASET_SPLIT,
                streaming=True,
                trust_remote_code=trust,
            )
        except Exception as e:
            last_err = e

    raise RuntimeError(
        f"Failed to load dataset '{DATASET_NAME}' for source='{source}'. "
        f"Tried configs={candidates} and DATASET_CONFIG='{DATASET_CONFIG}'. "
        f"Last error: {last_err}"
    )


def stream_source(source: str, target_tokens: int, shard_tokens: int, uploader: S3Uploader) -> int:
    existing_shards = _get_existing_shards(source)
    ds = _load_rpj_stream(source)
    it = iter(ds)

    # Прогресс по уже загруженным шардам: считаем, что каждый полный шард имеет shard_tokens.
    written_tokens = len(existing_shards) * shard_tokens
    shard_id = 0
    current_tokens = 0
    shard_path = LOCAL_OUT / source / f"part-{shard_id:05d}{_shard_suffix()}"
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    shard_file = _open_shard(shard_path)
    pbar = tqdm(total=target_tokens, desc=f"{source}", unit="tok")
    batch_texts: List[str] = []
    backoff = 1.0
    max_backoff = 60.0

    def _rotate_shard() -> None:
        nonlocal shard_id, shard_path, shard_file, current_tokens
        shard_file.close()
        if shard_id in existing_shards:
            # уже в S3 — удаляем локально и не заливаем повторно
            shard_path.unlink(missing_ok=True)
        else:
            uploader.submit(shard_path)
            uploader.maybe_backpressure()
        shard_id += 1
        shard_path = LOCAL_OUT / source / f"part-{shard_id:05d}{_shard_suffix()}"
        shard_path.parent.mkdir(parents=True, exist_ok=True)
        shard_file = _open_shard(shard_path)
        current_tokens = 0

    while written_tokens < target_tokens:
        try:
            example = next(it)
            backoff = 1.0
        except StopIteration:
            break
        except Exception as e:
            if _is_retryable_error(e):
                sleep_s = backoff * (0.75 + 0.5 * random.random())
                time.sleep(sleep_s)
                backoff = min(max_backoff, backoff * 2)
                continue
            raise

        text = example.get("text", "")
        if not text:
            continue

        batch_texts.append(text)
        if len(batch_texts) < TOKENIZE_BATCH_SIZE:
            continue

        lengths = _batch_token_lengths(batch_texts)
        for t, n_tokens in zip(batch_texts, lengths):
            if n_tokens <= 0:
                continue
            if current_tokens + n_tokens > shard_tokens:
                _rotate_shard()

            shard_file.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
            current_tokens += n_tokens
            written_tokens += n_tokens
            pbar.update(n_tokens)

            if written_tokens >= target_tokens:
                break

        batch_texts = []
        uploader.maybe_backpressure()

    if batch_texts and written_tokens < target_tokens:
        lengths = _batch_token_lengths(batch_texts)
        for t, n_tokens in zip(batch_texts, lengths):
            if n_tokens <= 0:
                continue
            if current_tokens + n_tokens > shard_tokens:
                _rotate_shard()
            shard_file.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
            current_tokens += n_tokens
            written_tokens += n_tokens
            pbar.update(n_tokens)
            if written_tokens >= target_tokens:
                break

    shard_file.close()
    if shard_id in existing_shards:
        shard_path.unlink(missing_ok=True)
    else:
        uploader.submit(shard_path)
    pbar.close()
    return written_tokens

def main():
    targets: Dict[str, int] = {
        src: int(TARGET_TOTAL_TOKENS * prop)
        for src, prop in SOURCE_PROPS.items()
    }
    total = sum(targets.values())
    if total != TARGET_TOTAL_TOKENS:
        last = next(reversed(targets))
        targets[last] += TARGET_TOTAL_TOKENS - total
    uploader = S3Uploader(bucket=S3_BUCKET, prefix=S3_PREFIX, local_root=LOCAL_OUT)
    summary: Dict[str, int] = {}
    for source, tgt in targets.items():
        print(f"Processing {source}: target {tgt/1e9:.2f}B tokens")
        written = stream_source(source, tgt, TOKENS_PER_SHARD, uploader=uploader)
        summary[source] = int(written)

    print("Waiting uploads to finish...")
    uploader.wait()

    summary_path = LOCAL_OUT / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "target_total_tokens": TARGET_TOTAL_TOKENS,
                "actual_total_tokens": int(sum(summary.values())),
                "per_source_tokens": summary,
                "tokens_per_shard": TOKENS_PER_SHARD,
                "tokenize_batch_size": TOKENIZE_BATCH_SIZE,
                "tokenizer_name": TOKENIZER_NAME,
                "s3_bucket": S3_BUCKET,
                "s3_prefix": S3_PREFIX,
                "delete_local_after_upload": DELETE_LOCAL_AFTER_UPLOAD,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Wrote summary to {summary_path}")
    print(f"Done. Dataset uploaded to s3://{S3_BUCKET}/{S3_PREFIX}/")

if __name__ == "__main__":
    main()