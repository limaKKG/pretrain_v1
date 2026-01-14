import os
import glob
import random
import json
import gzip
from typing import Any, Dict, Iterable, Optional, List
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
import torch 

class JsonlStreamingDataset(IterableDataset):
    def __init__(self, files: List[str], val_take: int = 0, is_val: bool = False, shard_id: int = 0, num_shards: int = 1):
        self.files = sorted(files)
        self.val_take = val_take
        self.is_val = is_val
        self.shard_id = shard_id
        self.num_shards = num_shards

    def shard(self, num_shards: int, index: int):
        return JsonlStreamingDataset(
            files=self.files,
            val_take=self.val_take,
            is_val=self.is_val,
            shard_id=index,
            num_shards=num_shards
        )

    def __iter__(self) -> Iterable[Dict[str, Any]]:
        num_shards = max(1, int(self.num_shards))
        shard_id = int(self.shard_id) % num_shards
        my_files = [f for i, f in enumerate(self.files) if i % num_shards == shard_id]

        share: Optional[int] = None
        if self.val_take and self.val_take > 0:
            share = (int(self.val_take) + num_shards - 1) // num_shards

        count = 0
        for file_path in my_files:
            opener = gzip.open if file_path.endswith(".gz") else open
            mode = "rt" if file_path.endswith(".gz") else "r"
            
            with opener(file_path, mode, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue

                    if share is not None:
                        if self.is_val:
                            if count >= share:
                                return
                        else:
                            if count < share:
                                count += 1
                                continue
                    
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    
                    count += 1
                    if self.is_val and share is not None and count >= share:
                        return

class StreamingDatasetWrapper(IterableDataset):
    def __init__(
        self,
        dataset,
        tokenizer: Any,
        block_size: int,
        *,
        shuffle_buffer: int = 0,
        batch_process_size: int = 256,
        seed: int = 42,
        text_field: str = "text",
        add_eos: bool = True,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.shuffle_buffer = shuffle_buffer
        self.batch_process_size = batch_process_size
        self.seed = seed
        self.text_field = text_field
        self.add_eos = add_eos

    def __iter__(self) -> Iterable[Dict[str, Any]]:
        ds = self.dataset

        def _shard_iter(it, num_shards: int, index: int):
            for i, x in enumerate(it):
                if i % num_shards == index:
                    yield x

        def _buffer_shuffle(it, buffer_size: int, seed: int):
            rng = random.Random(seed)
            buf = []
            for x in it:
                if len(buf) < buffer_size:
                    buf.append(x)
                    continue
                j = rng.randrange(buffer_size)
                yield buf[j]
                buf[j] = x
            rng.shuffle(buf)
            for x in buf:
                yield x

        rank = 0
        world_size = 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()

        worker = get_worker_info()
        num_workers = worker.num_workers if worker is not None else 1
        worker_id = worker.id if worker is not None else 0
        combined_shards = world_size * num_workers
        combined_index = rank * num_workers + worker_id
        if combined_shards > 1:
            if hasattr(ds, "shard"):
                ds = ds.shard(num_shards=combined_shards, index=combined_index)
            else:
                ds = _shard_iter(ds, combined_shards, combined_index)

        if self.shuffle_buffer and self.shuffle_buffer > 0:
            seed = self.seed + rank + (worker_id * 997)
            if hasattr(ds, "shuffle"):
                ds = ds.shuffle(buffer_size=self.shuffle_buffer, seed=seed)
            else:
                ds = _buffer_shuffle(ds, self.shuffle_buffer, seed)

        token_iterator = self._token_generator(ds)

        chunk: List[int] = []
        for tok in token_iterator:
            chunk.append(tok)
            while len(chunk) >= self.block_size:
                blk = chunk[: self.block_size]
                yield {"input_ids": blk, "labels": blk}
                chunk = chunk[self.block_size:]

    def _token_generator(self, ds) -> Iterable[int]:
        iterator = iter(ds)
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        detected_field: Optional[str] = None
        seen = 0
        skipped_no_text = 0
        max_skips_before_error = 5000
        last_keys: Optional[List[str]] = None
        max_chars_per_tokenize_call = 20_000

        def _iter_text_chunks(text: str, max_chars: int) -> Iterable[str]:
            if not text:
                return
            n = len(text)
            if n <= max_chars:
                yield text
                return
            start = 0
            while start < n:
                end = min(n, start + max_chars)
                if end < n:
                    window = text[start:end]
                    cut = max(window.rfind("\n"), window.rfind(" "), window.rfind("\t"))
                    if cut != -1 and cut > (max_chars // 2):
                        end = start + cut + 1
                yield text[start:end]
                start = end

        while True:
            try:
                item = next(iterator)
            except StopIteration:
                break

            seen += 1
            if not isinstance(item, dict):
                item = {"text": str(item)}
            last_keys = list(item.keys())
            text: Optional[str] = None

            if detected_field is not None:
                v = item.get(detected_field)
                if v is not None:
                    text = v if isinstance(v, str) else str(v)
                    if not text:
                        text = None

            if text is None:
                v = item.get(self.text_field)
                if v is not None:
                    text = v if isinstance(v, str) else str(v)
                    if text:
                        detected_field = self.text_field
                    else:
                        text = None

            if text is None:
                for k in ("text", "content", "document", "raw_content"):
                    v = item.get(k)
                    if v is None:
                        continue
                    cand = v if isinstance(v, str) else str(v)
                    if cand:
                        text = cand
                        detected_field = k
                        break

            if text is None:
                for k, v in item.items():
                    if v is None:
                        continue
                    cand = v if isinstance(v, str) else str(v)
                    if cand:
                        text = cand
                        detected_field = k
                        break

            if text is None:
                skipped_no_text += 1
                if skipped_no_text >= max_skips_before_error:
                    raise RuntimeError(
                        "Dataset yielded no usable text.\n"
                        f"Tried text_field='{self.text_field}' + fallbacks ('text','content',...).\n"
                        f"Last example keys: {last_keys}\n"
                        "Fix: set correct column by passing text_field to StreamingDatasetWrapper."
                    )
                continue

            skipped_no_text = 0

            for piece in _iter_text_chunks(text, max_chars=max_chars_per_tokenize_call):
                if not piece:
                    continue
                ids = self.tokenizer(
                    piece,
                    add_special_tokens=False,
                    return_attention_mask=False,
                )["input_ids"]
                if ids and isinstance(ids[0], list):
                    ids = ids[0]
                for tok in ids:
                    yield tok

            if self.add_eos and eos_id is not None:
                yield eos_id

        if seen == 0:
            raise RuntimeError("Dataset stream is empty (0 examples). Check your local JSONL files and text_field.")

class DataCollator:
    def __init__(self, tokenizer: Any, pad_to_multiple_of: Optional[int] = None) -> None:
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, batch: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        input_ids = [item['input_ids'] for item in batch]
        labels = [item['labels'] for item in batch]
        batch_input_ids = torch.tensor(input_ids, dtype=torch.long)
        batch_labels = torch.tensor(labels, dtype=torch.long)

        return {
            "input_ids": batch_input_ids, 
            "labels": batch_labels,
            "attention_mask": torch.ones_like(batch_input_ids) 
        }


class DataModule:
    def __init__(
        self,
        dataset_name: str,
        data_root: Optional[str],
        sources: Optional[List[str]],
        file_glob: str,
        tokenizer: Any,
        block_size: int,
        train_split: str="train",
        val_split: str="validation",
        batch_size: int= 4,
        num_workers: int = 2,
        shuffle_buffer: int = 10000,
        prefetch_factor: int = 2,
        dataset_config: Optional[str] = None, 
        trust_remote_code: bool = False, 
        val_take: int = 1000,
        text_field: str = "text",
    ) -> None:
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.trust_remote_code = trust_remote_code
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.train_split = train_split
        self.val_split = val_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle_buffer = shuffle_buffer
        self.prefetch_factor = prefetch_factor
        self.data_root = data_root
        self.sources = sources
        self.file_glob = file_glob
        self.val_take = val_take
        self.text_field = text_field
        self.train_dataset = None
        self.val_dataset = None

    def _resolve_local_files(self) -> List[str]:
        if not self.data_root:
            return []
        if not os.path.isdir(self.data_root):
            raise FileNotFoundError(
                f"Local data_root not found: {self.data_root}. "
                "Ensure s3 sync placed shards there."
            )
        sources = self.sources
        if not sources:
            sources = [
                d for d in os.listdir(self.data_root)
                if os.path.isdir(os.path.join(self.data_root, d))
            ]
        files: List[str] = []
        for src in sources:
            pattern = os.path.join(self.data_root, src, self.file_glob)
            matched = sorted(glob.glob(pattern))
            files.extend(matched)
        if not files:
            raise RuntimeError(
                f"No data files matched under {self.data_root} with pattern '{self.file_glob}'. "
                f"Checked sources={sources}"
            )
        return files

    def prepare(self) -> None:
        local_files = self._resolve_local_files()
        
        if not local_files:
            raise RuntimeError(
                f"No local data files found in {self.data_root}. "
                "Hugging Face datasets are disabled; please provide local JSONL data."
            )

        val_stream = JsonlStreamingDataset(local_files, val_take=self.val_take, is_val=True)
        train_stream = JsonlStreamingDataset(local_files, val_take=self.val_take, is_val=False)

        self.val_dataset = StreamingDatasetWrapper(
            dataset=val_stream,
            tokenizer=self.tokenizer,
            block_size=self.block_size,
            shuffle_buffer=0,             
            batch_process_size=256,
            text_field=self.text_field,
        )

        self.train_dataset = StreamingDatasetWrapper(
            dataset=train_stream,
            tokenizer=self.tokenizer,
            block_size=self.block_size,
            shuffle_buffer=self.shuffle_buffer, 
            batch_process_size=256,
            text_field=self.text_field,
        )

    def train_dataloader(self) -> Any:
        collator = DataCollator(self.tokenizer)
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=collator,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            pin_memory=True,
            persistent_workers=False, 
            drop_last=True
        )

    def val_dataloader(self) -> Any:
        collator = DataCollator(self.tokenizer)
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=collator,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            pin_memory=True,
            persistent_workers=False,
            drop_last=False
        )