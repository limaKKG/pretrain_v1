import json
import os
import random
from typing import Any, Dict, List, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

class SFTDataset(Dataset):
    def __init__(
        self,
        data_files: List[str],
        tokenizer: Any,
        max_length: int,
        text_field: str,
        add_eos: bool,
        mask_user: bool,
        chat_template: str = "chatml",
        pack: bool = True,
        seed: int = 42,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_field = text_field
        self.add_eos = add_eos
        self.mask_user = mask_user
        self.chat_template = chat_template
        self.pack = pack
        self.seed = seed
        if chat_template == "llama3":
            self.assistant_start = "<|start_header_id|>assistant<|end_header_id|>\n\n"
            self.assistant_end = "<|eot_id|>"
        elif chat_template == "chatml":
            self.assistant_start = "<|im_start|>assistant\n"
            self.assistant_end = "<|im_end|>"
        else:
            self.assistant_start = "Assistant: "
            self.assistant_end = "\n"
        self._warned_offsets = False

        raw_records = []
        for file_path in data_files:
            if not os.path.exists(file_path):
                continue
            with open(file_path, "r") as f:
                for line in f:
                    raw_records.append(json.loads(line))
        if (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0:
            split_name = "training" if any("train" in os.path.basename(p) for p in data_files) else "validation"
            print(f"Loaded {len(raw_records)} conversations for {split_name}")
        random.Random(self.seed).shuffle(raw_records) 

        if self.pack:
            self.samples = self._pack_samples(raw_records)
        else:
            self.samples = [self._process_single(r) for r in raw_records]

    def _process_single(self, record: Dict[str, Any]) -> Dict[str, Any]:
        text = record[self.text_field]
        if self.add_eos and not text.endswith(self.tokenizer.eos_token):
            text += self.tokenizer.eos_token
            
        offsets = None
        if getattr(self.tokenizer, "is_fast", False):
            try:
                enc = self.tokenizer(
                    text, 
                    add_special_tokens=False, 
                    return_offsets_mapping=True
                )
                input_ids = enc["input_ids"]
                offsets = enc["offset_mapping"]
            except (ValueError, TypeError, KeyError, NotImplementedError):
                enc = self.tokenizer(text, add_special_tokens=False)
                input_ids = enc["input_ids"]
                offsets = None
        else:
            enc = self.tokenizer(text, add_special_tokens=False)
            input_ids = enc["input_ids"]
        
        labels = list(input_ids)
        
        if self.mask_user:
            new_labels = [-100] * len(labels)
            start_search = 0
            while True:
                idx_start = text.find(self.assistant_start, start_search)
                if idx_start == -1:
                    break

                content_start = idx_start + len(self.assistant_start)
                idx_end = text.find(self.assistant_end, content_start)
                content_end = idx_end + len(self.assistant_end) if idx_end != -1 else len(text)

                if offsets is not None:
                    for i, (start, end) in enumerate(offsets):
                        if start < content_end and end > content_start:
                            new_labels[i] = labels[i]
                else:
                    if not self._warned_offsets and ((not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0):
                        print("Warning: Using approximate masking (offset_mapping not supported by tokenizer).")
                        self._warned_offsets = True
                    token_start = len(self.tokenizer.encode(text[:content_start], add_special_tokens=False))
                    token_end = len(self.tokenizer.encode(text[:content_end], add_special_tokens=False))
                    for i in range(min(token_start, len(labels)), min(token_end, len(labels))):
                        new_labels[i] = labels[i]

                if idx_end == -1:
                    break
                start_search = content_end
            labels = new_labels
            
        return {
            "input_ids": input_ids,
            "labels": labels,
        }

    def _pack_samples(self, raw_records: List[Dict[str, Any]]) -> List[Dict[str, torch.Tensor]]:
        packed_samples = []
        current_ids = []
        current_labels = []
        
        for record in raw_records:
            processed = self._process_single(record)
            ids, labels = processed["input_ids"], processed["labels"]
            
            if len(ids) > self.max_length:
                ids = ids[:self.max_length]
                labels = labels[:self.max_length]

            if len(current_ids) + len(ids) > self.max_length:
                packed_samples.append({
                    "input_ids": torch.tensor(current_ids, dtype=torch.long),
                    "labels": torch.tensor(current_labels, dtype=torch.long)
                })
                current_ids = []
                current_labels = []
            
            current_ids.extend(ids)
            current_labels.extend(labels)
        
        if current_ids:
            packed_samples.append({
                "input_ids": torch.tensor(current_ids, dtype=torch.long),
                "labels": torch.tensor(current_labels, dtype=torch.long)
            })
            
        return packed_samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]


class SFTDataCollator:
    def __init__(self, tokenizer: Any, pad_to_multiple_of: Optional[int] = 8) -> None:
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids = [item["input_ids"] for item in batch]
        labels = [item["labels"] for item in batch]
        max_batch_len = max(len(x) for x in input_ids)
        if self.pad_to_multiple_of:
            max_batch_len = ((max_batch_len + self.pad_to_multiple_of - 1) // 
                             self.pad_to_multiple_of * self.pad_to_multiple_of)
        padded_ids = []
        padded_labels = []
        for ids, labs in zip(input_ids, labels):
            pad_len = max_batch_len - len(ids)
            padded_ids.append(torch.cat([ids, torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=torch.long)]))
            padded_labels.append(torch.cat([labs, torch.full((pad_len,), -100, dtype=torch.long)]))
            
        input_ids = torch.stack(padded_ids)
        labels = torch.stack(padded_labels)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()
            
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


class SFTDataModule:
    def __init__(
        self,
        train_files: List[str],
        val_files: List[str],
        tokenizer: Any,
        max_length: int,
        batch_size: int,
        num_workers: int,
        seed: int,
        text_field: str,
        add_eos: bool,
        mask_user: bool,
        chat_template: str = "chatml",
        pack: bool = True,
    ) -> None:
        self.train_files = train_files
        self.val_files = val_files
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.text_field = text_field
        self.add_eos = add_eos
        self.mask_user = mask_user
        self.chat_template = chat_template
        self.pack = pack
        
        self.train_dataset = None
        self.val_dataset = None
        self.collator = SFTDataCollator(tokenizer=tokenizer)
        self.train_sampler: Optional[DistributedSampler] = None
        self.val_sampler: Optional[DistributedSampler] = None

    def prepare(self) -> None:
        self.train_dataset = SFTDataset(
            data_files=self.train_files,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            text_field=self.text_field,
            add_eos=self.add_eos,
            mask_user=self.mask_user,
            chat_template=self.chat_template,
            pack=self.pack,
            seed=self.seed,
        )
        self.val_dataset = SFTDataset(
            data_files=self.val_files,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            text_field=self.text_field,
            add_eos=self.add_eos,
            mask_user=self.mask_user,
            chat_template=self.chat_template,
            pack=self.pack,
            seed=self.seed,
        )

    def train_dataloader(self) -> DataLoader:
        prefetch = 2 if self.num_workers > 0 else None
        loader_kwargs = {"prefetch_factor": prefetch} if prefetch is not None else {}
        if torch.distributed.is_initialized():
            self.train_sampler = DistributedSampler(self.train_dataset, shuffle=False)
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=self.train_sampler,
            num_workers=self.num_workers,
            collate_fn=self.collator,
            pin_memory=True, 
            persistent_workers=True if self.num_workers > 0 else False,
            **loader_kwargs,
        )

    def val_dataloader(self) -> DataLoader:
        prefetch = 2 if self.num_workers > 0 else None
        loader_kwargs = {"prefetch_factor": prefetch} if prefetch is not None else {}
        if torch.distributed.is_initialized():
            self.val_sampler = DistributedSampler(self.val_dataset, shuffle=False)
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=self.val_sampler,
            num_workers=self.num_workers,
            collate_fn=self.collator,
            pin_memory=True, 
            persistent_workers=True if self.num_workers > 0 else False, 
            **loader_kwargs,
        )

    def set_epoch(self, epoch: int) -> None:
        if self.train_sampler is not None and hasattr(self.train_sampler, "set_epoch"):
            self.train_sampler.set_epoch(epoch)
