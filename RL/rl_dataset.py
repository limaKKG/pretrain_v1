import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

class RLDataset(Dataset):
    def __init__(
        self,
        data_files: List[str],
        tokenizer: Any,
        max_length: int = 2048,
        max_prompt_length: int = 1024,
        add_eos: bool = True,
        seed: int = 42,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
        self.add_eos = add_eos
        self.seed = seed
        
        raw_records = []
        for file_path in data_files:
            if not os.path.exists(file_path):
                continue
            with open(file_path, "r") as f:
                for line in f:
                    raw_records.append(json.loads(line))
        
        if (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0:
            print(f"Loaded {len(raw_records)} preference pairs.")
        
        random.Random(self.seed).shuffle(raw_records)
        self.records = raw_records

    def __len__(self) -> int:
        return len(self.records)

    def _tokenize(self, prompt: str, response: str) -> Dict[str, torch.Tensor]:
        if self.add_eos and not response.endswith(self.tokenizer.eos_token):
            response += self.tokenizer.eos_token
            
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        response_ids = self.tokenizer.encode(response, add_special_tokens=False)
        
        # Truncate prompt if needed
        if len(prompt_ids) > self.max_prompt_length:
            prompt_ids = prompt_ids[-self.max_prompt_length:]
            
        # Truncate response if total exceeds max_length
        if len(prompt_ids) + len(response_ids) > self.max_length:
            response_ids = response_ids[:self.max_length - len(prompt_ids)]
            
        input_ids = prompt_ids + response_ids
        
        # Loss mask: 0 for prompt, 1 for response
        loss_mask = [0] * len(prompt_ids) + [1] * len(response_ids)
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.long),
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        record = self.records[idx]
        prompt = record["prompt"]
        chosen = record["chosen"]
        rejected = record["rejected"]
        
        chosen_data = self._tokenize(prompt, chosen)
        rejected_data = self._tokenize(prompt, rejected)
        
        return {
            "chosen_input_ids": chosen_data["input_ids"],
            "chosen_loss_mask": chosen_data["loss_mask"],
            "rejected_input_ids": rejected_data["input_ids"],
            "rejected_loss_mask": rejected_data["loss_mask"],
        }


class RLDataCollator:
    def __init__(self, tokenizer: Any, pad_to_multiple_of: Optional[int] = 8) -> None:
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # We need to pad chosen and rejected separately, or pad all to the same max length.
        # To keep it simple and efficient for SimPO, let's pad everything in the batch 
        # to the same max length (max of all chosen and rejected).
        
        all_ids = []
        for item in batch:
            all_ids.append(item["chosen_input_ids"])
            all_ids.append(item["rejected_input_ids"])
            
        max_batch_len = max(len(x) for x in all_ids)
        if self.pad_to_multiple_of:
            max_batch_len = ((max_batch_len + self.pad_to_multiple_of - 1) // 
                             self.pad_to_multiple_of * self.pad_to_multiple_of)
                             
        def pad_tensor(t, pad_val, max_len):
            pad_len = max_len - len(t)
            return torch.cat([t, torch.full((pad_len,), pad_val, dtype=t.dtype)])

        chosen_ids = []
        chosen_loss_mask = []
        chosen_attn_mask = []
        rejected_ids = []
        rejected_loss_mask = []
        rejected_attn_mask = []
        
        for item in batch:
            c_ids = item["chosen_input_ids"]
            c_l_mask = item["chosen_loss_mask"]
            r_ids = item["rejected_input_ids"]
            r_l_mask = item["rejected_loss_mask"]
            
            chosen_ids.append(pad_tensor(c_ids, self.tokenizer.pad_token_id, max_batch_len))
            chosen_loss_mask.append(pad_tensor(c_l_mask, 0, max_batch_len))
            chosen_attn_mask.append(pad_tensor(torch.ones(len(c_ids), dtype=torch.long), 0, max_batch_len))
            
            rejected_ids.append(pad_tensor(r_ids, self.tokenizer.pad_token_id, max_batch_len))
            rejected_loss_mask.append(pad_tensor(r_l_mask, 0, max_batch_len))
            rejected_attn_mask.append(pad_tensor(torch.ones(len(r_ids), dtype=torch.long), 0, max_batch_len))
            
        return {
            "chosen_input_ids": torch.stack(chosen_ids),
            "chosen_attention_mask": torch.stack(chosen_attn_mask),
            "chosen_loss_mask": torch.stack(chosen_loss_mask),
            "rejected_input_ids": torch.stack(rejected_ids),
            "rejected_attention_mask": torch.stack(rejected_attn_mask),
            "rejected_loss_mask": torch.stack(rejected_loss_mask),
        }


class RLDataModule:
    def __init__(
        self,
        train_files: List[str],
        val_files: List[str],
        tokenizer: Any,
        max_length: int,
        max_prompt_length: int,
        batch_size: int,
        num_workers: int,
        seed: int,
        add_eos: bool = True,
    ) -> None:
        self.train_files = train_files
        self.val_files = val_files
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.add_eos = add_eos
        
        self.train_dataset = None
        self.val_dataset = None
        self.collator = RLDataCollator(tokenizer=tokenizer)
        self.train_sampler: Optional[DistributedSampler] = None
        self.val_sampler: Optional[DistributedSampler] = None

    def prepare(self) -> None:
        self.train_dataset = RLDataset(
            data_files=self.train_files,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            max_prompt_length=self.max_prompt_length,
            add_eos=self.add_eos,
            seed=self.seed,
        )
        self.val_dataset = RLDataset(
            data_files=self.val_files,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            max_prompt_length=self.max_prompt_length,
            add_eos=self.add_eos,
            seed=self.seed,
        )

    def train_dataloader(self) -> DataLoader:
        if torch.distributed.is_initialized():
            self.train_sampler = DistributedSampler(self.train_dataset, shuffle=True, seed=self.seed)
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=(self.train_sampler is None),
            sampler=self.train_sampler,
            num_workers=self.num_workers,
            collate_fn=self.collator,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
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
        )
