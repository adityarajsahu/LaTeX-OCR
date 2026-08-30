from typing import Dict
import torch
from torch.utils.data import Dataset
from datasets import load_dataset, DatasetDict

def load_latex_ocr_splits(dataset_name: str, val_fraction: float = 0.02, test_fraction: float = 0.02, seed: int = 42) -> DatasetDict:
    raw = load_dataset(dataset_name, split = "train")
    split1 = raw.train_test_split(test_size = val_fraction + test_fraction, seed = seed)
    train_ds = split1["train"]
    remainder = split1["test"]
    rel_test_fraction = test_fraction / (val_fraction + test_fraction)
    split2 = remainder.train_test_split(test_size = rel_test_fraction, seed = seed)

    return DatasetDict(train = train_ds, validation = split2["train"], test = split2["test"])

class LaTeXOCRDataset(Dataset):
    def __init__(self, hf_split, image_processor, tokenizer, max_length: int = 256):
        self.data = hf_split
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.data[idx]
        image = example["image"].convert("RGB")
        text = example["text"]

        pixel_values = self.image_processor(images = image, return_tensors = "pt")["pixel_values"][0]

        tokenized = self.tokenizer(text, truncation = True, max_length = self.max_length - 1, padding = False, add_special_tokens = False)
        input_ids = [self.tokenizer.bos_token_id] + tokenized["input_ids"] + [self.tokenizer.eos_token_id]
        input_ids = input_ids[: self.max_length]

        pad_len = self.max_length - len(input_ids)
        attention_mask = [1] * len(input_ids) + [0] * pad_len
        input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
        input_ids = torch.tensor(input_ids, dtype = torch.long)
        attention_mask = torch.tensor(attention_mask, dtype = torch.long)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "text": text
        }

def collate_fn(batch):
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
        "text": [b["text"] for b in batch]
    }