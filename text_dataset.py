from torch.utils.data import Dataset
from transformers import BertTokenizer
import pandas as pd
import torch

class TextDataset(Dataset):
    def __init__(self, dataframe, max_len=64, pretrained_model_id = 'bert-base-uncased'):
        self.data = dataframe
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data.iloc[index]["text"]
        target = self.data.iloc[index]["target"]
        
        tokens = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        
        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "target": torch.tensor(target, dtype=torch.long)
        }
        
        
    