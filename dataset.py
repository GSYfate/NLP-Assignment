import torch
from torch.utils.data import Dataset

class ReverseDataset(Dataset):
    def __init__(self, tokenizer, dataset, max_len):
        self.tokenizer = tokenizer
        self.text = dataset['text']
        self.prefix = 'Translate English to reverse-English: '
        self.max_len = max_len
        self.max_texts =100
        
    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = self.text[index][:self.max_texts]
        reverse_text = text[::-1]
        text = ' '.join(text.split())
        reverse_text = ' '.join(reverse_text.split())
        source_text = self.tokenizer.encode_plus(self.prefix + text, padding='max_length', max_length=self.max_len, return_tensors="pt")
        target_text = self.tokenizer.encode_plus(reverse_text, padding='max_length', max_length=self.max_len, return_tensors="pt")
        input_ids = source_text['input_ids'].squeeze()
        attention_mask = source_text['attention_mask'].squeeze()
        labels = target_text['input_ids'].squeeze()
        
        return {
            'input_ids': input_ids.to(dtype=torch.long).to(dtype=torch.long),
            'attention_mask': attention_mask.to(dtype=torch.long),
            'labels': labels.to(dtype=torch.long),
        }