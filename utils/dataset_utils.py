import torch
import numpy as np
from torch.utils.data import Dataset

class imdbDataSet(Dataset):
    def __init__(self, df, tokenizer=None):
        self.dataframe = df
        self.reviews = df['review'].tolist()
        self.encodings = tokenizer(
            self.reviews,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        self.labels = torch.tensor(
            np.where(df['sentiment'] == 'positive', 1, 0),
            dtype=torch.long  
        )

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        return {
            'input_ids': self.encodings['input_ids'][index],
            'attention_mask': self.encodings['attention_mask'][index],
            'sentiment': self.labels[index]
        }
