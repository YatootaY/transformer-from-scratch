import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class SequenceDataset(Dataset):

    def __init__(self,csv_file):
        self.data = pd.read_csv(csv_file)
        self.tokens = self.data[[f'token_{i:02d}' for i in range(1, 21)]].values
        self.masks = self.data[[f'mask_{i:02d}' for i in range(1, 21)]].values
        self.labels = self.data['label'].values

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        tokens = torch.tensor(self.tokens[idx], dtype=torch.long)
        masks = torch.tensor(self.masks[idx], dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.float)
        return tokens, masks, label

if __name__ == "__main__":
    test_dataset = SequenceDataset('test.csv')
    train_dataset = SequenceDataset('train.csv')
    val_dataset = SequenceDataset('validation.csv')

    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    for tokens, masks, labels in train_dataloader:
        print(tokens.shape, masks.shape, labels.shape)
        break