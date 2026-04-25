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
    
def get_dataloaders(train_path, val_path, test_path, batch_size=32):

    train_dataset = SequenceDataset(train_path)
    val_dataset = SequenceDataset(val_path)
    test_dataset = SequenceDataset(test_path)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
