import torch
from torch.utils.data import Dataset, DataLoader

from src.config import *


class TIMITP(Dataset):
    """
    Preprocessed TIMIT Dataset
    """

    def __init__(self, root, n_utts):
        self.X = torch.load(root + "/X.pt")
        self.y = torch.load(root + "/y.pt")
        self.N = n_utts
        assert len(self.X) == len(self.y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        X = self.X[item]
        y = self.y[item]
        # Resample X
        perm = torch.randint(0, len(X), (self.N,))
        X = X[perm]
        return X, y


if __name__ == '__main__':
    train_dataset = TIMITP("data/train", TRAIN_UTTS)
    test_dataset = TIMITP("data/test", TEST_UTTS)

    train_loader = DataLoader(train_dataset, batch_size=4)
    test_loader = DataLoader(test_dataset, batch_size=4)

    for x in train_loader:
        pass

    for x in test_loader:
        pass
