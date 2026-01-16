from torch.utils.data import ConcatDataset


class CustomConcatSubset(ConcatDataset):
    def __init__(self, dataset, cumulative_sizes, subset_transform=None):
        super().__init__(dataset, cumulative_sizes)
        self.subset_transform = subset_transform
        self.targets = dataset.targets
        self.classes = dataset.classes

    def __getitem__(self, idx):

        x, y = self.dataset[self.indices[idx]]

        if self.subset_transform:
            x = self.subset_transform(x)

        return x, y

    def __len__(self):
        return self.cumulative_sizes[-1]
