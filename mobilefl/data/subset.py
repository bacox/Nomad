# -*- coding: utf-8 -*-
from typing import Callable, Optional, Sequence

import torch
from torch.utils.data import DataLoader, Dataset, Subset


class CustomSubset(Subset):
    """A custom subset class with customizable data transformation."""

    def __init__(
        self,
        dataset: Dataset,
        indices: Sequence,
        targets: Sequence,
        classes: Sequence,
        subset_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(dataset, indices)
        self.subset_transform = subset_transform
        self.targets = targets
        self.classes = classes

    def __getitem__(self, idx) -> tuple:  # type: ignore
        x, y = self.dataset[self.indices[idx]]

        if self.subset_transform:
            x = self.subset_transform(x)

        return x, y

    def __len__(self) -> int:
        return len(self.indices)


def wikitext2batchify(dataset: DataLoader, batch_size: int, seq_len: int) -> Sequence:
    data = []
    for tokens in dataset:
        if len(tokens) != 0:
            data.extend(tokens)
    data = torch.LongTensor(data)  # type: ignore
    num_batches = data.shape[0] // batch_size  # type: ignore
    data = data[: num_batches * batch_size]
    data = data.view(batch_size, num_batches)  # type: ignore
    batches = []
    for idx in range(0, num_batches - seq_len - 1, seq_len):
        inputs = data[:, idx : idx + seq_len]
        target = data[:, idx + 1 : idx + seq_len + 1]
        if inputs.shape[1] == seq_len and target.shape[1] == seq_len:
            batches.append([inputs, target])
    return batches
