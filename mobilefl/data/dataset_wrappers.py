import torch
from torch.utils.data import DataLoader, Dataset
class DatasetShard(Dataset):
    def __init__(self, dataset, idxs, targets, classes):
        super(DatasetShard, self).__init__()
        self.dataset = dataset
        self.idxs = idxs
    def __len__(self):
        return len(self.idxs)
    def __getitem__(self, item):
        return self.dataset[self.idxs[item]]
def wikitext2batchify(dataset, batch_size, seq_len) -> DataLoader:
    data = []
    for tokens in dataset:
        if len(tokens) != 0:
            data.extend(tokens)
    data = torch.LongTensor(data)
    num_batches = data.shape[0] // batch_size
    data = data[: num_batches * batch_size]
    data = data.view(batch_size, num_batches)
    batches = []
    for idx in range(0, num_batches - seq_len - 1, seq_len):
        inputs = data[:, idx : idx + seq_len]
        target = data[:, idx + 1 : idx + seq_len + 1]
        if inputs.shape[1] == seq_len and target.shape[1] == seq_len:
            batches.append([inputs, target])
    return batches  
