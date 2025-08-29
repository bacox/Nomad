from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import WikiText2
from torchtext.vocab import build_vocab_from_iterator
from torchvision import datasets, transforms
from mobilefl.data.dataset_wrappers import DatasetShard, wikitext2batchify
from mobilefl.log_tools.logging_style import line
if TYPE_CHECKING:
    from server import Server
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dirichlet import get_dataset
from data.subset import CustomSubset
class Data:
    def __init__(self, config, verbose=False):
        self.config = config
        self.dataset_name = config.get("dataset")
        self.num_clients = config.get("num_clients")
        self.num_data_per_client = config.get("num_data_per_client")
        self.batch_size = config.get("batch_size")
        if self.dataset_name == "wikitext2":
            self.sequence_len = config.get("sequence_len")
        self.cuda = config.get("cuda")
        self.device = torch.device(f"cuda:{config.get('cuda_to_use')}") if torch.cuda.is_available() else "cpu"
        self.kwargs = {"num_workers": 4, "pin_memory": True, "prefetch_factor": 20} if self.cuda else {}
        self.get_data()
        self.num_label_per_client = config.get("num_label_per_client")
        self.dir_beta = config.get("dir_beta")
        self.iid = config.get("iid")
        self.server_iid = config.get("server_iid")
        self.client_iid = config.get("client_iid")
        self.num_servers = config.get("num_servers")
        self.alpha = config.get("alpha")
        self.verbose = verbose
    def get_data(self):
        """
        download the dataset
        """
        self.vocab = []
        if self.dataset_name == "mnist":
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
            self.train = datasets.MNIST("./train", train=True, download=True, transform=transform)
            self.test = datasets.MNIST("./test", train=False, download=True, transform=transform)
        elif self.dataset_name == "fmnist":
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
            self.train = datasets.FashionMNIST("./train", train=True, download=True, transform=transform)
            self.test = datasets.FashionMNIST("./test", train=False, download=True, transform=transform)
        elif self.dataset_name == "cifar":
            transform_train = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]
            )
            transform_test = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]
            )
            self.train = datasets.CIFAR10("./train", train=True, download=True, transform=transform_train)
            self.test = datasets.CIFAR10("./test", train=False, download=True, transform=transform_test)
        elif self.dataset_name == "cifar100":
            normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
            transform_train = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
            transform_test = transforms.Compose(
                [
                    transforms.ToTensor(),
                    normalize,
                ]
            )
            self.train = datasets.CIFAR100("./train", train=True, download=True, transform=transform_train)
            self.test = datasets.CIFAR100("./test", train=False, download=True, transform=transform_test)
        elif self.dataset_name == "wikitext2":
            train_dataset, valid_dataset, test_dataset = WikiText2()
            tokenizer = get_tokenizer("basic_english")
            train_dataset = [tokenizer(s) for s in train_dataset]
            test_dataset = [tokenizer(s) for s in test_dataset]
            topics = [
                "Sports",
                "Technology",
                "Politics",
                "Finance",
                "Economy",
                "History",
                "Geography",
                "Mathematics",
                "Literature",
                "Entertainment",
            ]
            self.train_labels = []
            for example in train_dataset:
                for i, topic in enumerate(topics):
                    if topic.lower() in example:
                        self.train_labels.append(
                            i + 1
                        )  
                        break  
            self.vocab = build_vocab_from_iterator(train_dataset, specials=["<unk>", "<eos>", "<pad>"], min_freq=4)
            self.vocab.set_default_index(0)
            print(line(f"VOCAB LENGTH : {len(self.vocab)}"))
            def text_transform(text):
                if len(text) != 0:
                    mapped = [self.vocab[token] for token in text] + [self.vocab["<eos>"]]
                    return torch.tensor(mapped, dtype=torch.long)
                return torch.tensor([])
            self.train = [text_transform(text) for text in train_dataset]
            self.test = [text_transform(text) for text in test_dataset]
        else:
            print("Not a valid Dataset!!")
            exit(0)
    def get_server_level_datasets(
        self, out_path: Path = Path(__file__).parent.parent
    ) -> Tuple[CustomSubset, CustomSubset]:
        if self.server_iid:
            print("Loading server iid style")
            train_dataset_server_level, test_dataset_server_level = (
                self.train,
                self.test,
            )
        else:
            print("Loading server non iid style")
            train_dataset_server_level, test_dataset_server_level, data_info = get_dataset(
                self.train, self.test, self.alpha, self.num_servers, self.config, out_path=out_path
            )
        return train_dataset_server_level, test_dataset_server_level
    def get_client_data_loaders(
        self,
        server: Server,
        clients_num,
        train_dataset_server_level,
        test_dataset_server_level,
        out_path: Path = Path(__file__).parent.parent,
    ):
        _labels = self.train_labels if self.dataset_name == "wikitext2" else None
        test_data_loaders = []
        train_data_loaders = []
        if self.server_iid:
            if self.client_iid:
                server.client_train_datasets = self.iid_equal_size_split(train_dataset_server_level, clients_num)
                server.client_test_datasets = self.iid_equal_size_split(
                    test_dataset_server_level, clients_num, train=False
                )
            else:
                server.client_train_datasets, server.client_test_datasets, data_info = get_dataset(
                    train_dataset_server_level,
                    test_dataset_server_level,
                    self.alpha,
                    server.clients_num,
                    self.config,
                    server_id=server.server_id,
                    out_path=out_path,
                )
        else:
            if self.client_iid:
                server.client_train_datasets = self.iid_equal_size_split(
                    train_dataset_server_level[server.server_id], server.clients_num
                )
                server.client_test_datasets = self.iid_equal_size_split(
                    test_dataset_server_level[server.server_id],
                    server.clients_num,
                    train=False,
                )
            else:
                (
                    server.client_train_datasets,
                    server.client_test_datasets,
                    data_info_client,
                ) = get_dataset(
                    train_dataset_server_level[server.server_id],
                    test_dataset_server_level[server.server_id],
                    self.alpha,
                    server.clients_num,
                    self.config,
                    server_id=server.server_id,
                    out_path=out_path,
                )
        if self.dataset_name == "wikitext2":
            for i in range(server.clients_num):
                train_data_loaders.append(
                    wikitext2batchify(
                        server.client_train_datasets[i],
                        self.batch_size,
                        self.sequence_len,
                    )
                )
                test_data_loaders.append(
                    wikitext2batchify(
                        server.client_test_datasets[i],
                        self.batch_size,
                        self.sequence_len,
                    )
                )
        else:
            for i in range(server.clients_num):
                train_data_loaders.append(
                    DataLoader(
                        server.client_train_datasets[i],
                        batch_size=self.batch_size,
                        shuffle=True,
                        **self.kwargs,  
                    )
                )
                test_data_loaders.append(
                    DataLoader(
                        server.client_test_datasets[i],
                        batch_size=self.batch_size,
                        shuffle=True,
                        **self.kwargs,  
                    )
                )
        return train_data_loaders, test_data_loaders
    def get_server_data_loaders(self) -> DataLoader:
        """
        currently the servers all have an identical test set
        """
        if self.dataset_name == "wikitext2":
            print(f"Loading server wikitext2 style with {self.sequence_len=}, {self.batch_size=}")
            self.server_test_loader = wikitext2batchify(self.test, self.batch_size, self.sequence_len)
            print(line("server loader"))
        else:
            self.server_test_loader = DataLoader(
                self.test, batch_size=self.batch_size, shuffle=True, **self.kwargs  
            )
        return self.server_test_loader
    def iid_equal_size_split(self, data, clients_num, train=True):
        """
        split the dataset into shards with equal size
        """
        if train:
            if self.num_data_per_client == 0:
                num_samples = len(data)
            else:
                num_samples = clients_num * self.num_data_per_client
        else:
            num_samples = len(data)
        num_samples_per_client = int(num_samples / clients_num)
        num_samples_per_client = int(num_samples / clients_num)
        datasets = []  
        all_idxs = [i for i in range(num_samples)]
        np.random.shuffle(all_idxs)
        for i in range(clients_num):
            client_idxs = all_idxs[i * num_samples_per_client : (i + 1) * num_samples_per_client]
            if self.dataset_name == "wikitext2":
                datasets.append(DatasetShard(data, client_idxs, None, None))
            else:
                datasets.append(CustomSubset(data, client_idxs, data.targets, data.classes))
        return datasets
    def label_skew_split(self, data, labels=None):
        if labels is None:  
            labels = data.targets
        print(line(f"Number of data points {len(labels)}"))
        sorted_idxs = np.argsort(
            labels
        ).tolist()  
        num_groups = self.num_clients * self.num_label_per_client
        num_samples_per_group = int(len(data) / num_groups)
        group_idx = np.arange(
            num_groups
        )  
        np.random.shuffle(group_idx)
        print(f"shuffled group index: {group_idx}")
        cnt = {}
        no_repeating = False
        while not no_repeating:
            for i in range(len(group_idx)):
                if i % self.num_label_per_client == 0:
                    cnt = {}
                lb = group_idx[i] // self.num_label_per_client
                if lb not in cnt:
                    cnt[lb] = 1
                    if i == len(group_idx) - 1:
                        no_repeating = True
                else:
                    print(f"repeating {group_idx[i]}")
                    np.random.shuffle(group_idx)
                    print(f"shuffled group index: {group_idx}")
        data_loaders = []  
        for client_id in range(self.num_clients):
            has_group_idx = group_idx[
                client_id * self.num_label_per_client : (client_id + 1) * self.num_label_per_client
            ]
            client_idxs = []
            for group in has_group_idx:
                if len(client_idxs) == 0:
                    client_idxs = sorted_idxs[group * num_samples_per_group : (group + 1) * num_samples_per_group]
                else:
                    client_idxs.extend(sorted_idxs[group * num_samples_per_group : (group + 1) * num_samples_per_group])
            if self.dataset_name == "wikitext2":
                assert False, "This branch is not implemented yet"
                print(f"client_idxs:{client_idxs}")
                data_loaders.append(
                    DataLoader(
                        DatasetShard(data, client_idxs),
                        batch_size=self.batch_size,
                        shuffle=True,
                        **self.kwargs,
                    )
                )
            else:
                data_loaders.append(
                    DataLoader(
                        DatasetShard(data, client_idxs),
                        batch_size=self.batch_size,
                        shuffle=True,
                        **self.kwargs,  
                    )
                )
        return data_loaders
    def lable_skew_dir(self, data):
        pass
