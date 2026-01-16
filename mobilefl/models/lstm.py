import torch
from torch import nn


class NextCharacterLSTM(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int, hidden_size: int, n_layers: int) -> None:
        super(NextCharacterLSTM, self).__init__()
        self.num_layers = n_layers
        self.hidden_dim = hidden_size
        self.embedding_dim = embed_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: torch.Tensor, hidden: nn.Module) -> tuple:
        embedded = self.embedding(x)
        output, hidden = self.lstm(embedded, hidden)
        output = self.fc(output)
        return output, hidden

    def init_hidden(self, batch_size: int, device: torch.device) -> tuple:
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        cell = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return hidden, cell

    def detach_hidden(self, hidden: tuple) -> tuple:
        hidden, cell = hidden
        hidden = hidden.detach()  # type: ignore
        cell = cell.detach()
        return hidden, cell
