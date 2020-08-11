import torch

class HAIModel(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        n_tags=params['n_tags']
        n_hidden=params['n_hid']
        n_layers=params['n_layers']
        
        self.rnn = torch.nn.GRU(
            input_size=n_tags,
            hidden_size=n_hidden,
            num_layers=n_layers,
            bidirectional=True,
            dropout=0,
        )
        self.fc = torch.nn.Linear(n_hidden * 2, n_tags)

    def forward(self, x):
        x = x.transpose(0, 1)  # (batch, seq, params) -> (seq, batch, params)
        self.rnn.flatten_parameters()
        outs, _ = self.rnn(x)
        out = self.fc(outs[-1])
        return x[0] + out  