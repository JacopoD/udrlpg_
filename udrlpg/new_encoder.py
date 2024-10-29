import torch.nn as nn

def mlp(sizes, act, dropout):
    layers = []
    for j in range(len(sizes) - 1):
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
        if dropout > 0.0:
            layers += [nn.Dropout(p=dropout)]

    return nn.Sequential(*layers)

def str_act_to_fn(nlin):
    if nlin == "elu":
        return nn.ELU
    if nlin == "celu":
        return nn.CELU
    if nlin == "gelu":
        return nn.GELU
    if nlin == "leakyrelu":
        return nn.LeakyReLU
    if nlin == "relu":
        return nn.ReLU
    if nlin == "tanh":
        return nn.Tanh
    if nlin == "sigmoid":
        return nn.Sigmoid
    raise NotImplementedError

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_sizes, activation, latent_dim, dropout, normalize_out) -> None:
        super(Encoder, self).__init__()

        self.enc = mlp([input_dim] + list(hidden_sizes), str_act_to_fn(activation), dropout=dropout)
        self.mean = [nn.Linear(hidden_sizes[-1], latent_dim)]
        self.log_var = [nn.Linear(hidden_sizes[-1], latent_dim)]

        if normalize_out:
            self.mean.append(nn.Tanh())
            self.log_var.append(nn.Tanh())

        self.mean = nn.Sequential(*self.mean)
        self.log_var = nn.Sequential(*self.log_var)

    def forward(self, x):
        x = self.enc(x)

        return self.mean(x), self.log_var(x)
    

