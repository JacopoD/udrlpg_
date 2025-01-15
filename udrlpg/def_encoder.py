# https://github.com/HSG-AIML/NeurIPS_2021-Weight_Space_Learning/blob/main/modules/model_definitions/components/def_encoder.py

import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, dim, nlayers, nlin, dropout):
        super().__init__()

        self.resblockList = nn.ModuleList()

        for ldx in range(nlayers - 1):
            self.resblockList.append(nn.Linear(dim, dim, bias=True))
            # add nonlinearity
            if nlin == "elu":
                self.resblockList.append(nn.ELU())
            if nlin == "celu":
                self.resblockList.append(nn.CELU())
            if nlin == "gelu":
                self.resblockList.append(nn.GELU())
            if nlin == "leakyrelu":
                self.resblockList.append(nn.LeakyReLU())
            if nlin == "relu":
                self.resblockList.append(nn.ReLU())
            if nlin == "tanh":
                self.resblockList.append(nn.Tanh())
            if nlin == "sigmoid":
                self.resblockList.append(nn.Sigmoid())
            if dropout > 0:
                self.resblockList.append(nn.Dropout(dropout))
        # init output layer
        self.resblockList.append(nn.Linear(dim, dim, bias=True))
        # add output nonlinearity
        if nlin == "elu":
            self.nonlin_out = nn.ELU()
        if nlin == "celu":
            self.nonlin_out = nn.CELU()
        if nlin == "gelu":
            self.nonlin_out = nn.GELU()
        if nlin == "leakyrelu":
            self.nonlin_out = nn.LeakyReLU()
        if nlin == "tanh":
            self.nonlin_out = nn.Tanh()
        if nlin == "sigmoid":
            self.nonlin_out = nn.Sigmoid()
        else:  # relu
            self.nonlin_out = nn.ReLU()

    def forward(self, x):
        # clone input
        x_inp = x.clone()
        # forward prop through res block
        for m in self.resblockList:
            x = m(x)
        # add input and new x together
        y = self.nonlin_out(x + x_inp)
        return y


class Encoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_sizes: int,
                 nlin: str, dropout: float, init_type: str):
        super(Encoder, self).__init__()

        res_blocks = 0
        res_block_lays = 0
        self.hidden_sizes = hidden_sizes
        self.i_dim = input_dim
        self.latent_dim = latent_dim
        self.init_type = init_type

        # set flag for residual blocks
        self.res = False
        if res_blocks > 0 and res_block_lays > 0:
            self.res = True

        if self.res:
            # start with encoder resblock
            self.resEncoder = nn.ModuleList()
            for _ in range(res_blocks):
                self.resEncoder.append(
                    ResBlock(
                        dim=self.i_dim, nlayers=res_block_lays, nlin=nlin, dropout=dropout
                    )
                )
        
        # dimensions = np.linspace(self.i_dim, self.latent_dim, self.hidden_layers + 2).astype("int")
        dimensions = [self.i_dim] + list(self.hidden_sizes) + [self.latent_dim]

        self.out_size = dimensions[-2]
        
        # init encoder
        self.encoder = nn.ModuleList()
        # compose layers
        for idx, _ in enumerate(dimensions[:-2]):
            self.encoder.append(nn.Linear(dimensions[idx], dimensions[idx + 1]))
            # add nonlinearity

            if nlin == "elu":
                self.encoder.append(nn.ELU())
            if nlin == "celu":
                self.encoder.append(nn.CELU())
            if nlin == "gelu":
                self.encoder.append(nn.GELU())
            if nlin == "leakyrelu":
                self.encoder.append(nn.LeakyReLU())
            if nlin == "relu":
                self.encoder.append(nn.ReLU())
            if nlin == "tanh":
                self.encoder.append(nn.Tanh())
            if nlin == "sigmoid":
                self.encoder.append(nn.Sigmoid())
            if dropout > 0:
                self.encoder.append(nn.Dropout(dropout))

        # initialize weights with se methods
        print("initialze encoder")
        self.encoder = self.initialize_weights(self.encoder)
        if self.res:
            self.resEncoder = self.initialize_weights(self.resEncoder)

    def initialize_weights(self, module_list):
        for m in module_list:
            if type(m) == nn.Linear:
                if self.init_type == "xavier_uniform":
                    nn.init.xavier_uniform_(m.weight)
                if self.init_type == "xavier_normal":
                    nn.init.xavier_normal_(m.weight)
                if self.init_type == "uniform":
                    nn.init.uniform_(m.weight)
                if self.init_type == "normal":
                    nn.init.normal_(m.weight)
                if self.init_type == "kaiming_normal":
                    nn.init.kaiming_normal_(m.weight)
                if self.init_type == "kaiming_uniform":
                    nn.init.kaiming_uniform_(m.weight)
                # set bias to some small non-zero value
                m.bias.data.fill_(0.01)
        return module_list

    def forward(self, x):
        # forward prop through resEncoder
        if self.res:
            for resblock in self.resEncoder:
                x = resblock(x)
        # forward prop through encoder
        for layer in self.encoder:
            x = layer(x)
            
        return x
