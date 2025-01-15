import torch
import torch.nn as nn
from udrlpg.utils import mlp

def get_hypernetwork_mlp_generator(
    layer_sizes,
    hidden_sizes,
    embedding_dim,
    features_per_embedding=32,
    scale_layer_out=False,
    scale_parameter=1,
    command_len=1
):
    input_hn = HyperNetwork(
        hidden_sizes=hidden_sizes,
        z_dim_w=embedding_dim + command_len,
        z_dim_b=embedding_dim + command_len,
        out_size_w=[
            layer_sizes[1] if len(layer_sizes) == 2 else features_per_embedding,
            layer_sizes[0],
        ],
        out_size_b=layer_sizes[1] if len(layer_sizes) == 2 else features_per_embedding,
    )

    if len(layer_sizes) > 2:
        output_hn = HyperNetwork(
            hidden_sizes=hidden_sizes,
            z_dim_w=embedding_dim + command_len,
            z_dim_b=embedding_dim + command_len,
            out_size_w=[layer_sizes[-1], features_per_embedding],
            out_size_b=layer_sizes[-1],
        )
    else:
        output_hn = None

    if len(layer_sizes) > 3:
        hidden_hn = HyperNetwork(
            hidden_sizes=hidden_sizes,
            z_dim_w=embedding_dim + command_len,
            z_dim_b=embedding_dim + command_len,
            out_size_w=[features_per_embedding, features_per_embedding],
            out_size_b=features_per_embedding,
        )
    else:
        hidden_hn = None

    in_tiling = [
        1,
        1 if len(layer_sizes) == 2 else layer_sizes[1] // features_per_embedding,
    ]
    out_tiling = (
        [layer_sizes[-2] // features_per_embedding, 1]
        if len(layer_sizes) >= 2
        else None
    )
    if len(layer_sizes) > 3:
        hidden_tiling = []
        for i in range(1, len(layer_sizes) - 2):
            ht = [
                layer_sizes[i] // features_per_embedding,
                layer_sizes[i + 1] // features_per_embedding,
            ]
            hidden_tiling.append(ht)
    else:
        hidden_tiling = None

    fc_generator = HyperNetworkGenerator(
        input_fc_hn=input_hn,
        hidden_fc_hn=hidden_hn,
        output_fc_hn=output_hn,
        in_tiling=in_tiling,
        hidden_tiling=hidden_tiling,
        out_tiling=out_tiling,
        embedding_dim=embedding_dim,
        layer_sizes=layer_sizes,
        scale_layer_out=scale_layer_out,
        scale_parameter=scale_parameter,
    )

    return fc_generator



class HyperNetwork(nn.Module):
    def __init__(
        self, hidden_sizes, z_dim_w=65, z_dim_b=4, out_size_w=[8, 8], out_size_b=8
    ):
        super(HyperNetwork, self).__init__()
        self.z_dim_w = z_dim_w
        self.z_dim_b = z_dim_b

        self.out_size_w = out_size_w
        self.out_size_b = out_size_b
        self.total_el_w = self.out_size_w[0] * self.out_size_w[1]

        sizes_w = [self.z_dim_w] + list(hidden_sizes) + [self.total_el_w]
        self.net_w = mlp(sizes_w, activation=nn.ReLU)
        sizes_b = [self.z_dim_b] + list(hidden_sizes) + [self.out_size_b]
        self.net_b = mlp(sizes_b, activation=nn.ReLU)

    def forward(self, z, command):
        # z: batch_size x z_dim
        # command: batch_size x 
        # this seems to take as input only a scalar (one unit tensor) so I am guessing that 
        # print(z.shape, command.shape)
        kernel_w = self.net_w(torch.cat((z, command), dim=1))
        kernel_w = kernel_w.view(-1, self.out_size_w[0], self.out_size_w[1])

        kernel_b = self.net_b(torch.cat((z, command), dim=1))
        kernel_b = kernel_b.view(-1, self.out_size_b)

        return kernel_w, kernel_b


class HyperNetworkGenerator(torch.nn.Module):
    def __init__(
        self,
        input_fc_hn: HyperNetwork,
        hidden_fc_hn: HyperNetwork = None,
        output_fc_hn: HyperNetwork = None,
        in_tiling=[1, 1],
        hidden_tiling=None,
        out_tiling=None,
        embedding_dim: int = 64,
        layer_sizes=None,
        scale_layer_out=False,
        scale_parameter=1,
    ):
        super().__init__()
        # layer generators
        self.input_hn = input_fc_hn
        self.hidden_hn = hidden_fc_hn
        self.output_hn = output_fc_hn
        # tilings
        self.in_tiling = in_tiling
        self.hidden_tiling = hidden_tiling
        self.out_tiling = out_tiling
        if layer_sizes is not None:
            self.layer_sizes = layer_sizes
        else:
            raise ValueError
        self.scale_layer_out = scale_layer_out
        self.scale_parameter = scale_parameter
        self.num_layers = 1
        if self.hidden_tiling is not None:
            self.num_layers += len(self.hidden_tiling)
        if self.out_tiling is not None:
            self.num_layers += 1

        # embeddings
        self.in_embeddings = torch.nn.Parameter(
            torch.randn(self.in_tiling + [embedding_dim])
        )
        self.out_embeddings = torch.nn.Parameter(
            torch.randn(self.out_tiling + [embedding_dim])
        )
        if self.num_layers >= 3:
            self.hidden_embeddings = torch.nn.ParameterList(
                [
                    torch.nn.Parameter(torch.randn(ft + [embedding_dim]))
                    for ft in self.hidden_tiling
                ]
            )
        else:
            self.hidden_embeddings = None

    def forward(
        self, command: torch.FloatTensor, conditioning: torch.FloatTensor = None
    ):
        """
        :param command: batch_size x 1
        :param command: batch_size x conditioning_size
        :return:
        """
        batch_size = command.shape[0]

        if conditioning is not None:
            command = torch.cat([command, conditioning], dim=1)
        # print(command.shape, command)


        generated_parameters = []

        # fully connected
        for i in range(self.num_layers):
            if i == 0:
                hn = self.input_hn
                tiling = self.in_tiling
                embeddings = self.in_embeddings
            elif i == self.num_layers - 1:
                hn = self.output_hn
                tiling = self.out_tiling
                embeddings = self.out_embeddings
            else:
                hn = self.hidden_hn
                tiling = self.hidden_tiling[i - 1]
                embeddings = self.hidden_embeddings[i - 1]
            # repeat embeddings across batch
            embeddings = embeddings[None].repeat(
                batch_size, 1, 1, 1
            )  # batch_size x tiles_in x tiles_out x z_dim
            # repeat command across tiles
            r_command = command[:, None, None, :].repeat(1, tiling[0], tiling[1], 1)
            embeddings = embeddings.view(-1, embeddings.shape[-1])
            r_command = r_command.view(-1, r_command.shape[-1])
            # print(r_command.shape, r_command)
            w, b = hn(embeddings, r_command)
            if self.scale_layer_out:
                w = (
                    w
                    * self.scale_parameter
                    / torch.sqrt(torch.tensor([self.layer_sizes[i]]).float()).to(
                        w.device
                    )
                )
                b = (
                    b
                    * self.scale_parameter
                    / torch.sqrt(torch.tensor([self.layer_sizes[i]]).float()).to(
                        b.device
                    )
                )

            w = w.view(
                batch_size, tiling[0], tiling[1], hn.out_size_w[0], hn.out_size_w[1]
            ).permute(0, 2, 3, 1, 4)
            w = w.reshape(
                batch_size, tiling[1] * hn.out_size_w[0], tiling[0] * hn.out_size_w[1]
            )
            b = b.reshape(batch_size, tiling[0], tiling[1], hn.out_size_w[0]).mean(
                dim=1
            )
            b = b.view(batch_size, tiling[1] * hn.out_size_w[0])
            generated_parameters.extend([w, b])

        flat_parameters = [p.view(p.shape[0], -1) for p in generated_parameters]
        flat_parameters = torch.cat(flat_parameters, dim=1)

        return flat_parameters