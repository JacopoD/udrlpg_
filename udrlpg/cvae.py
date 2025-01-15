# https://github.com/pytorch/examples/blob/main/vae/main.py
from udrlpg.new_encoder import Encoder
from udrlpg.my_hypernetwork import get_hypernetwork_mlp_generator
from udrlpg.utils import softclip, gaussian_nll
import torch.nn as nn
import torch
import torch.nn.functional as F

class ConditionalVAE(nn.Module):
    
    def __init__(self,
                latent_dim: int, kld_scaling: float, learn_sigma: bool,
                input_dim: int, enc_hidden_sizes: int, enc_normalize_out: bool, # encoder parameters
                enc_nlin: str, enc_dropout: float, # encoder parameters
                # enc_init_type: str, 
                policy_neurons: list, dec_hidden_sizes: tuple, embedding_dim: int, # decoder parameters
                scale_layer_out: bool, scale_parameter: int, # decoder parameters
                device: torch.device # other
                ):
        super(ConditionalVAE, self).__init__()

        self.device = device

        self.latent_dim = latent_dim
        self.kld_scaling = kld_scaling
        self.learn_sigma = learn_sigma

        self.encoder = Encoder(input_dim=input_dim,
                                hidden_sizes=list(enc_hidden_sizes),
                                latent_dim=self.latent_dim,
                                dropout=enc_dropout,
                                activation=enc_nlin,
                                normalize_out=enc_normalize_out
                                )

        self.decoder = get_hypernetwork_mlp_generator(layer_sizes=policy_neurons,
                                                        hidden_sizes=dec_hidden_sizes, 
                                                        embedding_dim=embedding_dim, 
                                                        scale_layer_out=scale_layer_out,
                                                        scale_parameter=scale_parameter,
                                                        command_len = 1 + self.latent_dim)
        
        # -1.6 0 
        if self.learn_sigma:
            self.log_sigma = torch.nn.Parameter(torch.full((1,), 0, dtype=torch.float32)[0], requires_grad=True)
        
    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)

        eps = torch.randn_like(std)

        return mean + eps * std

    def forward(self, x, command):

        mean, log_var = self.encoder.forward(x)

        # print(mean, log_var)

        z = self.reparameterize(mean, log_var)

        decoded = self.decoder(command, z)

        return None, decoded, mean, log_var
    
    def reconstruction_loss(self, x, x_hat):
        if self.learn_sigma:
            """ Computes the likelihood of the data given the latent variable,
            in this case using a Gaussian distribution with mean predicted by the neural network and variance = 1 """
        
            # Learning the variance can become unstable in some cases. Softly limiting log_sigma to a minimum of -6
            # ensures stable training.
            log_sigma = softclip(self.log_sigma, -6)
            
            rec = gaussian_nll(x_hat, log_sigma, x).sum()

            return rec
        else:
            return F.mse_loss(x_hat, x, reduction='sum')
    
    def loss(self, x, decoded, mean, log_var, return_separately=False):
        rec_loss = self.reconstruction_loss(x, decoded)

        kld = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

        # self._check_for_inf_nan(rec_loss)
        # self._check_for_inf_nan(kld)

        if return_separately:
            return rec_loss, kld * self.kld_scaling
        return rec_loss + kld * self.kld_scaling


    def sample(self, command: torch.Tensor):
        with torch.no_grad():
            # random noise
            z = torch.randn(command.shape[0], self.latent_dim).to(self.device)
            samples = self.decoder(command, z)

        return samples
    
    # debug function
    def _check_for_inf_nan(self, tensor:torch.Tensor):
        if torch.isnan(tensor).any():
            raise ZeroDivisionError

        if (torch.isinf(tensor).logical_or(torch.isneginf(tensor))).any():
            raise ZeroDivisionError