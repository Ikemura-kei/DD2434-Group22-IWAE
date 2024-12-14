import torch
import torch.nn as nn
from easydict import EasyDict

class EncoderNetwork(nn.Module):
    def __init__(self, input_dimensions, hidden_layers, latent_dimensions, activation="tanh", k=1):
        super(EncoderNetwork, self).__init__()
        
        self.k = k

        # Activation function map
        self.activation_map = {
            "TANH": nn.Tanh(),
            "RELU": nn.ReLU(),
            "SIGMOID": nn.Sigmoid(),
            "LEAKYRELU": nn.LeakyReLU()
        }

        # Select activation function
        self.activation = self.activation_map.get(activation.upper(), nn.Tanh())

        # Build the hidden layers
        self.network = self.build_network(input_dimensions, hidden_layers)

        # Layers for latent mean and log-variance
        self.fullyconnectedLayer_mean = nn.Linear(hidden_layers[-1], latent_dimensions)
        self.fullyconnectedLayer_logvar = nn.Linear(hidden_layers[-1], latent_dimensions)

    def build_network(self, input_dimensions, hidden_layers):

        layers = []
        current_size = input_dimensions

        for size in hidden_layers:
            layers.append(nn.Linear(current_size, size))
            layers.append(self.activation)
            current_size = size

        return nn.Sequential(*layers)

    def reparameterization(self, mean, logvar):
        """
        Reparameterization trick: z = μ + σ * ε
        - Enables backpropagation through the latent sampling process.
        """
        std = torch.exp(0.5 * logvar)  # Standard deviation
        eps = torch.randn_like(mean.repeat(1, self.k, 1), device=logvar.device) # Random noise ~ N(0, 1)
        return mean + eps * std

    def forward(self, input_vectors):

        hidden_output = self.network(input_vectors)
        mean = self.fullyconnectedLayer_mean(hidden_output)
        logvar = self.fullyconnectedLayer_logvar(hidden_output)
        z = self.reparameterization(mean, logvar)
        return mean, logvar, z



class DecoderNetwork(nn.Module):
    def __init__(self, latent_dimensions, hidden_layers, output_dimension, activation="tanh", output_activation="sigmoid"):
        super(DecoderNetwork, self).__init__()

        # Activation function map
        self.activation_map = {
            "TANH": nn.Tanh(),
            "RELU": nn.ReLU(),
            "SIGMOID": nn.Sigmoid(),
            "LEAKYRELU": nn.LeakyReLU()
        }

        # Select activation functions
        self.activation = self.activation_map.get(activation.upper(), nn.Tanh())
        self.output_activation = self.activation_map.get(output_activation.upper(), nn.Sigmoid())

        # Build the hidden layers
        self.network = self.build_network(latent_dimensions, hidden_layers)

        # Output layer
        self.output_layer = nn.Linear(hidden_layers[-1], output_dimension)

    def build_network(self, latent_dimension, hidden_layers):

        layers = []
        current_size = latent_dimension

        for size in hidden_layers:
            layers.append(nn.Linear(current_size, size))
            layers.append(self.activation)
            current_size = size

        return nn.Sequential(*layers)

    def forward(self, latent_vectors):

        transformed_latent_input = self.network(latent_vectors)
        logits = self.output_layer(transformed_latent_input)
        output = self.output_activation(logits)
        return output


class VAE(nn.Module):
    def __init__(self, args: EasyDict = None):
        super().__init__()
        # self.encoder = EncoderNetwork(28*28, [512, 256], 256, activation="tanh")
        # self.decoder = DecoderNetwork(256, [256, 512], 28*28, activation="tanh", output_activation="sigmoid")

        self.encoder = EncoderNetwork(**args['EncoderNetwork'])
        self.decoder = DecoderNetwork(**args['DecoderNetwork'])
        
    def forward(self, x):
        mean, logvar, z = self.encoder(x)
        y = self.decoder(z)
        
        return mean, logvar, z, y