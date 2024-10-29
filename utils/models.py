from torch import nn
import torch
import torch.nn.functional as F
import torch.nn as nn


# Simple feedforward binary classification model
class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_layer_size=64):
        super(Classifier, self).__init__()
        if hidden_layer_size == 1:
            self.fc = nn.Sequential(
                nn.Linear(input_dim, hidden_layer_size),
                nn.Sigmoid(),
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(input_dim, hidden_layer_size),
                nn.ReLU(),
                nn.Linear(hidden_layer_size, hidden_layer_size),
                nn.ReLU(),
                nn.Linear(hidden_layer_size, 1),
                nn.Sigmoid(),
            )

    def forward(self, x):
        return self.fc(x)


def create_mlp(input_dim, output_dim, hidden_layer_size, n_layers):
    layers = []
    layers.append(nn.Linear(input_dim, hidden_layer_size))
    layers.append(nn.ReLU())
    for _ in range(n_layers - 1):
        layers.append(nn.Linear(hidden_layer_size, hidden_layer_size))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(hidden_layer_size, output_dim))  # output layer
    return nn.Sequential(*layers)


class MLPWithFiLM(nn.Module):
    def __init__(
        self, input_dim, hidden_layer_size=64, film_hidden_size=256, n_layers_film=3
    ):
        super(MLPWithFiLM, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # MLP for the model
        if hidden_layer_size == 1:
            self.layer1 = nn.Linear(input_dim, 1)
        else:
            self.layer1 = nn.Linear(input_dim, hidden_layer_size)
            self.layer2 = nn.Linear(hidden_layer_size, hidden_layer_size)
            self.layer3 = nn.Linear(hidden_layer_size, 1)

        # MLPs for σ and µ, using lambda as input
        self.sigma_mlp = create_mlp(
            input_dim=1,
            output_dim=hidden_layer_size,
            hidden_layer_size=film_hidden_size,
            n_layers=n_layers_film,
        )
        self.mu_mlp = create_mlp(
            input_dim=1,
            output_dim=hidden_layer_size,
            hidden_layer_size=film_hidden_size,
            n_layers=n_layers_film,
        )

    def forward(self, x, lambda_reg):
        # Apply MLP to lambda_reg to get sigma and mu
        sigma = self.sigma_mlp(lambda_reg)
        mu = self.mu_mlp(lambda_reg)

        # Pass input through the model
        out = F.relu(self.layer1(x))

        # Apply FiLM transform
        out = out * sigma + mu
        if self.layer1.out_features == 1:
            return torch.sigmoid(out)
        out = F.relu(out)

        out = F.relu(self.layer2(out))

        # Apply FiLM transform
        out = out * sigma + mu
        out = F.relu(out)

        out = torch.sigmoid(self.layer3(out))
        return out


class CelebAClassifier(nn.Module):
    def __init__(self, hidden_layer_size=128):
        super(CelebAClassifier, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(
            64 * 16 * 16, hidden_layer_size
        )  # Assuming images are 64x64, adjust accordingly
        self.fc2 = nn.Linear(hidden_layer_size, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # sigmoid activation for binary classification
        return x


class ConvNetWithFiLM(nn.Module):
    def __init__(
        self,
        input_channels=3,
        hidden_layer_size=64,
        film_hidden_size=256,
        n_layers_film=3,
    ):
        super(ConvNetWithFiLM, self).__init__()

        # Convolutional layers for the model
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(
            32, hidden_layer_size, kernel_size=3, stride=1, padding=1
        )
        self.pool = nn.MaxPool2d(2, 2)  # this will reduce each dimension by half

        # MLP layers
        self.layer1 = nn.Linear(hidden_layer_size * 256, hidden_layer_size)
        self.layer2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.layer3 = nn.Linear(hidden_layer_size, 1)

        # MLPs for σ and µ, using lambda as input
        self.sigma_mlp = create_mlp(
            input_dim=1,
            output_dim=hidden_layer_size,
            hidden_layer_size=film_hidden_size,
            n_layers=n_layers_film,
        )
        self.mu_mlp = create_mlp(
            input_dim=1,
            output_dim=hidden_layer_size,
            hidden_layer_size=film_hidden_size,
            n_layers=n_layers_film,
        )

    def forward(self, x, lambda_reg):
        # Apply conv layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        # Flatten the output
        x = x.view(x.size(0), -1)

        # Apply MLP to lambda_reg to get sigma and mu
        sigma = self.sigma_mlp(lambda_reg)
        mu = self.mu_mlp(lambda_reg)

        # Pass input through the model
        out = F.relu(self.layer1(x))

        # Apply FiLM transform
        out = out * sigma + mu
        out = F.relu(out)

        out = F.relu(self.layer2(out))

        # Apply FiLM transform
        out = out * sigma + mu
        out = F.relu(out)

        out = torch.sigmoid(self.layer3(out))
        return out
