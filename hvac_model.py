import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

# Written by Anthony Bilic
# anthonyibilic@gmail.com

## DEV NOTE
# Simple 4 layer autoencoder approach.

class Encoder(pl.LightningModule):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        self.layer1 = nn.Linear(input_dim, 512)
        self.layer2 = nn.Linear(512, 256)
        self.layer3 = nn.Linear(256, 128)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class Decoder(pl.LightningModule):
    def __init__(self, input_dim, latent_dim):
        super(Decoder, self).__init__()
        self.layer1 = nn.Linear(latent_dim, 128)
        self.layer2 = nn.Linear(128, 256)
        self.layer3 = nn.Linear(256, 512)
        self.layer4 = nn.Linear(512, input_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = torch.sigmoid(self.layer4(x))
        return x

class AE(pl.LightningModule):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        latent_dim = 64  # Hard coded
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(input_dim, latent_dim)
        self.train_loss = []
        self.val_loss = []
        self.save_hyperparameters()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, features):
        mu, logvar = self.encoder(features)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar

    def loss_function(self, x_hat, x, z_mean, z_log_var):
        reconstruction_loss = F.binary_cross_entropy(x_hat, x, reduction='sum')
        kl_divergence_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
        total_loss = reconstruction_loss + kl_divergence_loss
        return total_loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.flatten(), y.flatten()
        y_hat, mu, logvar = self.forward(x)
        loss = self.loss_function(y_hat, y, mu, logvar)
        self.log('train_loss', loss)
        self.train_loss.append(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.flatten(), y.flatten()
        y_hat, mu, logvar = self.forward(x)
        loss = self.loss_function(y_hat, y, mu, logvar)
        self.log('val_loss', loss)
        self.val_loss.append(loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer