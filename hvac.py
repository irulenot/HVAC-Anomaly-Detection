import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import shutil
import os
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from hvac_model import AE

# Written by Anthony Bilic
# anthonyibilic@gmail.com

# DEV NOTE
# This approach is assuming the model is trained on data without anomalies

def load_data():
    df = pd.read_csv('data/HVAC Systems Anomaly Detection using ML/HVAC_NE_EC_19-21.csv')
    df = df.drop('Timestamp', axis=1)
    df.fillna(0, inplace=True)
    for column in df:
        if not is_numeric_dtype(df[column]):
            df[column] = pd.factorize(df[column])[0]
    data_per_day = 24 * 4  # Timestamps given every 15 min, thus 4 data in an hour
    batch_size = data_per_day  # 96
    X = df.values
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    days_of_data = int(len(X)/data_per_day)
    days_of_train_data = int(days_of_data*.8)
    days_of_val_data = int(days_of_data*.2)
    train_idx, val_idx = days_of_train_data*data_per_day, days_of_train_data*data_per_day + days_of_val_data*data_per_day
    X_train, X_val = X_scaled[0:train_idx], X_scaled[train_idx:val_idx]

    class MyDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            x = torch.Tensor(self.data[idx])
            return x, x  # Return the input as well as the target (which is the same as the input for an autoencoder)
    train_dataset = MyDataset(X_train)
    val_dataset = MyDataset(X_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, batch_size, X_train.shape[1]

def train_model(train_loader, val_loader, batch_size, num_features, model_save_path, threshold_path):
    input_dim = batch_size * num_features  # 960
    model = AE(input_dim)

    max_epochs = 200
    default_root_dir = 'lightning_logs/'
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        dirpath=default_root_dir,
        filename='model-{epoch:02d}',
        save_top_k=max_epochs
    )
    trainer = pl.Trainer(max_epochs=max_epochs, callbacks=[checkpoint_callback], default_root_dir='./')
    trainer.fit(model, train_loader, val_loader)

    # Get median losses
    loss_train, loss_val = model.train_loss, model.val_loss
    loss_per_train_epoch, loss_per_val_epoch = int(len(loss_train) / max_epochs), int(len(loss_val) / max_epochs)
    loss_train_avg, loss_val_avg = [], []
    for epoch in range(max_epochs):
        start_train_idx, end_train_idx = loss_per_train_epoch * epoch, loss_per_train_epoch * (epoch + 1)
        start_val_idx, end_val_idx = loss_per_val_epoch * epoch, loss_per_val_epoch * (epoch + 1)
        loss_train_avg.append(torch.median(torch.FloatTensor(loss_train[start_train_idx:end_train_idx])).cpu().item())
        loss_val_avg.append(torch.median(torch.FloatTensor(loss_val[start_val_idx:end_val_idx])).cpu().item())
    best_model_idx = loss_val_avg.index(min(loss_val_avg))

    epochs = range(1, max_epochs + 1)
    plt.plot(epochs, loss_train_avg, 'g', label='Training loss')
    plt.plot(epochs, loss_val_avg, 'b', label='Validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    print('Best model is: ' + str(best_model_idx+1))
    plt.show()

    # Determining threshold
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if best_model_idx < 11:
        best_model_path = "lightning_logs/model-epoch=0" + str(best_model_idx) + ".ckpt"
    else:
        best_model_path = "lightning_logs/model-epoch=" + str(best_model_idx) + ".ckpt"
    best_model = AE.load_from_checkpoint(best_model_path, input_dim=input_dim).to(device)
    recon_errors = []
    with torch.no_grad():
        for batch_features, _ in train_loader:
            batch = batch_features.flatten().to(device)
            x_hat, mu, logvar = best_model(batch)
            train_loss = best_model.loss_function(x_hat, batch, mu, logvar)
            recon_errors.append(train_loss.cpu())
        for batch_features, _ in val_loader:
            batch = batch_features.flatten().to(device)
            x_hat, mu, logvar = best_model(batch)
            val_loss = best_model.loss_function(x_hat, batch, mu, logvar)
            recon_errors.append(val_loss.cpu())
    recon_mean = np.nanmean(recon_errors)
    recon_stdev = np.nanstd(recon_errors)
    with open(threshold_path, "w+") as file:
        file.write(str(recon_mean + (2 * recon_stdev)))     # Threshold is 3 std away from mean loss. You can tighten the threshold here.

    shutil.copy(best_model_path, model_save_path)

def test_model(train_loader, val_loader, batch_size, num_features, model_save_path, threshold_path):
    input_dim = batch_size * num_features  # 960
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AE.load_from_checkpoint(model_save_path, input_dim=input_dim).to(device)

    with torch.no_grad():
        for batch_features, _ in train_loader:
            train_batch = batch_features.flatten().to(device)
            x_hat, mu, logvar = model(train_batch)
            train_error = model.loss_function(x_hat, train_batch, mu, logvar)
            break
        for batch_features, _ in val_loader:
            val_batch = batch_features.flatten().to(device)
            x_hat, mu, logvar = model(val_batch)
            val_error = model.loss_function(x_hat, val_batch, mu, logvar)
            break

    with open(threshold_path, 'r') as f:
      threshold = float(f.read())
    print('Error Threshold:', threshold)
    print('Training example error:', train_error.item())
    print('Validation example error:', val_error.item())

    print('Adding anomaly to validation example.')
    val_batch[0:450] = 1
    with torch.no_grad():
        x_hat, mu, logvar = model(val_batch)
        new_val_error = model.loss_function(x_hat, val_batch, mu, logvar)
    print('Validation example error (with anomaly):', new_val_error.item())



if __name__ == "__main__":
    train_loader, val_loader, batch_size, num_features = load_data()
    model_save_path = 'AE_HVAC_model.ckpt'
    threshold_path = 'AE_HVAC_threshold.txt'
    if not os.path.isfile(model_save_path):  # Train model if weight not present
        train_model(train_loader, val_loader, batch_size, num_features, model_save_path, threshold_path)
    test_model(train_loader, val_loader, batch_size, num_features, model_save_path, threshold_path)