from models.model import Model
from models.transformer_backbone import Transformer_Base
from preprocessor import sequence

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import optim
from torch.optim import lr_scheduler 
import torch.nn as nn

class My_Dataset(Dataset):
    def __init__(self, data, pred_len, seq_len, label_len):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.label_len = label_len

        self.marks = torch.ones(len(data))
    
    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data[s_begin:s_end]
        seq_y = self.data[r_begin:r_end]
        seq_x_mark = self.marks[s_begin:s_end]
        seq_y_mark = self.marks[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark.float(), seq_y_mark.float()

class Transformer(Model):
    def __init__(self):
        self.name = "Transformer"

    def fit(self, train, val=None, neurons=10, epochs=200, lookback=30):
        self.train = train
        self.lookback = lookback
        if val is not None:
            self.val = val

        self.config = {
            "pred_len":1,
            "output_attention":False,
            "embed_type":0, # 0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding
            "enc_in":1,
            "dec_in":1,
            "d_model":512,
            "embed":"timeF", # time features encoding, options:[timeF, fixed, learned]
            "freq":"b", # business days
            "dropout":0.05,
            "factor":1,
            "n_heads":8,
            "d_ff":2048,
            "activation":"gelu",
            "e_layers":2, # encoder layers
            "d_layers":1,
            "c_out":1,
            "label_len":1, # 1?
            "features": "S", # M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate
            "seq_len":lookback
        }

        train_dataset = My_Dataset(train, self.config["pred_len"], lookback, self.config["seq_len"])
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        if val is not None:
            extended_val = np.append(self.train[-self.lookback:], val)
            val_dataset = My_Dataset(extended_val, self.config["pred_len"], lookback, self.config["seq_len"])
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

        self.model = Transformer_Base(self.config).float()
        
        train_steps = len(train_loader)
        self.criterion = nn.MSELoss()
        model_optim = optim.Adam(self.model.parameters(), lr=0.0001)
        scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                            steps_per_epoch = train_steps,
                                            pct_start = 0.3,
                                            epochs = epochs,
                                            max_lr = 0.0001)

        for epoch in range(epochs):
            train_loss = []
            self.model.train()
            for i, (batch_x, batch_y, x_pos, y_pos) in enumerate(train_loader):
                batch_x = batch_x.float()
                batch_y = batch_y.float()

                model_optim.zero_grad()
                dec_inp = torch.zeros_like(batch_y[:, -self.config["pred_len"]:]).float()
                dec_inp = torch.cat([batch_y[:, :self.config["pred_len"]], dec_inp], dim=1).float()

                outputs = self.model(batch_x, x_pos, dec_inp, y_pos, batch_y)
                outputs = outputs[:, -self.config["pred_len"]:, 0:]
                batch_y = batch_y[:, -self.config["pred_len"]:, 0:]
                loss = self.criterion(outputs, batch_y)
                train_loss.append(loss.item())
                loss.backward()
                model_optim.step()
            train_loss = np.average(train_loss)
            vali_loss = self.vali(val_loader)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            
        return self.model
    
    def vali(self, val_loader):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, x_pos, y_pos) in enumerate(val_loader):
                batch_x = batch_x.float().unsqueeze(2)
                batch_y = batch_y.float().unsqueeze(2)

                dec_inp = torch.zeros_like(batch_y[:, -self.config["pred_len"]:])
                dec_inp = torch.cat([batch_y[:, :self.config["pred_len"]], dec_inp], dim=1)
                
                outputs = self.model(batch_x, x_pos, dec_inp, y_pos)
                outputs = outputs[:, -self.config["pred_len"]:, 0:]
                batch_y = batch_y[:, -self.config["pred_len"]:, 0:]

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = self.criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def predict(self, x):
        if self.val is not None:
            extended_data = np.append(self.val[-self.lookback:], x)
        else:
            extended_data = np.append(self.train[-self.lookback:], x)

        test_dataset = My_Dataset(extended_data, self.config["pred_len"], self.lookback, self.config["seq_len"])
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

        outputs = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().unsqueeze(2)
                batch_y = batch_y.float().unsqueeze(2)

                dec_inp = torch.zeros_like(batch_y[:, -self.config["pred_len"]:])
                dec_inp = torch.cat([batch_y[:, :self.config["pred_len"]], dec_inp], dim=1)

                outputs.append(self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark))

        return torch.cat(outputs)