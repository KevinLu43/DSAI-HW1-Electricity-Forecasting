from datetime import timedelta
from dateutil.parser import parse
import numpy as np
import pandas as pd
from random import randint
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import time

import matplotlib.pyplot as plt

import torch
from torch import cuda
from torch import device
from torch import from_numpy
from torch import nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader

scaler = MinMaxScaler()
batch_size = 10

if cuda.is_available():
    device = device("cuda")
else:
    device = device("cpu")


class Model:
    def __init__(self):
        return

    def __regulization__(self, data):
        return self.label_scaler.fit_transform(data)

    def __train__data__reorganize(self, load, workday, weekday, date, shift = False):
        X = []
        Y = []
        tag = []
        if shift:
            start = randint(0, 1)*7
        else:
            start = 0
        for i in range(start, load.size-21):
            X.append([
                 workday[i], load[i],
                 workday[i+7], load[i+7],
                 workday[i+14], load[i+14],
                 workday[i+21]])
            Y.append([load[i+21]])
            tag.append(date[i+21])

        return np.array(X), np.array(Y), tag

    def __test__data__reorganize(self, load, workday,date):
        X = []
        Date = []
        for i in range(load.size-14):
            X.append([
                 workday[i], load[i],
                 workday[i+7], load[i+7],
                 workday[i+14], load[i+14],
                 workday[i+14]])
            d = parse(date[i+14])
            Date.append(str((d+timedelta(days=7)).date()))
        return np.array(X), Date

    def __train_test_split(self, x, y,test_ratio):
        train_X, test_X, train_Y, test_Y = train_test_split(x, y, test_size=test_ratio, random_state=32)
        train_data = TensorDataset(from_numpy(np.array(train_X)), from_numpy(np.array(train_Y)))
        test_data = TensorDataset(from_numpy(np.array(test_X)), from_numpy(np.array(test_Y)))
        self.data_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)
        self.test_loader = DataLoader(test_data, batch_size=1)

    def train(self, df):
        # Split data and regulization
        self.label_scaler = scaler.fit(df[["load"]])
        load = self.__regulization__(df[["load"]]).flatten()
        weekday = df["weekday"].values
        workday = df["workday"].values
        date = df["date"].values
        
        # Reorganize data to 4 weeks a set
        X, Y, tag = self.__train__data__reorganize(load, workday, weekday, date, shift = True)
        # Split data to training set  
        self.__train_test_split(X, Y, 0.2)
        # Train model
        self.__model__ = train(self.data_loader, self.test_loader, 0.005)
        # Evaluate model
        return

    def predict(self, df):
        # Split data and regulization
        load = self.__regulization__(df[["load"]]).flatten()
        workday = df["workday"].values
        weekday = df["weekday"].values
        date = df["date"].values

        # Reorganize data to 4 weeks a set
        X, Date = self.__test__data__reorganize(load, workday, date)

        outputs = []
        targets = []
        for i in range(len(X)):
            self.__model__.eval()
            h = self.__model__.init_hidden(1)
            out, h = self.__model__(from_numpy(X[i]).to(device).float().view(1,1,7), h)
            outputs.append(self.label_scaler.inverse_transform(out.cpu().detach().numpy())[0][0])
        return pd.DataFrame({"date": Date,"operating_reserve(HW)": outputs})

class GRUNET(nn.Module):
    def __init__(self, hidden_dim, n_layers, drop_prob=0.1):
        super(GRUNET, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.gru = nn.GRU(7, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x, h):
        out, h = self.gru(x, h)
        out = self.fc(self.relu(out[:,-1]))
        return out, h
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)
        return hidden

def train(train_loader, test_loader, learn_rate, hidden_dim=16, EPOCHS=20):
    n_layers = 2
    # Instantiating the models
    model = GRUNET(hidden_dim, n_layers)
    model.to(device)
    
    # Defining loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=learn_rate)

    TEST_LOSS = []
    TRAIN_LOSS = []
    
    # Start training loop
    for epoch in range(1,EPOCHS+1):
        model.train()
        h = model.init_hidden(batch_size)
        avg_loss = 0
        counter = 0
        for x, label in train_loader:
            counter += 1
            h = h.data
            model.zero_grad()
            out, h = model(x.to(device).float().view(x.size()[0],1,7), h)
            loss = criterion(out, label.to(device).float())
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            print("Epoch {}......Step: {}/{}....... Average Loss for Epoch: {}".format(epoch, counter, len(train_loader), avg_loss/counter))
        TRAIN_LOSS.append(avg_loss/counter)
        test_loss = evaluate(model, test_loader)
        TEST_LOSS.append(test_loss)
        print("Epoch {}/{} Done, Total Loss: {}, Test Loss: {}".format(epoch, EPOCHS, avg_loss/counter, test_loss))
    plt.plot(range(1,EPOCHS+1), TRAIN_LOSS,color='r')
    plt.plot(range(1,EPOCHS+1), TEST_LOSS, color='b')
    plt.show()
    return model

def evaluate(model, test_loader):
    model.eval()
    criterion = nn.MSELoss()
    test_loss = 0
    counter = 0
    for x, label in test_loader:
        counter += 1
        h = model.init_hidden(1)
        out, h = model(x.to(device).float().view(x.size()[0],1,7), h)
        loss = criterion(out, label.to(device).float())
        test_loss += loss.item()
    return test_loss/counter