import torch
from datetime import datetime
import numpy as np
import os
import gc
from model import *
import torch.nn as nn
from data import *

if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu" 

device = torch.device(dev)

batch_size = 256
EPOCHS = 40
learning_rate = 0.01
    
print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

dataset = loadDataSet("./data_preparation")
dataset_train, dataset_validation = torch.utils.data.random_split(dataset, [0.8, 0.2])
dataloader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
dataloader_validation = torch.utils.data.DataLoader(dataset_validation, batch_size=batch_size, shuffle=False)

print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

model = CustomNetV3(device).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(f'number of params: {params}')
print(f'dataset size: {len(dataset)}')

# Taken and adjusted from https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
def train_one_epoch(epoch_index):
    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(dataloader):
        # Every data instance is an input + label pair
        inputs = data['fen']
        labels = data['eval']

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(torch.reshape(inputs, (-1, 13, 8, 8)))

        # Compute the loss and its gradients
        loss = criterion(outputs.squeeze(), labels.squeeze())
        loss.backward()

        # Adjust learning weights
        optimizer.step()
        # Gather data and report
        running_loss += loss.item()
        if i % 500 == 499:
            last_loss = running_loss / 500 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            running_loss = 0.
          
        del inputs
        del labels
        del loss

    gc.collect()

    return last_loss

class LogWriter():
    def __init__(self, path):
        self.file = open(path, 'a')

    def __del__(self):
        self.file.close()

    def write(self, s):
        self.file.write(s)
        self.file.flush()

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
path = './runs/stockfish_ai_{}'.format(timestamp)
print(path)
os.makedirs(path)
logwriter = LogWriter(path + '/log.txt')
logwriter.write("train\tval\t\n")
epoch_number = 0

best_vloss = 1_000_000.

losses = []
vlosses = []
for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch_number)

    # We don't need gradients on to do reporting
    model.train(False)

    running_vloss = 0.0
    for i, vdata in enumerate(dataloader_validation):
        vinputs = vdata['fen']
        vlabels = vdata['eval']
        voutputs = model(torch.reshape(vinputs, (-1, 13, 8, 8)))
        vloss = criterion(voutputs.squeeze(), vlabels.squeeze())
        running_vloss += vloss.item()

        del vinputs
        del vlabels
        del vloss

    gc.collect()
    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    losses.append(avg_loss)
    vlosses.append(avg_vloss)
    # Log the running loss averaged per batch
    # for both training and validation
    logwriter.write('{:.5f}\t{:.5f}\n'.format(avg_loss, avg_vloss))

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = os.path.join(path, 'model_{}_{}'.format(timestamp, epoch_number))
        torch.save(model.state_dict(), model_path)

    epoch_number += 1