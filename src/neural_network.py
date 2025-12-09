## Imports
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd


def neural_network(X_train, y_train, X_test, y_test):

    print("===Start training neural network model===")

    # Converting train and test t tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)



    in_size = X_train.shape[1]
    print("Input size:", in_size)
    ## Initializing model / Architecture
    class Net(nn.Module):
        def __init__(self, in_size, num_classes=10):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(in_size, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                
                
                nn.Linear(128, 64),
                nn.ReLU(),
                
                nn.Linear(64, 32),
                nn.ReLU(),

                nn.Linear(32, num_classes)
            )

        def forward(self, x):
            return self.layers(x)

    model_NN = Net(in_size)


    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model_NN.parameters(), lr=0.05)

    num_epochs = 6000
    train_losses = []
    test_accuracies = []

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Training loop
    for i in range(num_epochs):
        y_pred = model_NN(X_train_tensor)
        loss = criterion(y_pred, y_train_tensor)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        
        if (i+1) % 200 == 0:
            train_losses.append(loss.item())

            model_NN.eval()
            with torch.no_grad():
                test_outputs = model_NN(X_test_tensor)
                _, test_predicted = torch.max(test_outputs, 1)
                test_acc = (test_predicted == y_test_tensor).float().mean().item()
                test_accuracies.append(test_acc)
            
            model_NN.train()

            print(f'Epoch [{i+1}/{num_epochs}], Loss: {loss.item():.4f}, Test Acc: {test_acc:.4f}')

    return  model_NN, train_losses, test_accuracies