from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

## Imports
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

def evaluate_model(model, X_train, y_train, X_test, y_test, class_names):

    if hasattr(model, 'score'):
        train_acc = model.score(X_train,y_train)
        test_acc = model.score(X_test, y_test)
        prediction = model.predict(X_test)
        print(f'Train Accuracy: {train_acc:.4f}')
        print(f'Test Accuracy: {test_acc:.4f}')

    else:
        ## Evaluating using train data that was split
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)

        ## Evaluating using test data that was split
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)

        # Eval loop
        with torch.no_grad():
            y_train_pred = model(X_train_tensor)
            y_test_pred = model(X_test_tensor)
            _, train_predicted = torch.max(y_train_pred, dim=1)
            _, test_predicted = torch.max(y_test_pred, dim=1)
            train_acc = (train_predicted == y_train_tensor).float().mean().item()
            test_acc = (test_predicted == y_test_tensor).float().mean().item()
            print(f'Train Accuracy: {train_acc:.4f}')
            print(f'Test Accuracy: {test_acc:.4f}')
            prediction = test_predicted.numpy()
        
    report = classification_report(y_test, prediction, output_dict=True, target_names=class_names)

    return train_acc, test_acc, report

def evaluate_model_cnn(model, train_loader, test_loader, class_names, device):
    model.eval()
    model.to(device)

    def get_all_preds(loader):
        all_labels = []
        all_preds = []
        all_correct = 0
        all_total = 0
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                all_total += labels.size(0)
                all_correct += (predicted == labels).sum().item()
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
        
        all_acc = all_correct / all_total
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)

        report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
        return all_acc, report
    
    train_acc, _  = get_all_preds(train_loader)
    test_acc, report = get_all_preds(test_loader)

    model.train()
    return train_acc, test_acc, report
    


def plot_loss(train_losses, test_accuracies, model_name):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Training Progress')
    plt.legend()
    plt.grid()
    plt.savefig(f'results/figures/loss_curve_{model_name}.png')
    print("Image Saved")

