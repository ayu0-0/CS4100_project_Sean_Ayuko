import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np


def logistic_regression(X_train, y_train, X_test, y_test, epochs=100):

    print("===Start training logistic regression model===")

    model = SGDClassifier(loss='log_loss', max_iter=1, warm_start=True, learning_rate='constant', eta0=0.01, random_state=42)
    
    train_losses = []
    test_accuracies = []
    
    classes = np.unique(y_train)
    
    for epoch in range(epochs):
        model.partial_fit(X_train, y_train, classes=classes)
    
        y_proba = model.predict_proba(X_train)
        
        loss = log_loss(y_train, y_proba)
        train_losses.append(loss)

        test_acc = model.score(X_test, y_test)
        test_accuracies.append(test_acc)

        print(f'Epoch [{epoch}/{epochs}], Loss: {loss:.4f}, Test Acc: {test_acc:.4f}')
        
    return model, train_losses, test_accuracies

