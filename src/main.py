import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from data_processing import load_and_split_data_csv, unfreeze_zip, get_data_loaders;

from evaluate import evaluate_model, evaluate_model_cnn, plot_loss
from logistic_regression import logistic_regression
from neural_network import neural_network
from cnn import cnn

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import torch
import joblib


def main():

    ## run models using csv feature dataset

    files = {
        "3sec": "data/features_3_sec.csv",
        "30sec": "data/features_30_sec.csv"
    }

    results = []

    for file_feature, file_name in files.items():
        # data processing
        X_train, X_test, y_train, y_test, class_names = load_and_split_data_csv(file_name)

        models = {
            "logistic_regression": logistic_regression,
            "neural_network": neural_network
        }

        
        for model_name, train_func in models.items():

            # train model
            model, train_losses, test_accuracies = train_func(X_train, y_train, X_test, y_test)

            # save weights
            if isinstance(model, torch.nn.Module):
                save_path = f"saved_models/{model_name}_{file_feature}.pth"
                torch.save(model.state_dict(), save_path)
                print(f"Saved PyTorch model to {save_path}")

            else:
                save_path = f"saved_models/{model_name}_{file_feature}.joblib"
                joblib.dump(model, save_path)
                print(f"Saved Scikit-learn model to {save_path}")


            # evaluate model
            train_acc, test_acc, report = evaluate_model(model, X_train, y_train, X_test, y_test, class_names)

            # output the result
            f1_score = report['macro avg']['f1-score']

            results.append({
                "Model Name": model_name + f"({file_feature})",
                "Train Accuracy": train_acc,
                "Test Accuracy": test_acc,
                "F1 Macro": f1_score
            })
            
            report_df = pd.DataFrame(report)
            report_df = report_df.T 
            report_df.to_csv(f"results/report_{model_name}_{file_feature}.csv")

            plot_loss(train_losses, test_accuracies, f"{model_name}_{file_feature}")

    unfreeze_zip("archive.zip", "image_data")

    # run CNN model using image dataset
    # GLobal Vars needed
    DATA_DIR = "image_data/Data/images_original"
    BATCH_SIZE = 32
    IMG_SIZE = 128
    LEARNING_RATE = 0.0001
    EPOCHS = 30

    # Deciding where to train this data
    if torch.cuda.is_available():
        DEVICE = 'cuda'
    elif torch.backends.mps.is_available():
        DEVICE = 'mps'
    else:
        DEVICE = 'cpu'
    print(f"Using device: {DEVICE}")


    train_loader, test_loader, class_names = get_data_loaders(DATA_DIR, BATCH_SIZE, IMG_SIZE)

    model, train_losses, test_accuracies = cnn(train_loader, test_loader, len(class_names),LEARNING_RATE, EPOCHS, DEVICE)

    torch.save(model.state_dict(), "saved_models/cnn.pth")

    # evaluate model
    train_acc, test_acc, report = evaluate_model_cnn(model, train_loader, test_loader, class_names, DEVICE)

    # output the result
    f1_score = report['macro avg']['f1-score']

    results.append({
        "Model Name": "CNN",
        "Train Accuracy": train_acc,
        "Test Accuracy": test_acc,
        "F1 Macro": f1_score
    })

    results_df = pd.DataFrame(results)
    results_df.to_csv("results/comparison.csv", index=False)
    
    report_df = pd.DataFrame(report)
    report_df = report_df.T 
    report_df.to_csv(f"results/report_CNN.csv")

    plot_loss(train_losses, test_accuracies, "CNN")
    

if __name__ == "__main__":
    main()