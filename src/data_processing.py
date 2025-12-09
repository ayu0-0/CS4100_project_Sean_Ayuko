import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Needed Imports
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import classification_report
import zipfile
import os
from tqdm import tqdm


def load_and_split_data_csv(file_path):

    # Importing data
    df = pd.read_csv(file_path)
    print("Shape all data:", df.shape)

    # Encoding labels
    le = LabelEncoder()
    df['label_encoded'] = le.fit_transform(df['label'])

    

    y = df["label_encoded"].to_numpy()
    X = df.drop(columns=["filename", "length", "label", "label_encoded"]).to_numpy()

    [X_train, X_test, y_train, y_test] = train_test_split(X, y, test_size = .2, random_state=42)

    class_names = le.classes_

    print("\nCategories Encoded:")
    for i, category in enumerate(class_names):
        print(f"{category}: {i}")

    print("\nTrain Size:", len(X_train[:]))
    print("Test Size:", len(X_test[:]))

    scaler = StandardScaler().fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)       

    return X_train, X_test, y_train, y_test, class_names


## Loading data into Local dir in format needed to use
def unfreeze_zip(zip_path, extract_to):
    if os.path.exists(zip_path):
        print("Starting extraction...")
        
        # すでに解凍済みであればスキップするロジックを追加すると便利です
        if os.path.exists(extract_to) and len(os.listdir(extract_to)) > 5:
            print("Data already extracted. Skipping extraction.")
            return


        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:

                file_list = zip_ref.namelist()
                total_files = len(file_list)


                for member in tqdm(file_list, total=total_files, desc="Extracting"):
                    zip_ref.extract(member, extract_to)
                
                print("\nExtraction complete.")

        except Exception as e:
            print(f"\nExtraction Error occurred: {e}")
    else:
        print(f"error")

def get_data_loaders(data_dir, batch_size, img_size):
    # Preprocessing pipeline for images
    # Transformin / normalizing image data
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    try:
        full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)
    except Exception as e:
        exit()

    class_names = full_dataset.classes
    print("\nCategories Encoded:")
    for i, category in enumerate(class_names):
        print(f"{category}: {i}")

    # Actually splitting test / train data
    generator = torch.Generator().manual_seed(42)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_data, test_data = random_split(full_dataset, [train_size, test_size], generator=generator)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    print("\nTrain Size:", train_size)
    print("Test Size:", test_size)

    return train_loader, test_loader, class_names

