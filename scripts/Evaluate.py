import torch
import torch.nn as nn
import torch.utils.data as td
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split, StratifiedKFold
from skorch import NeuralNetClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt


# Import your variant models
from CNN_variant1 import CNNVariant1
from CNN_variant2 import CNNVariant2
from Train import CNN

# Custom skorch dataset to handle PyTorch tensors
class MyDataset(td.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], torch.tensor(self.y[index], dtype=torch.long)  # Assuming labels are integers

# Function to load your custom dataset and split into train, test, and validation sets
def custom_loader(data_path, batch_size, test_size=0.15, val_size=0.15, shuffle_test=False):
    # Define normalization values for your dataset
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.225, 0.225, 0.225])

    # Create an instance of the ImageFolder dataset with the specified transformations
    full_dataset = ImageFolder(
        root=data_path,
        transform=transforms.Compose([
            transforms.RandomCrop(48, 5),
            transforms.ToTensor(),
            normalize
        ])
    )

    # Split the dataset into train, test, and validation sets
    train_dataset, test_val_dataset = train_test_split(full_dataset, test_size=(test_size + val_size), random_state=42, shuffle=True)
    test_dataset, val_dataset = train_test_split(test_val_dataset, test_size=val_size/(test_size + val_size), random_state=42, shuffle=True)

    # Create data loaders for training, testing, and validation
    train_loader = td.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = td.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle_test, pin_memory=True)
    val_loader = td.DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle_test, pin_memory=True)

    return train_loader, test_loader, val_loader

# Function to evaluate the model on a given loader and return metrics
def evaluate_model(model, data_loader):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in data_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            y_true.extend(labels.numpy())
            y_pred.extend(predicted.numpy())

    return y_true, y_pred

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, display_labels, model_name):
    conf_matrix = confusion_matrix(y_true, y_pred)
    cm_display = ConfusionMatrixDisplay(conf_matrix, display_labels=display_labels)

    fig, ax = plt.subplots(figsize=(8, 8))
    cm_display.plot(cmap='viridis', ax=ax)
    ax.set_title(f'Confusion Matrix - {model_name}')
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    plt.show()

# Define the path to your dataset
data_path = "/Users/vidhisagathiya/Documents/NS_14_Project_Part#2/dataset"

# Load your custom dataset and split into train, test, and validation
train_loader, test_loader, val_loader = custom_loader(data_path, batch_size=64, test_size=0.15, val_size=0.15, shuffle_test=False)

# Evaluate the main model
model_main = CNN()
model_main.load_state_dict(torch.load('trained_model.pth'))  # Load pre-trained weights
model_main.eval()

# Evaluation phase
model_main.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model_main(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# Calculate and print accuracy for the test set
print('Test Accuracy on Main Model: {:.2f}%'.format(100 * correct / total))

# Make predictions on the validation set
y_true_main, y_pred_main = evaluate_model(model_main, val_loader)


# Load and evaluate CNN Variant 1
model_variant1 = CNNVariant1()
model_variant1.load_state_dict(torch.load('trained_model_variant1.pth'))  # Load pre-trained weights
model_variant1.eval()

# Evaluation phase for CNN Variant 1
model_variant1.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model_variant1(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# Calculate and print accuracy for the test set
print('Test Accuracy on Variant 1: {:.2f}%'.format(100 * correct / total))

# Make predictions on the validation set for CNN Variant 1
y_true_variant1, y_pred_variant1 = evaluate_model(model_variant1, val_loader)

# Load and evaluate CNN Variant 2
model_variant2 = CNNVariant2()
model_variant2.load_state_dict(torch.load('trained_model_variant2.pth'))  # Load pre-trained weights
model_variant2.eval()

# Evaluation phase for CNN Variant 2
model_variant2.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model_variant2(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# Calculate and print accuracy for the test set
print('Test Accuracy on Variant 2: {:.2f}%'.format(100 * correct / total))

# Make predictions on the validation set for CNN Variant 2
y_true_variant2, y_pred_variant2 = evaluate_model(model_variant2, val_loader)

# Document changes and observations for Variant 2
# ...

# Compare performance
# Print confusion matrix for each model
display_labels = ['Anger', 'Bored', 'Engaged', 'Neutral']  # Update with your actual class labels

# Main Model
plot_confusion_matrix(y_true_main, y_pred_main, display_labels, 'Main Model')

# Variant 1
plot_confusion_matrix(y_true_variant1, y_pred_variant1, display_labels, 'Variant 1')

# Variant 2
plot_confusion_matrix(y_true_variant2, y_pred_variant2, display_labels, 'Variant 2')

# ...

# Calculate and print metrics in a table
metrics_main = precision_recall_fscore_support(y_true_main, y_pred_main, average='macro'), \
                precision_recall_fscore_support(y_true_main, y_pred_main, average='micro'), \
                accuracy_score(y_true_main, y_pred_main)

metrics_variant1 = precision_recall_fscore_support(y_true_variant1, y_pred_variant1, average='macro'), \
                   precision_recall_fscore_support(y_true_variant1, y_pred_variant1, average='micro'), \
                   accuracy_score(y_true_variant1, y_pred_variant1)

metrics_variant2 = precision_recall_fscore_support(y_true_variant2, y_pred_variant2, average='macro'), \
                   precision_recall_fscore_support(y_true_variant2, y_pred_variant2, average='micro'), \
                   accuracy_score(y_true_variant2, y_pred_variant2)

# Print the metrics in a table
print("\nMetrics Comparison:\n")
print("{:<20} {:<20} {:<20} {:<20}".format("", "Precision", "Recall", "F1"))
print("{:<20} {:<20} {:<20} {:<20}".format("-" * 20, "-" * 20, "-" * 20, "-" * 20))

# Main Model Metrics
print("{:<20} {:<20.2f} {:<20.2f} {:<20.2f}".format("Main Model Macro", metrics_main[0][0], metrics_main[0][1], metrics_main[0][2]))
print("{:<20} {:<20.2f} {:<20.2f} {:<20.2f}".format("Main Model Micro", metrics_main[1][0], metrics_main[1][1], metrics_main[1][2]))
print("{:<20} {:<20.2f}".format("Main Model Accuracy", metrics_main[2]))
print("\n")
# Variant 1 Metrics
print("{:<20} {:<20.2f} {:<20.2f} {:<20.2f}".format("Variant 1 Macro", metrics_variant1[0][0], metrics_variant1[0][1], metrics_variant1[0][2]))
print("{:<20} {:<20.2f} {:<20.2f} {:<20.2f}".format("Variant 1 Micro", metrics_variant1[1][0], metrics_variant1[1][1], metrics_variant1[1][2]))
print("{:<20} {:<20.2f}".format("Variant 1 Accuracy", metrics_variant1[2]))
print("\n")
# Variant 2 Metrics
print("{:<20} {:<20.2f} {:<20.2f} {:<20.2f}".format("Variant 2 Macro", metrics_variant2[0][0], metrics_variant2[0][1], metrics_variant2[0][2]))
print("{:<20} {:<20.2f} {:<20.2f} {:<20.2f}".format("Variant 2 Micro", metrics_variant2[1][0], metrics_variant2[1][1], metrics_variant2[1][2]))
print("{:<20} {:<20.2f}".format("Variant 2 Accuracy", metrics_variant2[2]))
print("\n")