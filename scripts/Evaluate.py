# evaluate.py

import torch
import torch.nn as nn
import torch.utils.data as td
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
from skorch import NeuralNetClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from skorch.callbacks import EarlyStopping
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt

# Import variant models
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
        return self.X[index], self.y[index]

# Function to load custom dataset and split into train, test, and validation sets
def custom_loader(data_path, batch_size, test_size=0.15, val_size=0.15, shuffle_test=False):
    # Define normalization values for dataset
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
def plot_confusion_matrix(y_true, y_pred, display_labels):
    conf_matrix = confusion_matrix(y_true, y_pred)
    cm_display = ConfusionMatrixDisplay(conf_matrix, display_labels=display_labels)

    ax = plt.subplots(figsize=(8, 8))
    cm_display.plot(cmap='viridis', ax=ax)
    plt.show()

# Define the path to dataset
data_path = "/Users/vidhisagathiya/Documents/NS_14_Project_Part#2/dataset"

# Load custom dataset and split into train, test, and validation
train_loader, test_loader, val_loader = custom_loader(data_path, batch_size=64, test_size=0.15, val_size=0.15, shuffle_test=False)

# Evaluate the main model
model_main = CNN()
model_main.load_state_dict(torch.load('trained_model.pth'))
model_main.eval()
y_true_main, y_pred_main = evaluate_model(model_main, val_loader)

# Experiment 1: Vary the Number of Convolutional Layers
model_variant1 = CNNVariant1()
# Update hyperparameters and settings if needed
num_epochs = 10
learning_rate = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_variant1.parameters(), lr=learning_rate)

# Training phase
total_step = len(train_loader)
loss_list = []
acc_list = []

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Forward pass
        outputs = model_variant1(images)
        loss = criterion(outputs, labels)
        loss_list.append(loss.item())

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Train accuracy
        total = labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        acc_list.append(correct / total)

        # Print training statistics
        if (i + 1) % (total_step // 10) == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(), (correct / total) * 100))

# Save the trained model
torch.save(model_variant1.state_dict(), 'trained_model_variant1.pth')
print('Trained model variant 1 saved successfully.')

# Evaluation phase
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

# Make predictions on the validation set
y_true_variant1, y_pred_variant1 = evaluate_model(model_variant1, val_loader)


# Experiment 2: Experiment with Different Kernel Sizes
model_variant2 = CNNVariant2()
# Update hyperparameters and settings if needed
num_epochs = 10
learning_rate = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_variant2.parameters(), lr=learning_rate)

# Training phase
total_step = len(train_loader)
loss_list = []
acc_list = []

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Forward pass
        outputs = model_variant2(images)
        loss = criterion(outputs, labels)
        loss_list.append(loss.item())

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Train accuracy
        total = labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        acc_list.append(correct / total)

        # Print training statistics
        if (i + 1) % (total_step // 10) == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(), (correct / total) * 100))

# Save the trained model
torch.save(model_variant2.state_dict(), 'trained_model_variant2.pth')
print('Trained model variant 2 saved successfully.')

# Evaluation phase
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

# Make predictions on the validation set
y_true_variant2, y_pred_variant2 = evaluate_model(model_variant2, val_loader)

# Print confusion matrix for each model
display_labels = ['Anger', 'Bored', 'Engaged', 'Neutral']  # Update with actual class labels

# Main Model
plot_confusion_matrix(y_true_main, y_pred_main, display_labels)

# Variant 1
plot_confusion_matrix(y_true_variant1, y_pred_variant1, display_labels)

# Variant 2
plot_confusion_matrix(y_true_variant2, y_pred_variant2, display_labels)

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
