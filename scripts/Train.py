import torch
import torch.nn as nn
import torch.utils.data as td
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
from skorch import NeuralNetClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
from skorch.callbacks import EarlyStopping

# Custom skorch dataset to handle PyTorch tensors
class MyDataset(td.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

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

# Define the neural network model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_input_size = 64 * 12 * 12
        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(self.fc_input_size, 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x

# # Training function
# def train_model(model, train_loader, num_epochs, learning_rate):
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#     total_step = len(train_loader)
#     loss_list = []
#     acc_list = []

#     for epoch in range(num_epochs):
#         for i, (images, labels) in enumerate(train_loader):
#             # Forward pass
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             loss_list.append(loss.item())

#             # Backpropagation and optimization
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             # Train accuracy
#             total = labels.size(0)
#             _, predicted = torch.max(outputs.data, 1)
#             correct = (predicted == labels).sum().item()
#             acc_list.append(correct / total)

#             # Print training statistics
#             if (i + 1) % (total_step // 10) == 0:
#                 print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
#                       .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(), (correct / total) * 100))

#     return model, loss_list, acc_list

# Training function with validation and early stopping
def train_model_with_validation(model, train_loader, val_loader, num_epochs, learning_rate):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    total_step = len(train_loader)
    loss_list = []
    acc_list = []
    best_val_loss = float('inf')
    patience = 5  # Adjust as needed

    for epoch in range(num_epochs):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            # Forward pass
            outputs = model(images)
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

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()

        val_loss /= len(val_loader)
        print(f'Validation Loss: {val_loss}')

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, 'trained_model.pth')

        else:
            patience -= 1
            if patience == 0:
                print("Early stopping. No improvement in validation loss.")
                break

    return model, loss_list, acc_list


# Testing function
def test_model(model, test_loader):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy

# Save model function
def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f'Model saved successfully at {path}')

# Evaluate function using skorch
def evaluate_skorch_model(model, train_loader):
    net = NeuralNetClassifier(
        model,
        max_epochs=0,
        lr=learning_rate,
        batch_size=batch_size,
        optimizer=torch.optim.Adam,
        criterion=nn.CrossEntropyLoss(),
        callbacks=[('earlystopping', EarlyStopping(patience=5, threshold=0.01))],
        device=torch.device("cpu"),
    )

    X_train = np.concatenate([x.numpy() for x, _ in iter(train_loader)])
    y_train = np.concatenate([y.numpy() for _, y in iter(train_loader)])

    net.initialize()
    net.fit(MyDataset(X_train, y_train), y=y_train)

    return net

# Display confusion matrix
def display_confusion_matrix(y_test, y_pred, display_labels):
    conf_matrix = confusion_matrix(y_test, y_pred)
    cm_display = ConfusionMatrixDisplay(conf_matrix, display_labels=display_labels)

    # Plot the confusion matrix
    fig, ax = plt.subplots(figsize=(8, 8))
    cm_display.plot(cmap='viridis', ax=ax)
    plt.show()

if __name__ == "__main__":
    # Update hyperparameters and settings
    batch_size = 64
    num_epochs = 15
    learning_rate = 0.001

    # Define the path to your dataset
    data_path = "/Users/vidhisagathiya/Documents/NS_14_Project_Part#2/new_Dataset"

    # Load your custom dataset and split into train, test, and validation
    train_loader, test_loader, val_loader = custom_loader(data_path, batch_size, test_size=0.15, val_size=0.15, shuffle_test=False)

    # Create the neural network model
    model = CNN()

    # Train the model
    trained_model, loss_list, acc_list = train_model_with_validation(model, train_loader, val_loader, num_epochs, learning_rate)

    # Save the trained model
    save_model(trained_model, 'trained_model.pth')

    # Test the model
    test_accuracy = test_model(trained_model, test_loader)
    print(f'Test Accuracy on the test set: {test_accuracy * 100:.2f}%')

    # Display confusion matrix
    display_labels = ['Anger', 'Bored', 'Engaged', 'Neutral']
