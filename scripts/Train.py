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

# Update hyperparameters and settings
batch_size = 64
num_epochs = 10
learning_rate = 0.001

# Define the path to dataset
data_path = "/Users/vidhisagathiya/Documents/NS_14_Project_Part#2/dataset"

# Load custom dataset and split into train, test, and validation
train_loader, test_loader, val_loader = custom_loader(data_path, batch_size, test_size=0.15, val_size=0.15, shuffle_test=False)

# Create the neural network model
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training phase
total_step = len(train_loader)
loss_list = []
acc_list = []
for epoch in range(num_epochs):
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
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),(correct / total) * 100))

# Save the trained model
torch.save(model.state_dict(), 'trained_model.pth')
print('Trained model saved successfully.')

# Testing phase
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# Calculate and print accuracy for the test set
print('Test Accuracy on the test set: {:.2f}%'.format(100 * correct / total))

# Initialize and fit the skorch model
torch.manual_seed(0)
net = NeuralNetClassifier(
    model,
    max_epochs=0,  # Set max_epochs=0 to avoid further training during evaluation
    lr=learning_rate,
    batch_size=batch_size,
    optimizer=torch.optim.Adam,
    criterion=criterion,
    callbacks=[('earlystopping', EarlyStopping())],  
    device=torch.device("cpu"),  
)

# Convert PyTorch tensors to numpy arrays for skorch
X_train = np.concatenate([x.numpy() for x, _ in iter(train_loader)])
y_train = np.concatenate([y.numpy() for _, y in iter(train_loader)])

# Initialize and fit the skorch model
net.initialize()
net.fit(MyDataset(X_train, y_train), y=y_train)

# Make predictions on the validation set
y_pred_list = []
with torch.no_grad():
    for images, labels in val_loader:
        outputs = net.predict_proba(images)
        y_pred_list.append(outputs)

# Concatenate the predictions from the list
y_pred = np.concatenate(y_pred_list).argmax(axis=1)

# Get the true labels for the validation set
y_test = np.array([y for _, y in iter(val_loader.dataset)])

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on the validation set using skorch: {accuracy * 100:.2f}%")

# Generate and display the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
display_labels = ['Anger', 'Bored', 'Engaged', 'Neutral']

# Create ConfusionMatrixDisplay object
cm_display = ConfusionMatrixDisplay(conf_matrix, display_labels=display_labels)

# Plot the confusion matrix
fig, ax = plt.subplots(figsize=(8, 8))
cm_display.plot(cmap='viridis', ax=ax)
plt.show()
