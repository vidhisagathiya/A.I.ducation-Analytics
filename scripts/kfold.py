import torch
import torch.nn as nn
import torchvision.transforms as transforms
from sklearn.model_selection import StratifiedKFold
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset
from skorch import NeuralNetClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, classification_report
import numpy as np
from prettytable import PrettyTable

# Hyperparameters
num_epochs = 10
num_classes = 4
learning_rate = 0.001
random_seed = 42

# Image transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Dataset path
local_dataset_path = "/Users/vidhisagathiya/Documents/NS_14_Project_Part#2/Age/Young"
dataset = ImageFolder(root=local_dataset_path, transform=transform)

# Split dataset into train and test sets
torch.manual_seed(random_seed)
num_folds = 10
skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=random_seed)

# Class labels
classes = ('Anger', 'Bored', 'Engaged', 'Neutral')

# Extract labels for the entire dataset
y = np.array([y for x, y in iter(dataset)])

# Convolutional Neural Network (CNN) model definition
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

        # Fully connected layers
        out_size = lambda i, k, p , s: (i - k + 2 * p) // s + 1
        out1 = out_size(96, 3, 1, 1)  # Output size after conv1
        out2 = out_size(out1, 2, 0, 2)  # Output size after pool1
        out3 = out_size(out2, 3, 1, 1)  # Output size after conv2
        out4 = out_size(out3, 2, 0, 2)  # Output size after pool2
        expected_input_size = out4 * out4 * 64  # Assuming 64 channels in the last conv layer

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(expected_input_size, 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 4)
        )

    def forward(self, x):
        # Forward pass
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x

if __name__ == '__main__':
    # Initialize CNN model
    model = CNN()

     # Table to store results
    table = PrettyTable()
    table.field_names = ["Fold", "Accuracy", "Precision", "Recall", "F1-score"]

    # Define NeuralNetClassifier
    for fold, (train_index, test_index) in enumerate(skf.split(dataset, y)):
        print(f'\nFold {fold + 1}/{num_folds}')

        # Split dataset into train and test sets for the current fold
        train_dataset_fold = Subset(dataset, train_index)
        test_dataset_fold = Subset(dataset, test_index)

        # Initialize NeuralNetClassifier for the current fold
        net = NeuralNetClassifier(
            model,
            criterion=nn.CrossEntropyLoss(),
            optimizer=torch.optim.Adam,
            max_epochs=num_epochs,
            lr=learning_rate,
            batch_size=32,
            iterator_train__shuffle=True,
            device='cuda:0' if torch.cuda.is_available() else 'cpu'
        )

        # Training loop for the current fold
        for epoch in range(num_epochs):
            # Training
            net.fit(train_dataset_fold, y=y[train_index])

        # Testing for the current fold
        test_acc = net.score(test_dataset_fold, y[test_index])
        print(f'Test Accuracy: {test_acc * 100:.2f}%')

        # Evaluate
        y_pred = net.predict(test_dataset_fold)
        acc = accuracy_score(y[test_index], y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y[test_index], y_pred, average='weighted', zero_division=1)

        # Print metrics for the current fold
        print(f'Test Accuracy: {acc * 100:.2f}%')
        print("Classification Report:")
        print(classification_report(y[test_index], y_pred, target_names=classes))
        print(f'Precision: {precision:.2f}')
        print(f'Recall: {recall:.2f}')
        print(f'F1-measure: {f1:.2f}')

        # Add results to the table
        table.add_row([fold + 1, f'{acc * 100:.2f}%', f'{precision:.2f}', f'{recall:.2f}', f'{f1:.2f}'])

    # Print the table
    print(table)

    # Calculate and print average metrics across all folds
    avg_accuracy = np.mean(table.get_column("Accuracy"))
    avg_precision = np.mean(table.get_column("Precision"))
    avg_recall = np.mean(table.get_column("Recall"))
    avg_f1 = np.mean(table.get_column("F1-score"))

    print("\nAverage Across All Folds:")
    print(f'Average Accuracy: {avg_accuracy:.2f}%')
    print(f'Average Precision: {avg_precision:.2f}')
    print(f'Average Recall: {avg_recall:.2f}')
    print(f'Average F1-measure: {avg_f1:.2f}')