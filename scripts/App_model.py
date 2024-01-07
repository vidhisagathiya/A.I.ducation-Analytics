import torch
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
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


# Dynamic CNN class that allows varying architecture
class DynamicCNN(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(DynamicCNN, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1),
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

def denormalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def load_model(input_channels, filepath="trained_model.pth"):
    model = DynamicCNN(input_channels, 10)
    model.load_state_dict(torch.load(filepath))
    return model

def predict_single_image(image_path, model, class_names=None):
    transform = transforms.Compose([
        transforms.RandomCrop(48, 5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.225, 0.225, 0.225])
    ])

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension
    model.eval()
    with torch.no_grad():
        output = model(image)
        _, predicted_class = torch.max(output, 1)
    
    if class_names is not None:
        predicted_class_name = class_names[int(predicted_class)]
        return predicted_class_name
    else:
        return predicted_class.item()


def predict_dataset(dataset_loader, model):
    model.eval()
    predictions = []
    with torch.no_grad():
        for images, _ in dataset_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.cpu().numpy())
    return predictions

def plot_single_image_prediction(image_path, model, class_names=None):
    transform = transforms.Compose([
        transforms.RandomCrop(48, 5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.225, 0.225, 0.225])
    ])

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    model.eval()
    
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_class = torch.max(output, 1)
    
    predicted_class_name = class_names[int(predicted_class)] if class_names is not None else str(predicted_class.item())

    # Denormalize image for plotting
    denormalized_img = denormalize(input_tensor.squeeze(), [0.485, 0.456, 0.406], [0.225, 0.225, 0.225])
    img_for_plot = denormalized_img.permute(1, 2, 0).numpy()

    # Display the image with the predicted class name
    plt.imshow(img_for_plot)
    plt.title(f"Predicted class: {predicted_class_name}")
    plt.axis('off')
    plt.show()


def plot_results(model, data_loader, class_names=None, x=2, y=5):
    # Width per image (inches)
    width_per_image = 2.4

    # Get a batch of images and labels
    data_iter = iter(data_loader)
    images, labels = next(data_iter)

    # Select x*y random images from the batch
    indices = np.random.choice(images.size(0), x * y, replace=False)
    random_images = images[indices]
    random_labels = labels[indices]

    # Get predictions for these images
    random_images_reshaped = random_images.reshape(-1, input_channels, 48, 48)
    outputs = model(random_images_reshaped)
    _, predicted = torch.max(outputs.data, 1)

    # Create subplots
    fig, axes = plt.subplots(x, y, figsize=(y * width_per_image, x * width_per_image))

    # Iterate over the random images and display them along with their predicted labels
    for i, ax in enumerate(axes.ravel()):
        # Denormalize image
        img = denormalize(random_images[i], [0.485, 0.456, 0.406], [0.225, 0.225, 0.225])
        img = img.permute(1, 2, 0).numpy()  # Convert image from CxHxW to HxWxC format for plotting
        true_label = str(random_labels[i].item())
        pred_label = str(predicted[i].item())

        if class_names is not None:
            true_label = class_names[int(true_label)]
            pred_label = class_names[int(pred_label)]

        ax.imshow(img)
        ax.set_title(f"true='{true_label}', pred='{pred_label}'", fontsize=8)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Define the path to your dataset
    data_path = "/Users/vidhisagathiya/Documents/NS_14_Project_Part#2/dataset"
    input_channels = 3  # Adjust based on your dataset
    num_classes = 4  # Adjust based on your dataset

    # Load your custom dataset
    dataset = ImageFolder(
        root=data_path,
        transform=transforms.Compose([
            transforms.RandomCrop(48, 5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.225, 0.225, 0.225])
        ])
    )

    # Create data loader for the entire dataset
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, pin_memory=True)

    # Load the trained model
    loaded_model = load_model(input_channels)

     # Specify class names
    class_names = ["Anger", "Bored", "Engaged", "Neutral"]

    # Predict for a single image
    image_path = "/Users/vidhisagathiya/Documents/NS_14_Project_Part#2/image_8.jpg"  # Provide the path to your single image
    predicted_class = predict_single_image(image_path, loaded_model, class_names)
    print(f"Predicted class for the single image: {predicted_class}")
    plot_single_image_prediction(image_path, loaded_model, class_names)

    # Plot random images with predictions
    plot_results(loaded_model, dataset_loader, class_names=class_names)
