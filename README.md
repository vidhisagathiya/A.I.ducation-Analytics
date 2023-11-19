# A.I.ducation-Analytics
The objective of this project is to develop a Deep Learning Convolutional Neural Network (CNN) using PyTorch that can analyze images of students in a classroom or online meeting setting and categorize them into distinct states or activities. Your system has to be able to analyse images in order to recognize four classes.

# Project README

## Contents

1. [Introduction](#introduction)
2. [File Structure](#file-structure)
3. [Data Cleaning](#data-cleaning)
4. [Data Visualization](#data-visualization)
5. [Training the CNN Model](#training-the-cnn-model)
6. [CNN Variants](#cnn-variants)
7. [Evaluation](#evaluation)
8. [Contributors](#contributors)

## Introduction

Welcome to our A.I.ducation Analytics project. This README provides an overview of the project's contents, file structure, and instructions for data cleaning, data visualization, training the CNN model, CNN variants, and evaluation.

## File Structure

The project submission consists of the following files and directories:

- **/Dataset**: Contains the dataset and cleaned data.
- **/scripts**: Includes Python scripts for data cleaning and data visualization.
- **/report**: Holds project reports and documentation.
- **README.md**: This file, providing an overview of the project.

## Data Cleaning

Data cleaning is an essential part of our project. 
We already have a cleaned Dataset. No actions needed. Read below for the next steps.

## Image Processing

### Converting PNG to JPG

- The function convert_png_to_jpg converts .png images to .jpg.
- Specify the source_folder and destination_folder paths at the bottom section of the code to set source and target directories.
- Run the program to execute the image conversion.

### Image Relabeling

- The function relabel_files_in_folder renames all .jpg files in a directory in the format "image_number".
- Set the directory_path at the bottom of the code to the desired folder for relabeling.
- Run the program to execute the relabeling process.


## Data Visualization

Data visualization is crucial for gaining insights from the dataset. In the `/scripts` directory, you will find Python scripts for data visualization. Follow these steps to execute the data visualization process:

## Training-the-cnn-model

Training a Convolutional Neural Network (CNN) using PyTorch. It includes custom dataset loading, model definition, training, and evaluation phases.

## CNN-variants
Comparing the performance of final architecture against two distinct architectural variants to gain deeper insights 
- a variant of the CNN model (CNNVariant1) with additional convolutional layers.
- another variant of the CNN model (CNNVariant2) with different convolutional layers and kernel sizes. 

## Evaluation
Evaluate the performance of your main model and its two variants, using the same testing data for all models: For each model, generate a confusion matrix to visualize classification performance. Also evaluate the metrics accuracy, precision, recall, and F1- measure.

## Prerequisites:

Ensure you have the required libraries installed:

"pip install matplotlib" 

To set up the necessary environment using conda, follow the steps below:
- create a new conda environment. This helps in maintaining a clean workspace
and avoids conflicts with other packages. Open your terminal or Anaconda prompt
and run: conda create --name pytorch_env python=3.8
- Activate the environment: conda activate pytorch_env
- Install PyTorch and torchvision using the official channel:2
conda install pytorch torchvision -c pytorch
- other required libraries such as matplotlib for visualization,: conda install matplotlib

## Configuration

Modify the base_directory variable in the below script if your dataset is located in a different directory.
- Dataset_Visualization.py
- Train.py
- App_model.py

Update the image path (image_path) in App_model.py to predict for a single image
Update the folders list in the Dataset_Visualization.py script if your dataset has different classes or folders.

## Execution

Execute the program by running the following command in your terminal:

**"Dataset_Visualization.py"**

After execution, the program will display:

- A bar graph showing the number of images in each class.
- A 5x5 grid of randomly selected images from each class.
- Pixel intensity distributions for each image in the 5x5 grid.
- A combined pixel intensity distribution for all the selected images.
- Explore and analyze these visualizations to gain valuable insights from the dataset.


**CSV Generation for Image Paths and Labels:** 

- The function generate_label_csv scans a directory (including its subdirectories) for .jpg images and generates a CSV file with their paths and labels (folder names).
- Set the base_directory variable at the bottom of the code to point to your dataset's root directory.
- Run the program to create the CSV file, named image_paths_labels.csv by default.

**"Train.py"**
- Defines and trains a CNN using PyTorch on a custom image dataset, saving the trained model (trained_model.pth).
- Evaluates and prints the accuracy of the PyTorch model on the test set.
- Utilizes skorch to integrate the PyTorch model, training and evaluating it on the validation set.
- Prints the accuracy of the skorch model on the validation set.
- Generates and displays a confusion matrix for model evaluation using matplotlib.

**"App_model.py"**

- Utilizes the loaded model to predict the class of a single image and prints the result.
- Plots the provided single image along with the predicted class for visual verification.
- Generates a grid of random images from the dataset, displaying true and predicted labels for model evaluation.
- Loads the custom dataset and prepares it for prediction and visualization.
- Loads the pre-trained dynamic CNN model with the specified input channels.
  
**"CNN_variant1.py"**
- A variant of the CNN model with additional convolutional layers, enhancing its capacity to capture complex features.
- Extends the original model with two extra convolutional layers (64 channels each) and corresponding batch normalization and LeakyReLU activations.
- Utilizes max-pooling layers for down-sampling, enhancing spatial feature extraction.
- Retains the fully connected layers for classification, maintaining a structure of 64 * 12 * 12 input features.
- Outputs predictions for 10 classes (adjustable) after dropout and ReLU activations in the fully connected layers.

**"CNN_variant2.py"**
- Introduces a variant of the CNN model with experimentation on kernel sizes to enhance feature extraction.
- Employs convolutional layers with diverse kernel sizes (e.g., 3x3, 5x5, 2x2) for experimentation and improved feature representation.
- Utilizes max-pooling layers for down-sampling, maintaining spatial information.
- Retains fully connected layers with dropout and ReLU activations for classification tasks, adapting to diverse feature representations.
- Generates predictions for 10 classes (adjustable) after the final fully connected layers.

**"Evaluate.py":**
- Loads a pre-trained PyTorch model ("trained_model.pth") and evaluates its performance on a validation set.
- Conducts two experiments with variant CNN models, saving the trained models as "trained_model_variant1.pth" and "trained_model_variant2.pth".
- Plots confusion matrices for the main model and variant models on the validation set.
- Calculates and prints precision, recall, F1-score, and accuracy metrics for the main model and variants in a tabular format.

  
## Contributors

- Anurag Agarwal
- Vidhi Sagathiya
- Jimi Mehta

---

We hope you find this README helpful in navigating our project. If you encounter any issues or have questions, please don't hesitate to reach out to our team.

Thank you for exploring A.I.ducation Analytics!
