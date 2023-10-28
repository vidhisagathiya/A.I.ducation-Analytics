# A.I.ducation-Analytics
The objective of this project is to develop a Deep Learning Convolutional Neural Network (CNN) using PyTorch that can analyze images of students in a classroom or online meeting setting and categorize them into distinct states or activities. Your system has to be able to analyse images in order to recognize four classes.

# Project README

## Contents

1. [Introduction](#introduction)
2. [File Structure](#file-structure)
3. [Data Cleaning](#data-cleaning)
4. [Data Visualization](#data-visualization)
5. [Usage](#usage)
6. [Contributors](#contributors)

## Introduction

Welcome to our A.I.ducation Analytics project. This README provides an overview of the project's contents, file structure, and instructions for data cleaning and data visualization.

## File Structure

The project submission consists of the following files and directories:

- **/DatasetAI**: Contains the dataset and cleaned data.
- **/scripts**: Includes Python scripts for data cleaning and data visualization.
- **/reports**: Holds project reports and documentation.
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

## Prerequisites:

Ensure you have the required libraries installed:

"pip install matplotlib"

## Configuration

Modify the base_directory variable in the main_program.py script if your dataset is located in a different directory.

Update the folders list in the main_program.py script if your dataset has different classes or folders.

## Execution

Execute the program by running the following command in your terminal:

**"python main_program.py"**

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


## Contributors

- Anurag Agarwal
- Vidhi Sagathiya
- Jimi Mehta

---

We hope you find this README helpful in navigating our project. If you encounter any issues or have questions, please don't hesitate to reach out to our team.

Thank you for exploring A.I.ducation Analytics!
