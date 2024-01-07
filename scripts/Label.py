import os
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# base directory path is where your classes are saved
base_directory = "/Users/vidhisagathiya/Documents/NS_14_Project_Part#2/dataset/"
# folders array is classification Array
folders = ['Bored', 'Engaged', 'Anger', 'Neutral']
# stores randomly picked images
selected_images = []


# Prints bar graph distribution of Images
def bar_graph():
    file_counts = []
    for folder in folders:
        folder_path = os.path.join(base_directory, folder)
        count = count_files_in_folder(folder_path)
        file_counts.append(count)
    plt.bar(folders, file_counts)
    plt.xlabel('Classification')
    plt.ylabel('Number of Samples')
    plt.title('Number of Images in Each Class')
    plt.show()


# Counts number of files in the folder
def count_files_in_folder(folder_path):
    files_in_folder = os.listdir(folder_path)
    count = 0
    for item in files_in_folder:
        full_path = os.path.join(folder_path, item)
        if os.path.isfile(full_path):
            count += 1
    return count


# Picks random number of images from folders
def get_random_images_from_folder(folder_path, num_imgs):
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
    num_samples_to_return = min(num_imgs, len(image_files))
    return random.sample(image_files, num_samples_to_return)


# Prints a standard letter-sized page of 25x25 grid of randomly picked images
def grid_graph():
    total_img = 25
    img_per_folder = total_img // len(folders)

    for folder in folders:
        path_to_folder = os.path.join(base_directory, folder)
        folder_images = get_random_images_from_folder(path_to_folder, img_per_folder)

        for img in folder_images:
            img_path = os.path.join(base_directory, folder, img)
            selected_images.append((img_path, folder))

    while len(selected_images) < total_img:
        extra_folder = random.choice(folders)
        path_to_extra_folder = os.path.join(base_directory, extra_folder)
        folder_images = get_random_images_from_folder(path_to_extra_folder, 1)

        if folder_images:
            img_path = os.path.join(base_directory, extra_folder, folder_images[0])
            selected_images.append((img_path, extra_folder))

    fig, axes = plt.subplots(5, 5, figsize=(10, 10))

    for ax, (img_path, folder_name) in zip(axes.ravel(), selected_images):
        img = mpimg.imread(img_path)
        ax.imshow(img)
        ax.set_title(folder_name)
        ax.axis('off')

    plt.tight_layout()
    plt.show()


# Prints Pixel Intensity Histogram for each image picked in the random 25x25 grid
def pixel_int_dist():
    fig, axes = plt.subplots(5, 5, figsize=(12, 12))

    for ax, (img_path, folder_name) in zip(axes.ravel(), selected_images):
        img = mpimg.imread(img_path)
        if img.ndim == 3 and img.shape[2] == 3:  # Check if image is RGB
            r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
            ax.hist(r.ravel(), bins=256, color='red', alpha=0.5, range=[0, 256], density=True, histtype='stepfilled')
            ax.hist(g.ravel(), bins=256, color='green', alpha=0.5, range=[0, 256], density=True, histtype='stepfilled')
            ax.hist(b.ravel(), bins=256, color='blue', alpha=0.5, range=[0, 256], density=True, histtype='stepfilled')
        ax.set_title(folder_name)
        ax.axis('off')

    plt.tight_layout()
    plt.show()


# Prints the average Pixel Intensity Distribution of all 25 randomly picked images
def pixel_int_dist_histall():
    reds = []
    greens = []
    blues = []

    for img_path, folder_name in selected_images:
        img = mpimg.imread(img_path)

        # Check if image is RGB
        if img.ndim == 3 and img.shape[2] == 3:
            red_channel = img[:, :, 0].ravel()
            reds.extend(red_channel)

            green_channel = img[:, :, 1].ravel()
            greens.extend(green_channel)

            blue_channel = img[:, :, 2].ravel()
            blues.extend(blue_channel)

    # Plots RGB histograms for all randomly picked images combined
    plt.figure(figsize=(10, 5))
    plt.hist(reds, bins=256, color='red', alpha=0.5, range=[0, 256], density=True, label='Red',
             histtype='stepfilled')
    plt.hist(greens, bins=256, color='green', alpha=0.5, range=[0, 256], density=True, label='Green',
             histtype='stepfilled')
    plt.hist(blues, bins=256, color='blue', alpha=0.5, range=[0, 256], density=True, label='Blue',
             histtype='stepfilled')

    plt.legend(loc="upper right")
    plt.title("RGB Intensity Distributions of 25 Images")
    plt.xlabel("Pixel Value")
    plt.ylabel("Density")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    bar_graph()  # Print Bar Graph
    grid_graph()  # Print Grid 25x25
    pixel_int_dist()  # Print 25x25 Pixel Intensity Histogram
    pixel_int_dist_histall()  # Print Average Pixel Intensity Histogram
