import os
from IPython.display import FileLink
#if there is error with IPython, kindly check your python version and IPython Installation
from PIL import Image

#relabels all files in the form of "img_number"
def relabel_files_in_folder(directory_path, prefix="image"):
    all_files = os.listdir(directory_path)

    files_only = []
    for f in all_files:
        full_path = os.path.join(directory_path, f)
        if os.path.isfile(full_path):
            files_only.append(f)

    img_files = []
    for f in files_only:
        if f.endswith(('.jpg')):
            img_files.append(f)

    sorted_img_files = sorted(img_files)

    for idx, img_file in enumerate(sorted_img_files, start=1):
        old_path = os.path.join(directory_path, img_file)
        extension = os.path.splitext(img_file)[1]
        new_name = f"{prefix}_{idx}{extension}"
        new_path = os.path.join(directory_path, new_name)
        os.rename(old_path, new_path)
        print(f"File Renamed from {img_file} to {new_name}")

#converts png images to jpg images
def convert_png_to_jpg(source_folder, destination_folder):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for filename in os.listdir(source_folder):

        if filename.endswith('.png'):

            img_path = os.path.join(source_folder, filename)
            img = Image.open(img_path)

            if img.mode == 'RGBA':
                r, g, b, a = img.split()
                img = Image.merge("RGB", (r, g, b))

            dest_filename = filename[:-4] + '.jpg'
            dest_path = os.path.join(destination_folder, dest_filename)

            img.save(dest_path, 'JPEG', quality=90)

#generates csv file with path of images and labelname
def generate_label_csv(base_directory, csv_filename="image_paths_labels.csv"):
    if not os.path.exists(base_directory):
        print("The specified directory does not exist.")
        return

    data = []

    for root, _, files in os.walk(base_directory):
        for file in files:
            if file.endswith(('.jpg')):
                path = os.path.join(root, file)
                label = os.path.basename(root)
                data.append((path, label))

    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Path", "Label"])
        writer.writerows(data)

    print(f"CSV file generated: {csv_filename}")
    return csv_filename


if __name__ == '__main__':
    source_folder = "/Users/vidhisagathiya/Documents/NS_14_Project_Part#2/dataset/Neutral"
    destination_folder = "/Users/vidhisagathiya/Documents/NS_14_Project_Part#2/dataset/Neutral"
    convert_png_to_jpg(source_folder, destination_folder)

    base_directory = "/Users/vidhisagathiya/Documents/NS_14_Project_Part#2/dataset"
    csv_file = generate_label_csv(base_directory)

    directory_path = "/Users/vidhisagathiya/Documents/NS_14_Project_Part#2/dataset/Neutral"
    relabel_files_in_folder(directory_path)
