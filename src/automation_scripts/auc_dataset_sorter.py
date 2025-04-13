import os
import shutil
import re


def natural_sort_key(s):
    """Function to generate key for natural sorting of strings with numbers"""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]


def organize_images_into_folders(source_folder, output_folder):
    """Iterates through all of the images and puts 20 images it iterates in a folder"""

    views = ["side", "front"]

    for view in views:
        view_folder = os.path.join(source_folder, view)

        print(f"Processing in {view_folder}...")

        # Get a list of all image files in the source folder
        images = [file for file in os.listdir(view_folder) if file.lower().endswith(
            (".png", ".jpg", ".jpeg", ".gif", ".bmp"))]
        images.sort(key=natural_sort_key)  # Sort the images using natural sort

        # Define how many images per folder
        images_per_folder = 20

        # Iterate through the images in chunks of `images_per_folder`
        for i in range(0, len(images), images_per_folder):
            folder_index = i // images_per_folder + 1
            new_folder_path = os.path.join(
                output_folder, f"{view}_{folder_index}")

            # Create the new folder if it doesn't exist
            os.makedirs(f"{new_folder_path}", exist_ok=True)

            # Move the current batch of images into the new folder
            for image in images[i:i + images_per_folder]:
                src_path = os.path.join(view_folder, image)
                dest_path = os.path.join(new_folder_path, image)
                shutil.copy(src_path, dest_path)
    print(
        f"Organized {len(images)} images into folders of {images_per_folder} at {output_folder}.")


# Example usage
source_folder = "E:\\Thesis\\AUC_SAMDD_COMBINED\\New Combined View\\Look Behind"
output_folder = "./datasets/frame_sequences/f"
organize_images_into_folders(source_folder, output_folder)
