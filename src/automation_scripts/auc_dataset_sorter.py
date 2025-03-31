import os
import shutil


def organize_images_into_folders(source_folder, output_folder):
    """Iterates through all of the images and puts 20 images it iterates in a folder"""
    # Get a list of all image files in the source folder
    images = [file for file in os.listdir(source_folder) if file.lower().endswith(
        (".png", ".jpg", ".jpeg", ".gif", ".bmp"))]
    images.sort()  # Sort the images (optional, for consistent ordering)

    # Define how many images per folder
    images_per_folder = 20

    # Iterate through the images in chunks of `images_per_folder`
    for i in range(0, len(images), images_per_folder):
        folder_index = i // images_per_folder + 1
        new_folder_path = os.path.join(output_folder, str(folder_index))

        # Create the new folder if it doesn't exist
        os.makedirs(new_folder_path, exist_ok=True)

        # Move the current batch of images into the new folder
        for image in images[i:i + images_per_folder]:
            src_path = os.path.join(source_folder, image)
            dest_path = os.path.join(new_folder_path, image)
            shutil.copy(src_path, dest_path)

    print(f"Organized {len(images)} images into folders of {
          images_per_folder} at {output_folder}.")


# Example usage
# Replace with the path to your folder
source_folder = "E:\\Thesis\\AUC_SAMDD_COMBINED\\Combined View\\Testing\\Head Down"
# Replace with the path to your desired output folder
output_folder = "./datasets/frame_sequences/train/g"
organize_images_into_folders(source_folder, output_folder)
