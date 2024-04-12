import os
import shutil

def merge_images(root_dir, output_root_dir):
    # Create the output root directory if it doesn't exist
    if not os.path.exists(output_root_dir):
        os.makedirs(output_root_dir)

    # Iterate over each subdirectory (train, test, val)
    for sub_dir in ["train", "test", "val"]:
        # Get the path to the subdirectory
        sub_dir_path = os.path.join(root_dir, sub_dir)

        # Iterate over each nid folder in the subdirectory
        for nid_folder in os.listdir(sub_dir_path):
            nid_folder_path = os.path.join(sub_dir_path, nid_folder)

            # Get the path to the images directory within the nid folder
            images_dir = os.path.join(nid_folder_path, "images")

            # Check if the images directory exists
            if os.path.exists(images_dir) and os.path.isdir(images_dir):
                # Create the corresponding nid folder in the output root directory
                output_nid_folder = os.path.join(output_root_dir, nid_folder)
                if not os.path.exists(output_nid_folder):
                    os.makedirs(output_nid_folder)

                # Create the images directory within the output nid folder
                output_images_dir = os.path.join(output_nid_folder, "images")
                if not os.path.exists(output_images_dir):
                    os.makedirs(output_images_dir)

                # Copy all images from the images directory to the output images directory
                for image_file in os.listdir(images_dir):
                    image_src = os.path.join(images_dir, image_file)
                    image_dest = os.path.join(output_images_dir, image_file)
                    shutil.copy(image_src, image_dest)

# Example usage
root_directory = "./tiny-224"
output_root_directory = "./tiny-224-combined"
merge_images(root_directory, output_root_directory)
