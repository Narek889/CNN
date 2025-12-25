import os

def delete_folders_with_few_images(root_dir, min_images=5):
    """
    Deletes folders in root_dir that contain fewer than min_images images.

    Args:
        root_dir (str): The root directory to search for folders.
        min_images (int): Minimum number of images required to keep the folder.
    """
    deleted_folders = []

    for foldername in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, foldername)
        if os.path.isdir(folder_path):
            image_count = 0
            for filename in os.listdir(folder_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                    image_count += 1

            if image_count < min_images:
                try:
                    # Remove all files in the folder
                    for filename in os.listdir(folder_path):
                        file_path = os.path.join(folder_path, filename)
                        try:
                            if os.path.isfile(file_path):
                                os.unlink(file_path)
                        except Exception as e:
                            print(f"Failed to delete {file_path}: {e}")

                    # Remove the folder
                    os.rmdir(folder_path)
                    deleted_folders.append(folder_path)
                    print(f"Deleted folder: {folder_path} (only {image_count} images)")
                except Exception as e:
                    print(f"Failed to delete folder {folder_path}: {e}")

    print(f"\nDeleted {len(deleted_folders)} folders with fewer than {min_images} images.")

# Example usage:
delete_folders_with_few_images("C:/Users/narek/Documents/NPUA/CNN_կուրսային/cv2_facial_recognition/training_data", min_images=5)
