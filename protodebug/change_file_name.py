import os
import re
# def rename_files_in_folder(folder_path):
#     """
#     Goes through the specified folder and renames files by inserting 'original_' 
#     between repeated class names in the file name, based on the parent directory.
    
#     :param folder_path: Path to the folder to process
#     """
#     for root, _, files in os.walk(folder_path):
#         class_name = os.path.basename(root).split('.')[-1]  # Get the class name from the parent folder
#         for file_name in files:
#             # Check if the class name appears twice in the file name
#             print(class_name, file_name, file_name.count(class_name))
#             if file_name.count(class_name) >= 2:
#                 # Insert 'original_' between the two occurrences of the class name
#                 new_file_name = file_name.replace(
#                     f"{class_name}_{class_name}", f"{class_name}_original_{class_name}", 1
#                 )
                
#                 # Define full paths for renaming
#                 old_file_path = os.path.join(root, file_name)
#                 new_file_path = os.path.join(root, new_file_name)
                
#                 # Rename the file
#                 os.rename(old_file_path, new_file_path)
#                 print(f"Renamed: {old_file_path} -> {new_file_path}")

def rename_files_in_folder(folder_path):
    """
    Goes through the specified folder and renames files by inserting 'original_' 
    between repeated class names in the file name, based on the parent directory.
    
    :param folder_path: Path to the folder to process
    """
    for root, _, files in os.walk(folder_path):
        class_name = os.path.basename(root).split('.')[-1]  # Get the class name from the parent folder

        for file_name in files:
            # Create a case-insensitive pattern to match two occurrences of the class name
            pattern = re.compile(f"({class_name})_(\\1)", re.IGNORECASE)
            
            # Check if the class name appears twice in the file name (case insensitive)
            if pattern.search(file_name) and 'original' not in file_name:
                # Insert 'original_' between the two occurrences of the class name
                new_file_name = pattern.sub(f"\\1_original_\\2", file_name, count=1)
                
                # Define full paths for renaming
                old_file_path = os.path.join(root, file_name)
                new_file_path = os.path.join(root, new_file_name)
                
                # Rename the file
                os.rename(old_file_path, new_file_path)
                print(f"Renamed: {old_file_path} -> {new_file_path}")


# Example usage
# train_cropped_augmented_segmentation
folder_path = "/usr/xtmp/zg78/protodbug/datasets/cub200_cropped/clean_top_20/train_cropped_augmented_segmentation/"
rename_files_in_folder(folder_path)