import os

def check_files_with_substring(folder_path, substring):
    """
    Check if any file in the specified folder contains the given substring in its name.

    :param folder_path: Path to the folder to search in
    :param substring: Substring to look for in the file names
    :return: List of files containing the substring
    """
    matching_files = []

    # Iterate through all files in the folder
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            # Check if the substring is in the file name
            if substring in file_name:
                matching_files.append(os.path.join(root, file_name))

    # Print results
    if matching_files:
        print(f"Files containing '{substring}':")
        for file in matching_files:
            print(file)
    else:
        print(f"No files found with '{substring}' in the name.")

    return matching_files

# Example usage
folder_path = "/usr/xtmp/zg78/protodbug/datasets/cub200_cropped/clean_top_20"  # Replace with your folder path
substring = "aa6-4b4a-94c5-3a8bff093380"                 # Replace with the substring you're searching for
check_files_with_substring(folder_path, substring)
