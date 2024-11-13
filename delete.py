import os

def delete_files_by_prefix(folder_path, static_prefix, target_count):
    # Track deletion status
    files_deleted = 0
    prefix_str = f"audio ({static_prefix}"

    # Iterate over all files in the specified folder
    for filename in os.listdir(folder_path):
        # Check if the filename starts with the prefix and ends with $$bonafide.wav
        if filename.startswith(prefix_str) and filename.endswith("$$bonafide.wav"):
            file_path = os.path.join(folder_path, filename)
            try:
                os.remove(file_path)
                print(f"Deleted: {filename}")
                files_deleted += 1
            except Exception as e:
                print(f"Error deleting {filename}: {e}")
            
            # Stop if we've deleted the target number of files
            if files_deleted >= target_count:
                break
    
    if files_deleted == 0:
        print(f"No files found with prefix '{static_prefix}' ending with $$bonafide.wav.")
    else:
        print(f"Total files deleted: {files_deleted}")

# Example usage
folder_path = './Bonafide_Dataset_OG/train_copy/'  # Replace with your folder path
static_prefix = "10"  # The static prefix number (e.g., '12')
target_count = 18  # Number of files you want to delete

delete_files_by_prefix(folder_path, static_prefix, target_count)
