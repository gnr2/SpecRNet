import os

def count_spoof_bonafide(folder_path):
    spoof_count = 0
    bonafide_count = 0

    # Iterate over all files in the specified folder
    for filename in os.listdir(folder_path):
        # Check if the file is a .wav file and contains either "bonafide" or "spoof"
        if filename.endswith(".wav"):
            if "$$bonafide" in filename:
                bonafide_count += 1
            elif "$$spoof" in filename:
                spoof_count += 1

    return spoof_count, bonafide_count

# Example usage
folder_path = './dataset/train/'  # Replace with your folder path
spoof_count, bonafide_count = count_spoof_bonafide(folder_path)

print(f"Number of bonafide files: {bonafide_count}")
print(f"Number of spoof files: {spoof_count}")
