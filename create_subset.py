import zipfile
import os
import tqdm

# Define the input parameters
zip_file_path = "D:\WebDownload\imagenet-object-localization-challenge.zip"
output_folder = "D:\WebDownload\Imagenet_subset"
class_file = "data/textshape50.txt"  # Replace with the file containing class list

# Read the class IDs from the text file
with open(class_file, "r") as file:
    class_ids = [line.split(":")[0].strip() for line in file if ":" in line]

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

def extract_with_progress(zip_file, members, destination):
    total = len(members)
    count = 0
    for member in members:
        zip_file.extract(member, destination)
        # count += 1
        # if count % max(1, total // 100) == 0 or count == total:
        #     print(f"Progress: {count / total * 100:.2f}% ({count}/{total})", end="\r")
# Extract only the specified class folders
with zipfile.ZipFile(zip_file_path, 'r') as zf:
    train_members = [member for class_id in class_ids for member in zf.namelist() if member.startswith(f"ILSVRC/Data/CLS-LOC/train/{class_id}/")]

    extract_with_progress(zf, train_members, output_folder)

# Extract corresponding annotation files
annotations_folder = os.path.join(output_folder, "Annotations/train")
os.makedirs(annotations_folder, exist_ok=True)

with zipfile.ZipFile(zip_file_path, 'r') as zf:
    annotation_members = [member for class_id in class_ids for member in zf.namelist() if member.startswith(f"ILSVRC/Annotations/CLS-LOC/train/{class_id}/")]
    extract_with_progress(zf, annotation_members, output_folder)

with zipfile.ZipFile(zip_file_path, 'r') as zf:
    val_data_members = [member for member in zf.namelist() if member.startswith("ILSVRC/Data/CLS-LOC/val/")]
    extract_with_progress(zf, val_data_members, output_folder)

with zipfile.ZipFile(zip_file_path, 'r') as zf:
    val_annotation_members = [member for member in zf.namelist() if member.startswith("ILSVRC/Annotations/CLS-LOC/val/")]
    extract_with_progress(zf, val_annotation_members, output_folder)

print(f"\nExtraction complete. Selected classes, their annotations, and all validation data are extracted to '{output_folder}'.")