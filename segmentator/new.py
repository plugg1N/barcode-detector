import os
from PIL import Image, ImageEnhance
import random
import shutil
import math

# Define the paths to the old and new datasets
old_dataset_path = "Dataset_old"
new_dataset_path = "Dataset_new"

# Create the new dataset folder and subfolders
os.makedirs(new_dataset_path, exist_ok=True)
os.makedirs(os.path.join(new_dataset_path, "1d"), exist_ok=True)
os.makedirs(os.path.join(new_dataset_path, "dmtx"), exist_ok=True)

def augment_image(image_path, output_path):
    # Open the image
    image = Image.open(image_path)
    
    # Randomize augmentation parameters
    brightness_factor = random.uniform(0.4, 4.9)
    contrast_factor = random.uniform(0.4, 4.9)
    saturation_factor = random.uniform(0.4, 4.9)
    
    # Perform random image augmentations
    enhanced_image = ImageEnhance.Brightness(image).enhance(brightness_factor)
    enhanced_image = ImageEnhance.Contrast(enhanced_image).enhance(contrast_factor)
    enhanced_image = ImageEnhance.Color(enhanced_image).enhance(saturation_factor)

    # Generate the augmented image filename
    filename, extension = os.path.splitext(os.path.basename(image_path))
    augmented_filename = filename + "_augmented" + extension
    augmented_output_path = os.path.join(output_path, augmented_filename)
    
    # Save the augmented image
    enhanced_image.save(augmented_output_path)

# Function to copy images
def copy_images(source_folder, destination_folder):
    for filename in os.listdir(source_folder):
        source_file = os.path.join(source_folder, filename)
        destination_file = os.path.join(destination_folder, filename)
        shutil.copy2(source_file, destination_file)

# Copy images from the "1d" subfolder
source_1d_folder = os.path.join(old_dataset_path, "1d")
destination_1d_folder = os.path.join(new_dataset_path, "1d")
copy_images(source_1d_folder, destination_1d_folder)

# Copy images from the "dmtx" subfolder
source_dmtx_folder = os.path.join(old_dataset_path, "dmtx")
destination_dmtx_folder = os.path.join(new_dataset_path, "dmtx")
copy_images(source_dmtx_folder, destination_dmtx_folder)

# Augment images in the "1d" subfolder
source_1d_folder = os.path.join(old_dataset_path, "1d")
destination_1d_folder = os.path.join(new_dataset_path, "1d")
for filename in os.listdir(source_1d_folder):
    source_file = os.path.join(source_1d_folder, filename)
    augment_image(source_file, destination_1d_folder)

# Augment images in the "dmtx" subfolder
source_dmtx_folder = os.path.join(old_dataset_path, "dmtx")
destination_dmtx_folder = os.path.join(new_dataset_path, "dmtx")
for filename in os.listdir(source_dmtx_folder):
    source_file = os.path.join(source_dmtx_folder, filename)
    augment_image(source_file, destination_dmtx_folder)

# Define the paths to the old and new label folders
old_label_folder = "labels"
new_label_folder = "labels_new"

# Create the new label folder
os.makedirs(new_label_folder, exist_ok=True)

# Function to copy label files
def copy_labels(source_folder, destination_folder):
    for filename in os.listdir(source_folder):
        if filename.endswith('.txt'):  # Assuming label files have .txt extension
            source_file = os.path.join(source_folder, filename)
            destination_file = os.path.join(destination_folder, filename)
            shutil.copy2(source_file, destination_file)

def copy_augmented_labels(source_folder, destination_folder):
    for filename in os.listdir(source_folder):
        if filename.endswith('.txt'):  # Assuming label files have .txt extension
            source_file = os.path.join(source_folder, filename)
            
            # Generate the augmented label filename
            filename_without_extension = os.path.splitext(filename)[0]
            augmented_filename = filename_without_extension + "_augmented.txt"
            destination_file = os.path.join(destination_folder, augmented_filename)
            
            shutil.copy2(source_file, destination_file)

def rotate_image(image, angle):
    return image.rotate(angle, expand=True)

def rotate_coordinates(angle, coordinates, img_width, img_height):
    rotated_coordinates = []
    
    for coordinate in coordinates:
        x, y = coordinate
        
        # Convert the coordinates to relative values (0.0 to 1.0)
        x_rel = x / img_width
        y_rel = y / img_height
        
        # Apply rotation to relative coordinates
        x_rot = x_rel * math.cos(math.radians(angle)) - y_rel * math.sin(math.radians(angle))
        y_rot = x_rel * math.sin(math.radians(angle)) + y_rel * math.cos(math.radians(angle))
        
        # Convert the rotated coordinates back to absolute values
        x_abs = x_rot * img_width
        y_abs = y_rot * img_height
        
        # Ensure the coordinates are within the valid range (0.0 to 1.0)
        x_abs = max(0.0, min(x_abs, img_width))
        y_abs = max(0.0, min(y_abs, img_height))
        
        # Append the rotated coordinates to the list
        rotated_coordinates.append((x_abs, y_abs))
    
    return rotated_coordinates

# Iterate over the images in the old dataset
for root, dirs, files in os.walk(old_dataset_path):
    for file in files:
        if file.endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(root, file)
            image_name, image_ext = os.path.splitext(file)
            
            # Get the corresponding label file
            label_file = os.path.join(old_label_folder, f"{image_name}.txt")
            
            # Skip if the label file doesn't exist
            if not os.path.exists(label_file):
                continue
            
            # Read the label file
            with open(label_file, "r") as f:
                labels = f.readlines()
            
            # Generate random rotation angle from -45 to 90 degrees
            angle = random.uniform(-45, 90)
            
            # Open the image
            image = Image.open(image_path)
            
            # Rotate the image
            rotated_image = rotate_image(image, angle)
            
            # Generate the augmented image filename
            augmented_image_name = f"{image_name}_rotated{image_ext}"
            augmented_image_path = os.path.join(new_dataset_path, "1d" if "1d" in root else "dmtx", augmented_image_name)
            
            # Save the rotated image
            rotated_image.save(augmented_image_path)
            
            # Generate the rotated label filename
            rotated_label_file = os.path.join(new_label_folder, f"{image_name}_rotated.txt")
            
            # Rotate the coordinates and write the rotated labels
            with open(rotated_label_file, "w") as f:
                for label in labels:
                    label_parts = label.strip().split(" ")
                    class_number = int(label_parts[0])  # Extract the class number
                    coordinates = [(float(label_parts[i]), float(label_parts[i+1])) for i in range(1, len(label_parts), 2)]
                    
                    img_width, img_height = image.size
                    rotated_coordinates = rotate_coordinates(angle, coordinates, img_width, img_height)
                    
                    # Write the rotated labels to the file
                    f.write(str(class_number))
                    for coord in rotated_coordinates:
                        f.write(f" {coord[0]} {coord[1]}")
                    f.write("\n")

# Copy the original label files to the new dataset folder
copy_labels(old_label_folder, new_label_folder)

# Copy the augmented label files to the new dataset folder
copy_augmented_labels(old_label_folder, new_label_folder)
