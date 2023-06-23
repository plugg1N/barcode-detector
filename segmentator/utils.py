# Start segmentation
def run_segmentation(weights_path: str, img: str, output_path: str):
    import cv2
    import numpy as np
    from ultralytics import YOLO
    from PIL import Image

    image_path = img

    def extend_canvas(img):
        image = Image.open(img)
        x = 200  # horizontal
        y = 200   # vertical

        width, height = image.size
        new_width = width + 2*x
        new_height = height + 2*y

        new_image = Image.new(
            image.mode, (new_width, new_height), (255, 255, 255))
        new_image.paste(image, (x, y))
        new_image.save('new.png')

    extend_canvas(img)
    img = cv2.imread('new.png')
    model = YOLO(weights_path)

    results = list(model(img, conf=0.5))
    result = results[0]

    img = cv2.resize(img, (result.masks.shape[2], result.masks.shape[1]))
    result_array = result.masks.data.cpu().numpy()
    result_array = result_array.reshape(-1, result_array.shape[-1])
    _, thresh = cv2.threshold(result_array, 0.5, 1, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh.astype(
        np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # OR HERE!
    largest_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.01 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    points = approx.reshape((4, 2))

    edge_points = np.array([points], dtype=np.int32)
    rect = cv2.minAreaRect(edge_points)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    cv2.drawContours(img, [box], 0, (0, 0, 255), 2)

    cv2.imwrite(f"{output_path}/{image_path}_predicted.jpeg", img)
    print(f"Image saved to: {output_path}/{image_path}_predicted.jpeg")
    print(f"BBOX coordinates: {box}")


# Create a csv file
def create_csv(data_dir: str, csv_name: str):
    import os
    import pandas as pd

    data_folder = data_dir

    image_files = []
    for folder_name in os.listdir(data_folder):
        folder_path = os.path.join(data_folder, folder_name)
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                image_files.append((file_name, folder_name))

    csv_file_path = csv_name
    df = pd.read_csv(csv_file_path, header=None)

    def map_category(row):
        file_name = row[0]
        for image_file in image_files:
            if file_name == image_file[0]:
                return image_file[1]
        return None

    df['barcode_category'] = df.apply(map_category, axis=1)

    new_csv_file_path = 'newnew.csv'
    df.to_csv(new_csv_file_path, header=False, index=False)
    print("CSV created!")


# Start augmentations
def start_augmentation(data_dir: str, labels_dir: str):
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



# Train model (kinda harsh)
def create_splitup(data_dir: str, split_csv: str):
    import pandas as pd
    import shutil
    import os

    if os.path.exists("train") and os.path.exists("val") and os.path.exists("test"):
        return

    else:
        df = pd.read_csv(split_csv, header=None)

        subdivision_types = ['train', 'val', 'test']
        subfolder_types = ['1d', 'dmtx']

        for subdivision_type in subdivision_types:
            os.makedirs(os.path.join(
                subdivision_type, "images"), exist_ok=True)
            os.makedirs(os.path.join(
                subdivision_type, "labels"), exist_ok=True)
            for category in subfolder_types:
                os.makedirs(os.path.join(subdivision_type,
                            "images", category), exist_ok=True)
                os.makedirs(os.path.join(subdivision_type,
                            "labels", category), exist_ok=True)

        for index, row in df.iterrows():
            filename = row[0].strip()
            subdivision_type = row[1].strip()
            category = row[2].strip()

            image_src = os.path.join(data_dir, category, filename)
            image_dst = os.path.join(
                subdivision_type, 'images', category, filename)
            label_src = os.path.join(
                'labels', f"{os.path.splitext(filename)[0]}.txt")
            label_dst = os.path.join(
                subdivision_type, 'labels', category, f"{os.path.splitext(filename)[0]}.txt")

            shutil.copyfile(image_src, image_dst)

            shutil.copyfile(label_src, label_dst)

        print("Images and labels separated successfully!")


# Create *.yaml file
def create_yaml_division(data_dir: str):
    import os

    full_path = os.path.abspath(data_dir)
    new_path = os.path.normpath(os.path.join(full_path, ".."))

    yaml_string = rf"""train: '{new_path}/train'
val: '{new_path}/val'
test: '{new_path}/test'

nc: 2

names: [ '1d', 'dmtx' ]
"""
    with open('data.yaml', 'w') as file:
        file.write(yaml_string)


# Start training
def start_training(model: str, yaml_path: str, weights_path: str, epochs: int = 50, batch_size: int = 8):
    from ultralytics import YOLO

    model = YOLO(model)

    model.train(data=yaml_path, epochs=epochs, imgsz=640, batch=batch_size,
                project=weights_path, task='segment')
    

# Create augmented csv
def create_augmented_csv(csv_name: str):
    import pandas as pd
    import os
    
    # Read the original CSV file
    df = pd.read_csv(csv_name, header=None)

    # Create a new DataFrame for modified filenames
    new_df = pd.DataFrame(columns=df.columns)

    # Iterate over each row in the original DataFrame
    for index, row in df.iterrows():
        filename = os.path.basename(row[0])  # Extract filename without directory path
        subdivision = row[1]
        classname = row[2]
        
        # Append the original row to the new DataFrame
        new_df = pd.concat([new_df, pd.DataFrame([row], columns=new_df.columns)], ignore_index=True)
        
        # Generate modified filenames and append new rows
        filename_base, extension = os.path.splitext(filename)
        new_filename_1 = filename_base + '_augmented' + extension
        new_filename_2 = filename_base + '_rotated' + extension
        
        new_row_1 = [new_filename_1, subdivision, classname]
        new_row_2 = [new_filename_2, subdivision, classname]
        
        new_df = pd.concat([new_df, pd.DataFrame([new_row_1], columns=new_df.columns)], ignore_index=True)
        new_df = pd.concat([new_df, pd.DataFrame([new_row_2], columns=new_df.columns)], ignore_index=True)

    # Save the new DataFrame to a new CSV file
    new_df.to_csv('modified.csv', index=False, header=False)


def start_validation(weights_path: str, output_path: str = None):
    from ultralytics import YOLO
    import os
    import cv2
    import numpy as np
    from shapely.geometry import Polygon
    import pandas as pd

    absolute_coordinates = []
    predicted_coordinates = []
    sizes = []
    iou_values = []

    model = YOLO(weights_path)

    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

    for root, dirs, files in os.walk('val/images'):
        for file in files:
            ext = os.path.splitext(file)[-1].lower()
            if ext in image_extensions and ('1d' in root or 'dmtx' in root):
                image_path = os.path.join(root, file)

                img = cv2.imread(image_path)
                result = list(model(img, conf=0.5))[0]

                img_width = result.masks.shape[2]
                img_height = result.masks.shape[1]

                size = (img_width, img_height)

                sizes.append(size)

                # Handle finding box coordinates
                img = cv2.resize(img, (img_width, img_height))
                result_array = result.masks.data.cpu().numpy()
                result_array = result_array.reshape(-1, result_array.shape[-1])
                _, thresh = cv2.threshold(result_array, 0.5, 1, cv2.THRESH_BINARY)

                contours, _ = cv2.findContours(thresh.astype(
                    np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                largest_contour = max(contours, key=cv2.contourArea)
                epsilon = 0.01 * cv2.arcLength(largest_contour, True)
                approx = cv2.approxPolyDP(largest_contour, epsilon, True)
                points = approx.reshape((4, 2))

                edge_points = np.array([points], dtype=np.int32)
                rect = cv2.minAreaRect(edge_points)
                box = cv2.boxPoints(rect)
                box = np.intp(box)

                predicted_coordinates.append(box.tolist())

    val_folder = "val"

    for root, dirs, files in os.walk(val_folder):
        for file_idx, file in enumerate(files):
            if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
                image_file_path = os.path.join(root, file)
                label_file_path = os.path.join(root.replace(
                    "images", "labels"), file.replace(file.split('.')[-1], "txt"))

                with open(label_file_path, 'r') as f:
                    label = f.readline().strip().split()
                    polygon = [float(x) for x in label[1:]]

                    # Get the size of the original image
                    image_width, image_height = sizes[file_idx]

                    # Convert relative coordinates to absolute coordinates and round them
                    abs_coords = [polygon[i] * (image_width if i % 2 == 0 else image_height) for i in range(len(polygon))]

                    # Rearrange the coordinates into pairs of X and Y coordinates
                    abs_coords = [(abs_coords[i], abs_coords[i+1]) for i in range(0, len(abs_coords), 2)]

                    # Append the absolute coordinates to the list
                    absolute_coordinates.append(abs_coords)

    def get_iou(boxA, boxB):
        polyA = Polygon(boxA)
        polyB = Polygon(boxB)

        intersection_area = polyA.intersection(polyB).area
        union_area = polyA.union(polyB).area

        iou = intersection_area / union_area

        return iou

    for i in range(len(absolute_coordinates)):
        iou_values.append(get_iou(absolute_coordinates[i-1],
                                  predicted_coordinates[i-1]))

    print(f"AVG IoU: {sum(iou_values) / len(iou_values)}")

    if output_path is not None:
        filenames = []
        for root, dirs, files in os.walk("val"):
            for file in files:
                if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
                    filenames.append(os.path.join(file))

        df = pd.DataFrame({'Filename': filenames, 'IoU': iou_values})

        df.to_csv(f'{output_path}/iou_values.csv', index=False, header=False)

        print(f"CSV Saved to {output_path}/iou_values.csv!")


# Check existance of a splitup
def check_splitup():
    import os

    if os.path.exists("val"):
        return True
    return False
