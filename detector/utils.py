# Run a prediction and save the image
from typing import Optional


def run_prediction(weights_path: str, img: str, output_path: str):
    import cv2
    from ultralytics import YOLO
    from ultralytics.yolo.utils.plotting import Annotator
    import os

    model = YOLO(weights_path)
    image = cv2.imread(img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model.predict(image)

    for result in results:
        confidence = result.boxes.conf[0].item()
        class_box = round(result.boxes.cls.item())
        x_top = int(result.boxes.xyxy[0][0])
        y_top = int(result.boxes.xyxy[0][1])
        x_low = int(result.boxes.xyxy[0][2])
        y_low = int(result.boxes.xyxy[0][3])
        print(
            f"class = {'1d' if class_box == 0 else 'dmtx'}; conf = {confidence}; bbox = {x_top} {y_top} {x_low} {y_low}")

        annotator = Annotator(image)
        boxes = result.boxes
        for box in boxes:
            b = box.xyxy[0]
            if confidence >= 0.7:
                # Best confidednce - green
                annotator.box_label(b, color=(0, 255, 0))
            elif confidence < 0.7 and confidence > 0.5:
                annotator.box_label(b, color=(255, 255, 0))  # Medium - yellow
            else:
                annotator.box_label(b, color=(255, 0, 0))  # Worst - red
    image = annotator.result()
    cv2.imwrite(f"{output_path}/{os.path.basename(img)}_predicted.jpg", image)
    print(f"Saved to: {output_path}/{os.path.basename(img)}_predicted.jpg")


# Start data augmentation
def start_augmentation(data_dir: str, labels_dir: str):
    import os
    from PIL import Image, ImageEnhance
    import random
    import shutil
    import math
    import pandas as pd

    # Define the paths to the old and new datasets
    old_dataset_path = data_dir
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
    old_label_folder = labels_dir
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
    
    def rotate_coordinates(angle, x, y, width, height, img_width, img_height):
        # Convert the coordinates to YOLO format
        x_center = x + width / 2
        y_center = y + height / 2

        # Convert the coordinates to relative values
        x_rel = x_center / img_width
        y_rel = y_center / img_height
        width_rel = width / img_width
        height_rel = height / img_height

        # Apply rotation to relative coordinates
        x_rot = x_rel * math.cos(math.radians(angle)) - y_rel * math.sin(math.radians(angle))
        y_rot = x_rel * math.sin(math.radians(angle)) + y_rel * math.cos(math.radians(angle))

        # Convert the rotated coordinates back to absolute values
        x_abs = x_rot * img_width
        y_abs = y_rot * img_height
        width_abs = width_rel * img_width
        height_abs = height_rel * img_height

        # Convert the coordinates back to YOLO format
        x_new = x_abs - width_abs / 2
        y_new = y_abs - height_abs / 2

        return x_new, y_new, width_abs, height_abs
    
    old_dataset_path = data_dir
    new_dataset_path = "Dataset_new"
    labels_path = labels_dir
    new_labels_path = "labels_new"

    # Iterate over the images in the old dataset
    for root, dirs, files in os.walk(old_dataset_path):
        for file in files:
            if file.endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(root, file)
                image_name, image_ext = os.path.splitext(file)

                # Get the corresponding label file
                label_file = os.path.join(labels_path, f"{image_name}.txt")

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
                rotated_label_file = os.path.join(new_labels_path, f"{image_name}_rotated.txt")

                # Rotate the coordinates and write the rotated labels
                with open(rotated_label_file, "w") as f:
                    for label in labels:
                        label_parts = label.strip().split(" ")
                        class_id = label_parts[0]
                        x, y, width, height = map(float, label_parts[1:])

                        img_width, img_height = image.size
                        x_rot, y_rot, width_rot, height_rot = rotate_coordinates(angle, x, y, width, height, img_width, img_height)
                        f.write(f"{class_id} {round(abs(x_rot), 6)} {round(abs(y_rot), 6)} {round(abs(width_rot), 6)} {round(abs(height_rot), 6)}\n")

                # Copy the original label file to the labels_new folder
                new_label_file = os.path.join(new_labels_path, f"{image_name}.txt")
                shutil.copy(label_file, new_label_file)
    
    # Copy label files from the old label folder to the new label folder
    copy_labels(old_label_folder, new_label_folder)

    # Copy augmented label files from the old label folder to the new label folder
    copy_augmented_labels(old_label_folder, new_label_folder)





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
                project=weights_path)

# Evaluate IoU


def IOU(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    x_inter1 = max(x1, x3)
    y_inter1 = max(y1, y3)
    x_inter2 = min(x2, x4)
    y_inter2 = min(y2, y4)
    width_inter = abs(x_inter2 - x_inter1)
    height_inter = abs(y_inter2 - y_inter1)
    area_inter = width_inter * height_inter
    width_box1 = abs(x2 - x1)
    height_box1 = abs(y2 - y1)
    width_box2 = abs(x4 - x3)
    height_box2 = abs(y4 - y3)
    area_box1 = width_box1 * height_box1
    area_box2 = width_box2 * height_box2
    area_union = area_box1 + area_box2 - area_inter
    iou = area_inter / area_union
    return iou


# Validation handling


def start_validation(weights_path: str, data_dir: str, output_path: Optional[str] = None):
    from ultralytics import YOLO
    import cv2
    import os
    from PIL import Image
    import pandas as pd

    val_folder = "val"

    absolute_coordinates = []

    for root, dirs, files in os.walk(val_folder):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
                image_file_path = os.path.join(root, file)
                label_file_path = os.path.join(root.replace(
                    "images", "labels"), file.replace(file.split('.')[-1], "txt"))

                with open(label_file_path, 'r') as f:
                    label = f.readline().strip().split()
                    x_rel = float(label[1])
                    y_rel = float(label[2])
                    width_rel = float(label[3])
                    height_rel = float(label[4])

                    # Get the size of the original image
                    with Image.open(image_file_path) as img:
                        image_width, image_height = img.size

                    # Convert relative coordinates to absolute coordinates
                    x_abs = x_rel * image_width
                    y_abs = y_rel * image_height
                    width_abs = width_rel * image_width
                    height_abs = height_rel * image_height

                    # Calculate absolute coordinates of the bounding box
                    x_min = round(x_abs - (width_abs / 2))
                    y_min = round(y_abs - (height_abs / 2))
                    x_max = round(x_abs + (width_abs / 2))
                    y_max = round(y_abs + (height_abs / 2))

                    # Append the absolute coordinates (without class ID) to the list
                    absolute_coordinates.append([x_min, y_min, x_max, y_max])


# Prediction by a model
    model = YOLO(weights_path)

    predicted_coordinates = []
    predicted_areas = []

    for root, dirs, files in os.walk(val_folder):
        for file in files:
            # Assuming images can have any of these extensions
            if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
                image_file_path = os.path.join(root, file)

                # Read the image using OpenCV and convert it to RGB
                img = cv2.imread(image_file_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Make predictions using the YOLOv5 model
                results = model.predict(img)

                for result in results:
                    confidence = result.boxes.conf[0].item()
                    class_box = round(result.boxes.cls.item())
                    x_top = int(result.boxes.xyxy[0][0])
                    y_top = int(result.boxes.xyxy[0][1])
                    x_low = int(result.boxes.xyxy[0][2])
                    y_low = int(result.boxes.xyxy[0][3])

                    # Append the rounded coordinates of the bounding box to the list
                    predicted_coordinates.append([x_top, y_top, x_low, y_low])

    iou_counter = []

    for i in range(len(absolute_coordinates)):
        iou = IOU(absolute_coordinates[i], predicted_coordinates[i])
        iou_counter.append(iou)

    iou_avg = sum(iou_counter) / len(iou_counter)

    dfs = []

    for root, dirs, files in os.walk(val_folder):
        for file in files:
            # Assuming images can have any of these extensions
            if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
                image_file_path = os.path.join(root, file)
                filename = os.path.splitext(
                    os.path.basename(image_file_path))[0]
                iou = iou_counter.pop(0)

                df = pd.DataFrame({"Filename": [filename], "IoU": [iou]})
                dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    df.to_csv(f"{output_path}/iou_values.csv", index=False, header=False)
    return iou_avg


# Check if splitup exists
def check_splitup():
    import os

    if os.path.exists("val"):
        return True
    return False
