import pandas as pd
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path).convert("RGB")
        label = self.annotations.iloc[index, 2]

        if self.transform:
            image = self.transform(image)

        return image, label

class ImageAugmentor:
    def __init__(self, brightness=0.2, crop_size=256, rotation=10, contrast=0.2):
        self.transforms = transforms.Compose([
            transforms.RandomRotation(rotation),
            transforms.RandomCrop(crop_size),
            transforms.ColorJitter(brightness=brightness, contrast=contrast),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

    def augment(self, image):
        return self.transforms(image)

def generate_augmented_images(csv_file, root_dir):
    df = pd.read_csv(csv_file)
    augmentor = ImageAugmentor()

    for i, row in df.iterrows():
        img_path = os.path.join(root_dir, row["filename"])
        img = Image.open(img_path).convert("RGB")

        # Generate and save augmented image
        for j in range(3): # generate 3 augmented images for each original image
            augmented_img = augmentor.augment(img)
            new_filename = os.path.splitext(row["filename"])[0] + f"_modified{j}.jpeg"
            new_filepath = os.path.join(root_dir, new_filename)
            augmented_img.save(new_filepath)

            # Add augmented image to CSV
            df = df.append({"filename": new_filename, "subdivision type": row["subdivision type"], "category": row["category"]}, ignore_index=True)

    # Save updated CSV
    df.to_csv(csv_file, index=False)

# Example usage:
csv_file = "images.csv"
root_dir = "/path/to/images"

# Generate augmented images and update CSV
generate_augmented_images(csv_file, root_dir)

# Create CustomDataset object for training
train_dataset = CustomDataset(csv_file, root_dir, transform=transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor()]))
