from glob import glob
from compose import compose
from bidict import bidict
import os
from enum import Enum
from tqdm.auto import tqdm
from operator import itemgetter
from functools import partial
from sklearn.metrics import classification_report

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import shutil
from typing import Optional

# REVIEW: cases when one *actually* needs to put imports inside subroutines are rare
# moslty this is used for rare usecases involving optionally installed libraries.
# This could be the case if the training was performed using tf and
# the inferenece via another framework, e.g. onnx. Whenver all methods share
# the roughly the same set of imports, they should be placed in the preambola
# of the file.

class Subdataset(Enum):
        TRAIN = 'train'
        VAL = 'val'
        TEST = 'test'

IMG_SIZE = (50, 50)
LABEL_MAP = bidict({"other": 0, "1d": 1, "dmtx": 2})


# Will crearte new csv adding 3-rd column, containing image categories
def create_updated_csv(split_csv: str, data_dir: str) -> pd.DataFrame:
    # REVIEW: there is no need to match the names here: you could include them
    # into the original csv (relative to data_dir)
    # Secondly, this method can be rewritten both dramatically shorter and faster
    df_classes = pd.DataFrame({'path': glob(os.path.join(data_dir, "*/*"))})
    df_classes['class'] = df_classes['path'].apply(compose(os.path.basename, os.path.dirname))
    df_classes['name'] = df_classes['path'].apply(os.path.basename)
    df_split = pd.read_csv(split_csv)
    df = df_split.merge(df_classes, how = 'right', left_on = 'image', right_on = 'name')
    # REVIEW: as you are already composing a custom dataframe, keep the local paths
    return df[['image', 'subset', 'class', 'path']]

    # data_folder = data_dir

    # if os.path.isfile(output_csv_name):
    #     print("CSV already exists. Aborting creation...")
    #     return

    # image_files = []
    # REVIEW: this is an O(n^2) implementation of 'merge' :(
    # for folder_name in os.listdir(data_folder):
    #     folder_path = os.path.join(data_folder, folder_name)
    #     if os.path.isdir(folder_path):
    #         for file_name in os.listdir(folder_path):
    #             image_files.append((file_name, folder_name))

    # csv_file_path = split_csv
    # df_split = pd.read_csv(csv_file_path)

    # def map_category(row):
    #     file_name = row[0]
    #     for image_file in image_files:
    #         if file_name == image_file[0]:
    #             return image_file[1]
    #     return None

    # df_split['barcode_category'] = df_split.apply(map_category, axis=1)

    # new_csv_file_path = output_csv_name
    # df_split.to_csv(new_csv_file_path, header=False, index=False)
    # print("Additional CSV has been created. OK")


# Handle training process
def start_training(data_dir: str, split_csv: pd.DataFrame, weights_path: str,
                   epochs: int = 14, batch_size: int = 20):
    # Make splits for "train", "test", "val"
    def make_split(data_dir: str, split_csv: pd.DataFrame):
        dataset_path = data_dir

        splits_dir = os.path.join(dataset_path, "Splits")
        train_dir = os.path.join(splits_dir, "train")
        val_dir = os.path.join(splits_dir, "val")
        test_dir = os.path.join(splits_dir, "test")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        df = split_csv

        for index, row in df.iterrows():
            filename = row[0]
            subdivision_type = row[1]
            class_type = row[2]
            source_path = os.path.join(dataset_path, class_type, filename)
            if subdivision_type == "train":
                destination_path = os.path.join(train_dir, filename)
            elif subdivision_type == "val":
                destination_path = os.path.join(val_dir, filename)
            else:
                destination_path = os.path.join(test_dir, filename)
            shutil.copy(source_path, destination_path)

    if os.path.exists(f"{data_dir}/Splits"):
        print("Splits are already created, skipping...")
    else:
        make_split(data_dir, split_csv)

    TARGET_SIZE = (60, 60)

    dataset_path = os.path.join(data_dir, "Splits")

    df = split_csv

    train_df = df[df["subset"] == "train"]
    val_df = df[df["subset"] == "val"]
    test_df = df[df["subset"] == "test"]

    batch_size = batch_size

    train_datagen = ImageDataGenerator(rescale=1./255,
                                       rotation_range=45,
                                       width_shift_range=0.1,
                                       height_shift_range=0.1,
                                       brightness_range=(0.1, 0.6),
                                       shear_range=0.2,
                                       zoom_range=0.4,
                                       horizontal_flip=True,
                                       vertical_flip=False,
                                       fill_mode="nearest")

    val_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_dataframe(
        train_df,
        directory=os.path.join(dataset_path, "train"),
        x_col="image",
        y_col="class",
        target_size=TARGET_SIZE,
        batch_size=batch_size,
        class_mode="categorical"
    )

    val_generator = val_datagen.flow_from_dataframe(
        val_df,
        directory=os.path.join(dataset_path, "val"),
        x_col="image",
        y_col="class",
        target_size=TARGET_SIZE,
        batch_size=batch_size,
        class_mode="categorical"
    )

    test_generator = test_datagen.flow_from_dataframe(
        test_df,
        directory=os.path.join(dataset_path, "test"),
        x_col="image",
        y_col="class",
        target_size=TARGET_SIZE,
        batch_size=batch_size,
        class_mode="categorical"
    )

    num_classes = len(train_df["class"].unique())
    input_shape = (TARGET_SIZE[0], TARGET_SIZE[1], num_classes)

    model = keras.models.Sequential([
            keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(128, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(num_classes, activation='softmax')
        ])

    label_smoothing = 0.1  # Adjust this value to control the level of label smoothing
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    epochs = epochs
    model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=len(val_generator))

    score = model.evaluate(test_generator, verbose=0)

    print("Test set loss: {:.4f}".format(score[0]))
    print("Test set accuracy: {:.4f}".format(score[1]))

    model.save(f"{weights_path}.h5")
    print(f"Model is saved to {weights_path}.h5!")



def make_prediction(weights_path: str, img: str):
    # REVIEW: this is okay, but why? You already use a keras loader in your code
    img_new = load_img(img, target_size=IMG_SIZE)
    # img_new = Image.open(img).convert("L")
    # img_new = img_new.resize((50, 50))
    # img_new = np.array(img_new)
    # img_new = np.array(img_new).reshape(-1, 50, 50, 1)
    # img_new = img_new / 255.0

    model = keras.models.load_model(weights_path)
    # REVIEW: another trick: note how x[None] adds an new axis to a numpy array x
    predictions = model.predict(np.array(img_new)[None], verbose=0)
    # results = predictions[0]
    # label_names = ['1d', 'dmtx', 'other']
    max_prob_idx = np.argmax(predictions)
    # REVIEW: now you see why bidict works well here
    predicted_label = LABEL_MAP.inv[max_prob_idx]

    # REVIEW: it is conventional to return tuples (immutable) rather than lists whenever possible
    return predicted_label, predictions[0]

def validate_handler(
        data_dir: str,
        weights_path: str,
        split_csv: str,
        subset: str,
        output_path: Optional[str] = None,
        verbose: bool = False
    ):
    df = create_updated_csv(split_csv, data_dir)
    df = df[df['subset'] == subset]

    model = tf.keras.models.load_model(weights_path)

    # REVIEW: this is NOT the way people usully write this, I just got bored
    # however this code functions not less efficient than the original one.
    # Note that map is lazy, so no computations are done here yet!
    def prob_to_class(prediction: np.ndarray) -> str:
        # note that if all probabilities are below 0.5,
        # argmax() returns the first entry (0)
        # which corresponds to 'other'
        return LABEL_MAP.inv[(prediction[0] > 0.5).argmax()]
    prediction = map(
        compose(
            prob_to_class,
            partial(model.predict, verbose=0),
            itemgetter(None),
            np.array,
            partial(load_img, target_size=IMG_SIZE),
            itemgetter('path'),
            itemgetter(1)
        ),
        df.iterrows()
    )

    # REVIEW: and now the actual computations are performed
    df['prediction'] = [*tqdm(prediction, total=len(df.index), desc="Predicting")]

    # calculate metrics
    print(classification_report(
        df['class'], df['prediction']
    ))

    os.makedirs(output_path, exist_ok=True)
    if verbose:
        df[df['class'] != df['prediction']].apply(
            lambda row:
                load_img(row['path'], target_size=IMG_SIZE).save(
                    os.path.join(
                        output_path,
                        f"{row['image']}_gt{row['class']}_{row['prediction']}.jpg"
                    )
                ), axis = 1
        )

    output_path = os.path.join(output_path, f'predictions_{subset}.csv')
    df.to_csv(output_path, index = None)
