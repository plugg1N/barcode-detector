import fire
from enum import Enum
import utils
from typing import Optional


class Subdataset(Enum):
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'


class Detect:
    # Train
    def train(self, *, data_dir: str, split_csv: str,
              model: str, labels_dir: str, epochs: int = 50, yaml_path: str = 'data.yaml', batch_size: int = 8,
              weights_path: str, **train_params):
        

        print("Starting augmentation")
        utils.start_augmentation(data_dir, labels_dir)
        print("Augmentation is done")
        utils.create_csv(data_dir, split_csv)
        print("CSV is created")
        utils.create_augmented_csv('newnew.csv')
        print("Augmented CSV is created")
        utils.create_splitup('Dataset_new', 'modified.csv')
        print("Split is created")
        utils.create_yaml_division('Dataset_new')
        print("YAML is created")
        utils.start_training(model, yaml_path, int(
            epochs), batch_size, weights_path)

    # Validate
    def validate(self, *,
                 data_dir: str, model: str, split_csv: str,
                 output_path: Optional[str] = None):

        if utils.check_splitup():
            print("Splitup is created, starting validation...")
            print(
                f"\n\nAVG IoU: {utils.start_validation(model, data_dir, output_path)}")
        else:
            print("Splitup is not created, starting splitup...")
            utils.create_csv(data_dir, split_csv)
            utils.create_splitup(data_dir, 'newnew.csv')
            utils.create_yaml_division(data_dir)
            print(
                f"\n\nAVG IoU: {utils.start_validation(model, data_dir, output_path)}")

    # Predict
    def predict(self, weights_path: str, img: str, output_path: str):
        utils.run_prediction(weights_path, img, output_path)


if __name__ == '__main__':
    fire.Fire(Detect)
