import fire
import utils


class Segment:

    # TODO: weights_path
    def train(self, *, data_dir: str, split_csv: str, labels_dir: str,
              model: str, epochs: int = 50, yaml_path: str = 'data.yaml', batch_size: int = 8,
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
    def validate(self, *, weights_path: str, data_dir: str = None, split_csv: str = None, output_path: str = None):
        if utils.check_splitup():
            utils.start_validation(weights_path, output_path)
        else:
            utils.create_csv(data_dir, split_csv)
            utils.create_splitup(data_dir, 'newnew.csv')

    # Predict
    def predict(self, weights_path: str, img: str, output_path: str):
        utils.run_segmentation(weights_path, img, output_path)


if __name__ == '__main__':
    fire.Fire(Segment)
