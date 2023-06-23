import os
from typing import Optional as O

import fire
import numpy as np
import utils

from PIL import Image, ImageOps

DEFAULT_SPLIT_PATH = os.path.join(os.path.dirname(__file__), 'default_split.csv')
DEFAULT_WEIGHTS = os.path.join(os.path.dirname(__file__), 'classifier_weights.h5')

class Classifier:

    # TODO: weights_path
    def train(self, *,
              data_dir: str,
              weights_path: O[str] = DEFAULT_WEIGHTS,
              split_csv: str = DEFAULT_SPLIT_PATH,
              epochs: int = 14,
              batch_size: int = 20
    ):
        # REVIEW: there is no sense in moving all the methods to another file
        # and leaving the alises here. You could siply implement all the required code
        # here. This is not JAVA :) Writing implementations without interfaces and
        # even declaring multiple classes per file is prefectly legal in the
        # world of snake-whisperers.



        df = utils.create_updated_csv(split_csv, data_dir)
        utils.start_training(data_dir, df, weights_path, epochs, batch_size)


    def validate(self,
                 subset: str,
                 *,
                 data_dir: str,
                 weights_path: str = DEFAULT_WEIGHTS,
                 split_csv: str = DEFAULT_SPLIT_PATH,
                 output_path: O[str] = "out",
                 verbose: bool = False
                 ):
        utils.validate_handler(data_dir, weights_path,
                               split_csv, subset, output_path, verbose)

    def predict(self, img: str, *, weights_path: str = DEFAULT_WEIGHTS):
        # REVIEW: use unpacking!
        label, probabilities = utils.make_prediction(weights_path, img)
        probabilities = np.round(probabilities, 2) # just for pretty-printing
        # REVIEW: did you know about this feature of f-strings?
        # f"{var=}" actually prints var=<value of var>
        # while !s indicates that we want to use __str__ rather than
        # __repr__, which looks a bit nices in this case.
        print(f"{label=!s}, {probabilities=!s}")


if __name__ == '__main__':
    fire.Fire(Classifier)
