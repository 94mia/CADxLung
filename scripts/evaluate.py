import keras
from keras import models
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os

def evaluate_model(model_name, test_path, full_path):
    prefix = "/home/zlstg1/cding0622/scripts/models"
    model_path = os.path.join(prefix, model_name)
    model = models.load_model(model_path)

    print("\n#### Running model on testing set ####\n")
    #predict on testing set
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=(48, 48),
        batch_size=1,
        shuffle=False,
        color_mode='grayscale',
        class_mode='binary')
    p_acc = model.evaluate_generator(test_generator, test_generator.samples)
    print("Accuracy:  ", p_acc[1])
    
    """
    # for elimination!
    count = 0
    for i in range(test_generator.samples):
        file = test_generator.filenames[i]
        if "benign" in file and pred[i] < 0.5:
            count += 1
        elif "malignant" in file and pred[i] >= 0.5:
            count += 1
    """

    print("\n\n#### Running model on full dataset ####\n")

    full_datagen = ImageDataGenerator(rescale=1./255)
    full_generator = full_datagen.flow_from_directory(
        full_path,
        target_size=(48, 48),
        batch_size=1,
        shuffle=False,
        color_mode='grayscale',
        class_mode='binary')

    f_acc = model.evaluate_generator(full_generator, full_generator.samples)
    print("Accuracy:  ", f_acc[1], "\n")


if __name__ == "__main__":
    folder = "full"
    TEST_DIR = "/home/zlstg1/cding0622/project/%s_data/test/" % folder
    # either data or aug_data folder for full dataset
    FULL_DIR = "/home/zlstg1/cding0622/project/all/data/"
    evaluate_model("model_full.h5", TEST_DIR, FULL_DIR)
    #variance_eliminate("model_var_data.h5", "/home/zlstg1/cding0622/project/manual/data/train")


