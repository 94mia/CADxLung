import keras
from keras import models
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import shutil

def cnn_eliminate(model_name, eli_path, train_path):
    prefix = "/home/zlstg1/cding0622/scripts/models"
    model_path = os.path.join(prefix, model_name)
    model = models.load_model(model_path)

    train_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(48, 48),
        batch_size=1,
        shuffle=False,
        color_mode='grayscale',
        class_mode='binary')
    p_acc = model.evaluate_generator(train_generator, train_generator.samples)
    print("Accuracy:  ", p_acc[1])
    
    pred = model.predict_generator(train_generator, train_generator.samples, verbose=1)
    print("\n")
    elim_list = []
    for i in range(train_generator.samples):
        file = train_generator.filenames[i]
        if "benign" in file and pred[i] > 0.5:
            elim_list.append(file)
        elif "malignant" in file and pred[i] <= 0.5:
            elim_list.append(file)
    print("%d files over threshold" % len(elim_list)) 
    for f in elim_list:
        #f = f.split("/", 1)[1] 
        #print(f)
        tmp = os.path.join(eli_path, f)
        #os.remove(tmp)
        shutil.move(tmp, "/home/zlstg1/cding0622/project/eli_visualize")

if __name__ == "__main__":
    cnn_eliminate("model_full.h5", "/home/zlstg1/cding0622/project/cnn_eli/full_data/train", "/home/zlstg1/cding0622/project/ori/full_data/train")

