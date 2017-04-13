import numpy as np
np.random.seed(2418)
import tensorflow as tf
tf.set_random_seed(2418)
import keras
from keras import optimizers
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.core import Reshape
from keras import regularizers
from evaluate import evaluate_model

#keras.optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08, decay=0.0) # default: 0.001 
#keras.optimizers.SGD(lr=0.05, momentum=0.0, decay=0.0, nesterov=False)

def fit_model(folder, TRAIN_DIR, VALIDATE_DIR):
    #reg = regularizers.l2(0.001)
    model = Sequential()
    model.add(Conv2D(32,  (5, 5), input_shape=(48, 48, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(32, (4, 4)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(50))
    model.add(Activation('relu'))

    #model.add(Dense(50))
    #model.add(Activation('relu'))

    model.add(Dense(1))
    model.add(Activation("sigmoid"))

    model.compile(loss='binary_crossentropy',
                  optimizer= 'rmsprop',
                  metrics=['accuracy'])

    train_datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        rescale=1./255,
       	)

    # this is the augmentation configuration we will use for testing:
    # only rescaling
    valid_datagen = ImageDataGenerator(rescale=1./255)

    # this is a generator that will read pictures found in
    # subfolers of 'data/train', and indefinitely generate
    # batches of augmented image data
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,  # this is the target directory
        target_size=(48, 48),
        batch_size=64,
        shuffle=True,
        color_mode='grayscale',
        class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

    # this is a similar generator, for validation data
    validation_generator = valid_datagen.flow_from_directory(
        VALIDATE_DIR,
        target_size=(48, 48),
        batch_size=64,
        shuffle=True,
        color_mode='grayscale',
        class_mode='binary')

    weight = model.fit_generator(
            train_generator,
            verbose = 2,
            steps_per_epoch=640,
            epochs=20,
            validation_data=validation_generator,
            validation_steps=400)

    if folder.split("/")[0] == "ori":
        folder = "ori-" + folder.split("/")[1]
    if folder.split("/")[0] == "aug":
        folder = "aug-" + folder.split("/")[1]
    if folder.split("/")[0] == "cnn_eli":
        folder = "cnneli-" + folder.split("/")[1]
    if folder.split("/")[0] == "gauss_data":
        folder = "gauss-" + folder.split("/")[1]
    model.save("models/model_{0}.h5".format(folder))

    return "model_{0}.h5".format(folder)


if __name__ == "__main__":
    #folder_list = ["ori/full", "ori/var1.0", "ori/var2.0", "ori/var2.5", "ori/var3.0", "ori/var3.3", "ori/var3.8", "cnn_eli/full", "eli"]
    folder_list = ["aug/full", "aug/var1.0", "aug/var2.0", "aug/var2.5", "aug/var3.0",  "aug/var3.3"]
    #folder_list = ["gauss_data/5full", "gauss_data/6full", "gauss_data/8full", "gauss_data/15full"]
    #folder_list = ["cnn_eli/full"]
    for folder in folder_list:
        print("\n\n######## Training folder: ", folder, "########\n\n")
        TRAIN_DIR = "/home/zlstg1/cding0622/project/%s_data/train/" % folder
        VALIDATE_DIR = "/home/zlstg1/cding0622/project/%s_data/valid/" % folder
        TEST_DIR = "/home/zlstg1/cding0622/project/%s_data/test/" % folder
        FULL_DIR = "/home/zlstg1/cding0622/project/all/data/"
        if "aug" in folder:
            FULL_DIR = "/home/zlstg1/cding0622/project/all/aug_data/"
        model_name = fit_model(folder, TRAIN_DIR, VALIDATE_DIR)
        evaluate_model(model_name, TEST_DIR, FULL_DIR)
        
