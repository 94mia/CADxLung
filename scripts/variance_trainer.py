import tensorflow as tf
import keras
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from sklearn.cross_validation import StratifiedKFold
#from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from keras.layers.core import Reshape
import numpy as np
from keras import regularizers
from keras import optimizers

def v_model(folder):
    TRAIN_DIR = "/home/zlstg1/cding0622/project/manual/%s/train/" % folder 
    VALIDATE_DIR = "/home/zlstg1/cding0622/project/manual/%s/valid/" % folder

    #keras.optimizers.RMSprop(lr=0.005, rho=0.9, epsilon=1e-08, decay=0.0) # default: 0.001 
    #keras.optimizers.SGD(lr=0.05, momentum=0.0, decay=0.0, nesterov=False)

    reg = regularizers.l2(0.001)
    model = Sequential()
    model.add(Conv2D(32,  (5, 5), input_shape=(48, 48, 1), kernel_regularizer=reg))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(32, (4, 4), kernel_regularizer=reg))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(50))
    model.add(Activation('relu'))

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
    test_datagen = ImageDataGenerator(rescale=1./255)

    # this is a generator that will read pictures found in
    # subfolers of 'data/train', and indefinitely generate
    # batches of augmented image data
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,  # this is the target directory
        target_size=(48, 48),
        batch_size=64,
        shuffle=True,
        color_mode='grayscale',
        class_mode='binary') 
    # this is a similar generator, for validation data
    validation_generator = test_datagen.flow_from_directory(
        VALIDATE_DIR,
        target_size=(48, 48),
        batch_size=64,
        shuffle=True,
        color_mode='grayscale',
        class_mode='binary')

    weight = model.fit_generator(
            train_generator,
            verbose = 2,
            steps_per_epoch=1280,
            epochs=20,
            validation_data=validation_generator,
            validation_steps=800)

    if folder.split("/")[0] == "aug":
        folder = "aug-" + folder.split("/")[1]
    model.save("models/model_var_{0}.h5".format(folder))

if __name__ == "__main__":
	v_model("data")	 
