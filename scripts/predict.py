import tensorflow as tf
import keras
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Convolution2D, MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from sklearn.cross_validation import StratifiedKFold
#from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from keras.layers.core import Reshape
import numpy as np

def predict(model_in):
	TEST_DIR = "/home/zlstg1/cding0622/varfull_class_data/"

	test_datagen = ImageDataGenerator(rescale=1./255)

	test_generator = test_datagen.flow_from_directory(
		TEST_DIR,
		target_size = (48,48),
		batch_size = 10,
		color_mode = "grayscale",
		shuffle = True,
		class_mode = "binary")
	
	model = keras.models.load_model(model_in)
	scoreSeg = model.evaluate_generator(test_generator,test_generator.nb_sample)
	#print("scoreSeg: ", scoreSeg)
	print("model: ", model_in)
	print("Accuracy = ",scoreSeg[1])

	predict = model.predict_generator(test_generator, test_generator.nb_sample)
	print("predict", predict)

if __name__ == "__main__":
	predict("model_var2.5.h5")
	predict("model_var2.8.h5")
	predict("model_var3.0.h5")
	predict("model_var3.3.h5")
	predict("model_var3.8.h5")
	predict("model_full.h5")
