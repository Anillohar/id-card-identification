import os
import sys
import cv2
import pickle as p

import matplotlib.pyplot as plt
from scipy.spatial import distance
from keras.models import model_from_json

from utils import INPUT_SHAPE, MODEL_WEIGHT_PATH, INPUT_IMAGE_DATA_DIR_PATH, DATA_DIR_PATH


json_file = open(MODEL_WEIGHT_PATH+'model.json', 'r')
model_json = json_file.read()
json_file.close()

# building model
model = model_from_json(model_json)

# load weights into new model
model.load_weights(MODEL_WEIGHT_PATH+"model.h5")
print("Loaded model from disk")


def imread(path):
    image = cv2.imread(path)
    resized_image = cv2.resize(image, (INPUT_SHAPE[0], INPUT_SHAPE[1]))
    return resized_image


class PrepareImagesForSimilarityAnalysis():
	
	def __init__(self):
		self.X = []
		self.model_output_X = []
		self.index_to_category = {}

		self.create_image_categories(INPUT_IMAGE_DATA_DIR_PATH)
		self.map_model_output_for_category_images()
		self.save_output_mappings()

	def create_image_categories(self, path):
	    i=0
	    for category in os.listdir(path):
	        print("loading category: " + category)
	        print("index starting from: "+ str(i))
	        categorypath = path + str(category)
	        for filename in os.listdir(categorypath):
	            image_path = categorypath + "/" + filename
	            image = imread(image_path)
	            self.X.append(image)
	            index_to_category[i] = category
	            i=i+1
	            if i%20==0:
	                break

	def map_model_output_for_category_images(self):
		model_op = []
		for image in self.X:
		    img=img/255.0
		    img=img.reshape(1,INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2])
		    model_prediction=model.predict(img)
		    model_prediction=model_prediction.reshape(4096)
		    model_op.append(model_prediction)
		self.model_output_X = np.asarray(model_op)

	def save_output_mappings(self):
		output_mappings = [len(self.X), self.model_output_X]
		p.dump(output_mappings, open(DATA_DIR_PATH+'output_mappings.p', 'wb'))


if 'output_mappings.p' not in os.listdir(DATA_DIR_PATH):
	PrepareImagesForSimilarityAnalysis()

length_X, model_output_X = p.load(open(DATA_DIR_PATH+'output_mappings.p', 'rb'))

for image_name in sys.argv:
	test_img = imread(image_name)
	test_img=test_img/255.0
	image=test_img.reshape(1,INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2])
	model_prediction=model
	.predict(image)
	final = []
	for i in range(length_X):
	    dist = distance.euclidean(model_output_X[i], model_prediction)
	    final.append([dist,i])
	final.sort()
	print('image: ', image_name, 'belongs to category: 'index_to_category[final[0][1]])
	# plt.imshow(test_img)
