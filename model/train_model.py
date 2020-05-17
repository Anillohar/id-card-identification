import numpy as np 
from keras.models import Model
from keras.layers import Input

from utils import INPUT_SHAPE, MODEL_WEIGHT_PATH
from siamese_network import SiameseNetwork

class TrainModel():

    def __init__(self):
        self.siamese_network = SiameseNetwork()
        self.input_shape = self.INPUT_SHAPE
        self.triplet_model = self.siamese_network.model

        self.train_siamese_network()
        self.model = self.trained_model()
        self.save_model_weights()
        
    def train_siamese_network(self, batchsize=32, no_of_epochs=20):
        triplet_pairs = self.siamese_network.get_data()
        call_backs = siamese_network.get_callback()
        self.triplet_model.fit(x=[triplet_pairs[:, 0, :], triplet_pairs[:, 1, :], triplet_pairs[:, 2, :]], 
                          y=np.zeros(shape=(triplet_pairs.shape[0], 12288)), batch_size=batchsize, 
                          callbacks=call_backs, epochs=no_of_epochs, shuffle=True, validation_split=0.14)
    
    def trained_model(self):
        triplet_model_required_layers = self.triplet_model.layers[-2]
        anchor_input = Input(self.input_shape, name='anchor_input')
        encoded_anchor = triplet_model_required_layers(anchor_input)
        return Model(anchor_input, encoded_anchor)

    def save_model_weights(self):
        # serialize model to JSON
        model_json = self.model.to_json()
        with open(MODEL_WEIGHT_PATH+"model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights(MODEL_WEIGHT_PATH+"model.h5")
        print("Saved model to disk")

if __name__=='__main__':
    TrainModel()