import numpy as np 
from keras.models import Model
from keras.layers import Input

from utils import INPUT_SHAPE, MODEL_WEIGHT_PATH, MODEL_NAME
from siamese_network import SiameseNetwork


class TrainModel:

    def __init__(self):
        self.input_shape = INPUT_SHAPE

        # Setup Siamese network for training purpose
        self.siamese_network = SiameseNetwork()
        self.triplet_model = self.siamese_network.model

        # Train the model and save weights
        self.train_siamese_network()
        self.model = self.trained_model()
        self.save_model_weights()
        
    def train_siamese_network(self, batch_size=32, no_of_epochs=20):
        triplet_pairs = self.siamese_network.get_data()
        call_backs = self.siamese_network.get_callback()
        self.triplet_model.fit(
            x=[
                triplet_pairs[:, 0, :],
                triplet_pairs[:, 1, :],
                triplet_pairs[:, 2, :]
            ],
            y=np.zeros(shape=(triplet_pairs.shape[0], 12288)),
            batch_size=batch_size,
            callbacks=call_backs,
            epochs=no_of_epochs,
            validation_split=0.14,
            shuffle=True
        )
    
    def trained_model(self):
        triplet_model_required_layers = self.triplet_model.layers[-2]
        anchor_input = Input(self.input_shape, name='anchor_input')
        encoded_anchor = triplet_model_required_layers(anchor_input)
        return Model(anchor_input, encoded_anchor)

    def save_model_weights(self):
        # serialize model to JSON
        model_json = self.model.to_json()
        with open(MODEL_WEIGHT_PATH + MODEL_NAME + ".json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights(MODEL_WEIGHT_PATH + MODEL_NAME + ".h5")
        print("Saved model to disk")


if __name__ == '__main__':
    TrainModel()
