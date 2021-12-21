import os
import cv2
import pickle as p
import numpy as np

training_data_path = os.getcwd() + '/data/training/'

aadhar_data_path = training_data_path + 'aadhar_cards/'
pan_data_path = training_data_path + 'pan_cards/'
driving_license_data_path = training_data_path + 'driving_license/'

aadhar_cards = os.listdir(aadhar_data_path)
pan_cards = os.listdir(pan_data_path)
driving_licenses = os.listdir(driving_license_data_path)

path_files_mapping = {aadhar_data_path: aadhar_cards,
                      pan_data_path: pan_cards,
                      driving_license_data_path: driving_licenses}

path_files = [aadhar_cards, pan_cards, driving_licenses]
all_cards_data = dict()


def get_or_create_triplet_pairs():
    if not os.path.isfile(training_data_path + 'triplet_pairs.p'):
        prepare_data()
    return p.load(open(training_data_path + 'triplet_pairs.p', 'rb'))


def read_files_from_path(path, files, image_size=(105, 105)):
    for file_name in files:
        image = cv2.imread(path + file_name)
        resized_image = cv2.resize(image, image_size)
        image = resized_image.astype(np.float32)
        image /= 255.
        all_cards_data[file_name] = image


def prepare_data():
    for path in path_files_mapping.keys():
        files = path_files_mapping[path]
        read_files_from_path(path, files)
    # p.dump(all_cards_data, open('all_cards_data.p', 'wb'))

    triplet_pairs = []
    # triplet_fake_outputs = [np.empty((1, 12288))]

    for path in path_files_mapping.keys():
        for idx, anchor in enumerate(path_files_mapping[path]):
            if idx + 1 < len(path_files_mapping[path]):
                for positive in path_files_mapping[path][idx + 1:]:
                    for path_1 in path_files_mapping.keys():
                        if path != path_1:
                            for idx_1, negative in enumerate(path_files_mapping[path_1]):
                                triplet_pairs.append(
                                    [all_cards_data[anchor],
                                     all_cards_data[positive],
                                     all_cards_data[negative]])
    print(len(triplet_pairs))
    # triplet_fake_outputs = triplet_pairs*len(triplet_pairs)
    triplet_pairs = np.array(triplet_pairs)
    p.dump(triplet_pairs, open(training_data_path + 'triplet_pairs.p', 'wb'))
    # p.dump(triplet_fake_outputs, open('triplet_fakie_outputs.p', 'wb'))
