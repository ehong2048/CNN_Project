#import tensorflow.keras.models as models
#import tensorflow.keras.layers as layers
#import tensorflow.keras.losses as losses
# import tensorflow.keras.optimizer as optimizers
#import tf.keras.utils as utils
from tensorflow.keras import utils

def load_labels(labels_filename = str):
    # reads text from file based on passed in labels filename and creates a list of lists of each image's labels
    # each label is in the form of a one-hot encoded vector with 40 attributes
    label_file = open(labels_filename, 'r')
    labels = list(label_file.readlines()) # list of the labels of each image
    for i in range(2, len(labels)):
        img_labels = labels[i] # labels of indexed image/row
        print(img_labels)

    # next get rid of first two rows, and then 1st item in list
    return

"""
train = utils.image_dataset_from_directory(
    'img_align_celeba',
    label_mode = 'categorical',
    image_size=(218, 178),
    shuffle = True,
    seed = 0,
    validation_split = 0.3,
    subset = 'training',
)

val = utils.image_dataset_from_directory(
    'img_align_celeba',
    label_mode = 'categorical',
    image_size=(250, 250),
    shuffle = True,
    seed = 0,
    validation_split = 0.3,
    subset = 'validation',
)
"""

load_labels('anno/list_attr_celeba.txt')