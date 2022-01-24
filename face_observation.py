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
    labels = label_file.readlines()
    # next get rid of first two rows, and then 1st item in list
    return

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