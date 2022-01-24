#import tensorflow.keras.models as models
#import tensorflow.keras.layers as layers
#import tensorflow.keras.losses as losses
# import tensorflow.keras.optimizer as optimizers
#import tf.keras.utils as utils
from tensorflow.keras import utils

def load_labels(labels_filename = str):
    """
    # reads text from file based on passed in labels filename and creates a list of lists of each image's labels
    # each label is in the form of a one-hot encoded vector with 40 attributes
   """
    label_file = open(labels_filename, 'r')
    labels = list(label_file.readlines()) # list of the labels of each image
    processed_labels = []
    for i in range(2, len(labels)):
        img_labels = labels[i] # labels of the indexed image/row
        img_labels = img_labels.split() #turns labels string into list
        processed_labels.append(img_labels)
    
    print(len(processed_labels))
    return processed_labels

labels = load_labels('anno/list_attr_celeba.txt')

"""
train = utils.image_dataset_from_directory(
    'img_align_celeba',
    labels = labels,
    label_mode = 'categorical',
    image_size=(218, 178),
    shuffle = True,
    seed = 0,
    validation_split = 0.3,
    subset = 'training',
)

val = utils.image_dataset_from_directory(
    'img_align_celeba',
    labels = labels,
    label_mode = 'categorical',
    image_size=(250, 250),
    shuffle = True,
    seed = 0,
    validation_split = 0.3,
    subset = 'validation',
)
"""


