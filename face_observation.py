from xml.dom import ValidationErr
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow.keras.losses as losses
import tensorflow.keras.optimizers as optimizers
from tensorflow.keras import utils

def load_labels(labels_filename = str):
    """
    Purpose: reads text from file based on passed in labels filename and creates a list of lists of each image's labels
    (each label is in the form of a one-hot encoded vector in string form with 40 attributes)
   """
    label_file = open(labels_filename, 'r')
    labels = list(label_file.readlines()) # list of the labels for each image
    processed_labels = [] # initiates list of processed labels for each image
    for i in range(2, len(labels)):
        img_labels = labels[i] # labels of the indexed image/row
        img_labels = img_labels.split() #turns labels string into list
        processed_labels.append(img_labels[1:])
    
    print(len(processed_labels))
    return processed_labels

labels = load_labels('anno/list_attr_celeba.txt')

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

class Net():
    def __init__(self, image_size):
        self.model = models.Sequential() # layers are in sequence
        # insert other layers
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(40, activation = 'softmax'))
        
        self.loss = losses.MeanSquaredError()
        self.optimizers(optimizers.Adam(learning_rate = 0.001))

        self.model.compile(
            loss = self.loss,
            optimizer = self.optimizer,
            metrics = ['accuracy']
        )

    def __str__(self):
        self.model.summary()
        return

net = Net((218, 178, 3))
print(net)

net.fit(
    train,
    batch_size = 32,
    epochs = 200,
    verbose = 2,
    validation_data = val,
    validation_batch_size = 32
)
