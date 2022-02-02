import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow.keras.losses as losses
import tensorflow.keras.optimizers as optimizers
from tensorflow.keras import utils

# FOR NEXT TIME: Maybe just change to one attribute so that the dataset doesn't overlap (i.e. don't do multilabel classification)
# Separate script to sort into folders for smiling and not smiling based on the text file


def load_labels(labels_filename = str):
    """
    Purpose: reads text from file based on passed in labels filename and creates a list of lists of each image's labels
    (each label is in the form of a one-hot encoded vector in string form with 40 attributes)
   """
    label_file = open(labels_filename, 'r')
    labels = list(label_file.readlines()) # list of the labels for each image
    processed_labels = [] # initiates list of processed labels for the smiling trait of each image
    #for i in range(2, len(labels)):
    for i in range(2, 5002): 
        img_labels = labels[i] # saves string for indexed image/row
        img_labels = img_labels.split() #turns labels string into list
        processed_labels.append(img_labels[32])
    
    print(f'Number of Images: {len(processed_labels)}')
    print(type(processed_labels))
    return processed_labels

labels = load_labels('anno/list_attr_celeba.txt')
print(labels[0:20])

print("--Retreive training data--")
train = utils.image_dataset_from_directory(
    'celeba_images',
    label_mode = 'categorical',
    image_size=(218, 178),
    shuffle = True,
    seed = 0,
    validation_split = 0.3,
    subset = 'training',
)
print(len(train))
print(type(train))

print("--Retreive val data--")
val = utils.image_dataset_from_directory(
    'celeba_images',
    label_mode = 'categorical',
    image_size=(218, 178),
    shuffle = True,
    seed = 0,
    validation_split = 0.3,
    subset = 'validation',
)


class Net():
    def __init__(self, image_size):
        self.model = models.Sequential() # layers are in sequence
        self.model.add(layers.Conv2D(8, 13, strides = 3, input_shape = image_size, activation = 'relu')) # (output depth, frame size, kwargs)
        self.model.add(layers.MaxPool2D(pool_size = 2)) # (frame size, kwargs, strides equals frame size as default
        self.model.add(layers.Conv2D(16, 3))
        self.model.add(layers.MaxPool2D(pool_size = 2))
        # insert other layers
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(1024, activation = 'relu'))
        self.model.add(layers.Dense(256, activation = 'relu'))
        self.model.add(layers.Dense(60, activation = 'relu'))
        self.model.add(layers.Dense(2, activation = 'softmax'))
        
        self.loss = losses.MeanSquaredError()
        self.optimizer = optimizers.Adam(learning_rate = 0.001)

        self.model.compile(
            loss = self.loss,
            optimizer = self.optimizer,
            metrics = ['accuracy']
        )

    def __str__(self):
        self.model.summary()
        return ""

net = Net((218, 178, 3))
print(net)

net.model.fit(
    train,
    batch_size = 32,
    epochs = 200,
    verbose = 2,
    validation_data = val,
    validation_batch_size = 32
)