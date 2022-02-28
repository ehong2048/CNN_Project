# Purpose: Walk through directory and separate the images into two subdirectories
# One for smiling images and one for not smiling

import subprocess

folder_fname = 'anno/list_attr_celeba.txt'

data = open(folder_fname, 'r')
data = list(data.readlines()) # list of the labels for each image
labels = [] # initiates list of processed labels for the smiling trait of each image
img_filenames = [] # initiates list of image filenames

for i in range(2, 10002): 
    img_data = data[i] # saves string for indexed image/row
    img_data = img_data.split() #turns labels string into list

    img_filenames.append(img_data[0])
    if img_data[32] == '1':
        labels.append('smiling')
    else:
        labels.append('not_smiling')

print(img_filenames[:10])
print(labels[:10])

for label in set(labels):
    subprocess.run(['mkdir', label])
    subprocess.run(['mkdir', label])

for i, filename in enumerate(img_filenames):
    if 'img_align_celeba/' + filename not in labels[i] + "/":
        subprocess.run(['mv', 'img_align_celeba/' + filename, labels[i] + "/"])

print("---Finished sorting data into smiling and non_smiling subdirectories!---")
