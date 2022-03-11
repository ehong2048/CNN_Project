# Already made a copy of celeba_images folder (with smiling and not_smiling images sorted into separate sub-folders) 
# into another folder titled celeba_images_bbox
# Purpose: to go through images in celeb_images_bbox and crop them into just the bounding box

# current ERROR: I think the bounding box data corresponds to the original celeba data, not the aligned data...
# so might need to redo or change to the landmarks data

import os
import cv2

bbox_file = 'anno/list_bbox_celeba.txt'

bbox_data = open(bbox_file, 'r')
bbox_data = list(bbox_data.readlines()) # bbox values for each image
bbox_vals = [] # initiates list of lists for the bbox values of each image
img_filenames = [] # initiates list of image filenames

for i in range(2, 10002): #skip first two lines 
    img_bbox = bbox_data[i] # saves string for indexed image/row
    img_bbox = img_bbox.split() #turns bbox labels string into list

    img_filenames.append(img_bbox[0])
    bbox_vals.append(img_bbox[1:])

print(img_filenames[:10])
print(bbox_vals[:10])


for root, dirs, files in os.walk("celeba_images_bbox"):
    for filename in files[0:5]:
        f = os.path.join(root, filename)
        if os.path.isfile(f):
            print(f)
        img = cv2.imread(f)
        cv2.imshow("Original", img)
        
        for i in range(len(img_filenames)):
            if img_filenames[i] == filename:
                bbox = bbox_vals[i]
                print(bbox)
                x1 = int(bbox[0])
                y1 = int(bbox[1])
                width = int(bbox[2])
                height = int(bbox[3])
        
        img = img[x1:x1+width, y1:y1+height]

        cv2.imshow(f'Cropped by bbox', img)
        cv2.waitKey(0)


        

print("Done!")
        