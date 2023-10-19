# -----------------------------------------------------------------------------
# DataAugmentation Class
# Author: Xavier Beltran Urbano
# Date Created: 17-10-2023
# -----------------------------------------------------------------------------


# Import necessary libraries
import numpy as np
import cv2 as cv
from keras.preprocessing.image import ImageDataGenerator
import random
import matplotlib.pyplot as plt


# Features Class
class DataAugmentation:
    # Constructor for the Features class
    def __init__(self):
        # Create an instance of ImageDataGenerator with your desired augmentation settings
        self.datagen = ImageDataGenerator(
            rotation_range=40,
            shear_range=0.2,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )

    def apply(self, vec_img, vec_gt, desired_number_images):
        images_to_augment = desired_number_images - len(vec_img)
        # Combine the two vectors
        combined = list(zip(vec_img, vec_gt))
        # Shuffle the combined list
        random.shuffle(combined)
        # Unpack the shuffled elements back into separate vectors
        vec_img, vec_gt = zip(*combined)
        vec_img = list(vec_img)
        vec_gt = list(vec_gt)

        final_vec=[]
        final_gt_vec = []
        augment_number=(desired_number_images-len(vec_img))
        i = 0
        while i < augment_number:
            # Augment the original image and add it to the list
            value=random.randint(0,len(vec_img)-1)
            img_augmented = next(self.datagen.flow(np.expand_dims(vec_img[value], axis=0), batch_size=1))
            final_vec.append(img_augmented[0].astype(np.uint8))
            final_gt_vec.append(vec_gt[value])

            i += 1
        return np.concatenate([vec_img,final_vec], axis=0), np.concatenate([vec_gt,final_gt_vec])



# Entry point of the script
if __name__ == "__main__":
    # Load an image for feature extraction
    image_folder = "/Users/xavibeltranurbano/Desktop/MAIA/GIRONA/CAD/MACHINE LEARNING/BINARY/train/nevus/nev00002.jpg"
    img = cv.imread(image_folder)
    img=cv.cvtColor(img, cv.COLOR_BGR2RGB)
    vec_img=[]
    vec_img.append(img)
    vec_img.append(img)
    vec_img.append(img)
    vec_gt=[0,0,1,1]
    # Create an instance of the Features class
    data_aug = DataAugmentation()

    # Extract and print color features
    aug_vec_img,aug_vec_gt = data_aug.apply(vec_img,vec_gt,desired_number_images=20)

    print("PLOT IMAGES")
    for i in range(aug_vec_img.shape[0]):
        print(aug_vec_img[i].shape)
        plt.imshow(aug_vec_img[i],cmap='gray')
        plt.show()
