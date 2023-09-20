# -----------------------------------------------------------------------------
# DataGenerator Class
# Author: Xavier Beltran Urbano
# Date Created: 2023-09-20
# -----------------------------------------------------------------------------

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import Sequence

class DataGenerator(Sequence):
    """
    Custom data generator for training deep learning models on image segmentation tasks.

    Attributes:
        image_directory (str): Directory containing input images.
        mask_directory (str): Directory containing corresponding segmentation masks.
        list_IDS (list): List of IDs (file names without extensions) for the data samples to include.
        batch_size (int): Size of each batch.
        target_size (tuple): Target size for image and mask resizing.
        data_augmentation (bool): Flag indicating whether to apply data augmentation.
        shuffle (bool): Flag indicating whether to shuffle the data.
    """

    def __init__(self, image_directory, mask_directory, list_IDS, batch_size, target_size, data_augmentation, shuffle=True):
        self.image_directory = image_directory
        self.mask_directory = mask_directory
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.data_augmentation = data_augmentation

        # Get the list of image and mask file names based on the provided IDs
        self.image_filenames = sorted([filename for filename in os.listdir(image_directory) if filename.endswith('.jpg') and filename.split('.')[0] in list_IDS])
        self.mask_filenames = sorted([filename for filename in os.listdir(mask_directory) if filename.replace('_segmentation.png', '') in list_IDS])

        # Calculate the number of batches
        self.num_batches = len(self.image_filenames) // self.batch_size

        # Initialize indices for data shuffling
        self.indices = np.arange(len(self.image_filenames))

        # Shuffle the indices if required
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        """
        Get the number of batches per epoch.
        """
        return self.num_batches

    def __getitem__(self, index):
        """
        Generate a batch of data.

        Args:
            index (int): Index of the batch.

        Returns:
            batch_images (np.ndarray): Batch of preprocessed images.
            batch_masks (np.ndarray): Batch of corresponding masks.
        """
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]

        batch_image_filenames = [self.image_filenames[i] for i in batch_indices]
        batch_mask_filenames = [self.mask_filenames[i] for i in batch_indices]

        # Load and preprocess batch images and masks
        batch_images = []
        batch_masks = []
        for image_filename, mask_filename in zip(batch_image_filenames, batch_mask_filenames):
            image_path = os.path.join(self.image_directory, image_filename)
            mask_path = os.path.join(self.mask_directory, mask_filename)

            img = load_img(image_path, target_size=self.target_size, color_mode='rgb')
            img = img_to_array(img)
            img = img / 255.0  # Normalize the image to values between 0 and 1

            mask = img_to_array(load_img(mask_path, target_size=self.target_size))
            mask = (mask > 0).astype(np.float32)

            # Data augmentation: Flip images horizontally and/or vertically.
            if self.data_augmentation:
                rand_flip1 = np.random.randint(0, 2)
                rand_flip2 = np.random.randint(0, 2)
                if rand_flip1 == 1:
                    img = np.flip(img, 0)
                    mask = np.flip(mask, 0)
                if rand_flip2 == 1:
                    img = np.flip(img, 1)
                    mask = np.flip(mask, 1)

            batch_images.append(img)
            batch_masks.append(mask)

        batch_images = np.array(batch_images)
        batch_masks = np.array(batch_masks)

        return batch_images, batch_masks

    def on_epoch_end(self):
        """
        Shuffle the data indices at the end of each epoch if required.
        """
        if self.shuffle:
            np.random.shuffle(self.indices)
