# -----------------------------------------------------------------------------
# Utils File
# Author: Xavier Beltran Urbano
# Date Created: 31-10-2023
# -----------------------------------------------------------------------------

import numpy as np
import cv2 as cv
import os
from tqdm import tqdm
import pandas as pd
from featureExtraction import FeatureExtraction
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from preprocessing import Preprocessing


class Utils:
    def __init__(self,type_training='Binary',img_path=None,img_size=(256,256)):
        """
        Utility class with various methods for dataset reading, feature extraction, normalization, and more.

        Args:
            type_training (str): Type of training (e.g., "Binary", "Multiclass"). Determines the training approach.
            img_path (str): Path to the image directory.
            img_size (tuple): Size of the images to be used.
        """
        self.preprocessing = Preprocessing()
        self.features = FeatureExtraction()
        self.type_training =type_training
        self.img_path=img_path
        self.img_size=img_size

# Function to read the dataset
    def read_dataset(self,subset):
        # Initialize vectors
        vec_img = []
        vec_labels = []
        path_images = os.path.join(self.img_path, subset) # Define the path to the dataset
        if self.type_training=='Binary':
            label_to_number = {'nevus': 0, 'others': 1}  # Define label-to-number mapping
        else:
            label_to_number = {'scc': 0, 'bcc': 1,'mel':2 } # Define label-to-number mapping

        for label, label_number in label_to_number.items():
            images_path = os.path.join(path_images, label)
            image_files = os.listdir(images_path)

            with tqdm(total=len(image_files), desc=f'Reading images {subset} ({label})...') as pbar:  # Initialize the progress bar
                for file_name in image_files:
                    image = os.path.join(images_path, file_name)
                    img = cv.imread(image)
                    if img is None:
                        print(f"Error loading image  {file_name}")
                    else:
                        resized_img = cv.resize(img, self.img_size)
                        vec_img.append(resized_img)
                        vec_labels.append(label_number)
                        pbar.update(1)  # Update the progress bar

        return np.asarray(vec_img), np.asarray(vec_labels)

    # Read test dataset
    def read_test_dataset(self,subset):
        # Initialize vectors
        vec_img = []
        path_images = os.path.join(self.img_path, subset) # Define the path to the dataset
        image_files = os.listdir(path_images)

        with tqdm(total=len(image_files), desc=f'Reading images {subset} ') as pbar:  # Initialize the progress bar
            for file_name in image_files:
                image = os.path.join(path_images, file_name)
                img = cv.imread(image)
                if img is None:
                    print(f"Error loading image  {file_name}")
                else:
                    img=self.preprocessing.preprocess_image_ROI(img)
                    resized_img = cv.resize(img, self.img_size)
                    vec_img.append(resized_img)
                    pbar.update(1)  # Update the progress bar

        return np.asarray(vec_img)

    # Function to modify the column headings
    def column_headings(self):
        # Define feature category names and counts
        feature_categories = {
            "variegation_features": 3,
            "color_moments_features": 12,
            "rgb_histogram": 192,
            "hsv_histogram": 192,
            "lab_histogram": 192,
            "lbp_feature": 18,
            "haralick_feature": 13,
            "glcm_features": 20
        }

        # Generate column headings for each feature category
        column_headings_ = []
        for category, count in feature_categories.items():
            column_headings_.extend([f"{category} {i + 1}" for i in range(count)])

        return column_headings_

    # Function to normalise features
    def normalise_features(self,vec_features, vec_gt=None):
        # We normalize the data
        normalized_features = self.features.feature_normalization(vec_features)

        # Shuffle the rows
        normalized_features = normalized_features.sample(frac=1, random_state=0)
        normalized_features = normalized_features.reset_index(drop=True)

        if vec_gt is not None:
            vec_gt = vec_gt.sample(frac=1, random_state=0)
            vec_gt = vec_gt.reset_index(drop=True)
            return normalized_features, vec_gt
        else:
            return normalized_features


    # Function to extract features
    def extract_features(self,vec_img, vec_img_gt=None,subset=None):
        vec_features = []
        vec_gt = []

        with tqdm(total=len(vec_img), desc=f'{subset}: Extracting features... ') as pbar:  # Initialize the progress bar
            for i in range(len(vec_img)):
                vec_features.append(self.features.extract_all(vec_img[i]))
                if vec_img_gt is not None:  # If ground truth is provided
                    vec_gt.append(vec_img_gt[i])
                pbar.update(1)  # Update the progress bar

        vec_features = pd.concat(vec_features)

        # Set meaningful column headings
        vec_features.columns = self.column_headings()

        # We normalize the data
        if vec_img_gt is not None:
            normalized_features, vec_gt = self.normalise_features(vec_features, vec_gt)
            return normalized_features, vec_gt
        else:
            normalized_features = self.normalise_features(vec_features)
            return normalized_features

    # Function to read and extract features
    def load_and_extract_features(self, subset):
        # Read the dataset
        data, data_gt = self.read_dataset(subset=subset)
        print(f"\n{subset} set: {data.shape[0]} images")

        # Extract features from the dataset
        vec_features, vec_gt = self.extract_features(data, data_gt, subset=subset)

        # Normalise features
        vec_features, vec_gt = self.normalise_features(vec_features, vec_gt)

        return vec_features, vec_gt

    # Function to read and extract features of the test set
    def load_and_extract_features_test(self, subset):
        # Read the dataset
        data = self.read_test_dataset(subset=subset)
        print(f"\n{subset} set: {data.shape[0]} images")

        # Extract features from the dataset
        vec_features = self.extract_features(data, subset=subset)

        # Normalise features
        vec_features = self.normalise_features(vec_features)

        return vec_features

    # Function to save features
    def save_features_to_csv(self,features, subset, base_path,ground_truth=None):
        # Save features
        if ground_truth is not None:
              pd.Series(ground_truth).to_csv(f'{base_path}/gt_features_{subset}_{self.img_size[0]}x{self.img_size[1]}.csv', index=False)
        features.to_csv(f'{base_path}/features_{subset}_{self.img_size[0]}x{self.img_size[1]}.csv', index=False)

    # Function to DEAL WITH CLASS IMBALANCE
    def apply_smote_undersample(self,train_feat, train_labels):
        # Define the sampling strategy
        sampling_strategy = {0: 2713, 1: 1993, 2: 1800}

        # SMOTE Approach
        sm = SMOTE(sampling_strategy=sampling_strategy, random_state=24)
        under = RandomUnderSampler(sampling_strategy={0: 1694, 1: 1694, 2: 1694})
        steps = [('o', sm), ('u', under)]
        pipeline = Pipeline(steps=steps)

        # Sample the features:
        train_feats, trains_labels = pipeline.fit_resample(train_feat, train_labels)
        return train_feats, trains_labels