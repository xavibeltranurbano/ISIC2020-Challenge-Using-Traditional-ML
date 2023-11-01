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
from featureExtraction import FeatureExtarction
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline


# Function to read the dataset
def read_dataset(img_size, img_path,subset):
    # Initialize vectors
    vec_img = []
    vec_labels = []
    path_images = os.path.join(img_path, subset) # Define the path to the dataset
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
                    resized_img = cv.resize(img, img_size)
                    vec_img.append(resized_img)
                    vec_labels.append(label_number)
                    pbar.update(1)  # Update the progress bar

    return np.asarray(vec_img), np.asarray(vec_labels)

# Function to modify the column headings
def column_headings():
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
def normalise_features(vec_features, vec_gt):
    features=FeatureExtarction()
    # We normalize the data
    normalized_features = features.feature_normalization(vec_features)

    # Shuffle the rows
    normalized_features = normalized_features.sample(frac=1, random_state=0)
    normalized_features = normalized_features.reset_index(drop=True)

    vec_gt = vec_gt.sample(frac=1, random_state=0)
    vec_gt = vec_gt.reset_index(drop=True)
    return normalized_features,vec_gt

# Function to extract features
def extract_features(vec_img, vec_img_gt, subset):
    vec_features = []
    vec_gt = []
    features = FeatureExtarction()  # Initialize your feature extraction class

    with tqdm(total=len(vec_img), desc=f'{subset}: Extracting features... ') as pbar:  # Initialize the progress bar
        for i in range(len(vec_img)):
            vec_features.append(features.extract_all(vec_img[i]))
            vec_gt.append(vec_img_gt[i])
            pbar.update(1)  # Close the progress bar

    vec_features = pd.concat(vec_features)
    vec_gt= pd.Series(vec_gt)

    # Set meaningful column headings
    vec_features.columns = column_headings()

    # We normalize the data
    normalized_features,vec_gt = normalise_features(vec_features,vec_gt)

    return normalized_features, vec_gt

# Function to read and extract features
def load_and_extract_features(img_size, img_path, subset):
    # Read the dataset
    data, data_gt = read_dataset(img_size=img_size,img_path=img_path, subset=subset)
    print(f"\n{subset} set: {data.shape[0]} images")

    # Extract features from the dataset
    vec_features, vec_gt = extract_features(data, data_gt, subset=subset)

    # Normalise features
    vec_features, vec_gt = normalise_features(vec_features, vec_gt)

    return vec_features, vec_gt

# Function to save features
def save_features_to_csv(features, ground_truth, subset, base_path,img_size):
    # Save features
    features.to_csv(f'{base_path}/features_{subset}_{img_size[0]}x{img_size[1]}.csv', index=False)
    pd.Series(ground_truth).to_csv(f'{base_path}/features_{subset}_{img_size[0]}x{img_size[1]}.csv', index=False)

# Function to DEAL WITH CLASS IMBALANCE
def apply_smote_undersample(train_feat, train_labels):
    # Define the sampling strategy
    sampling_strategy = {0: 2713, 1: 1993, 2: 1694}

    # SMOTE Approach
    sm = SMOTE(sampling_strategy=sampling_strategy, random_state=24)
    under = RandomUnderSampler(sampling_strategy={0: 1694, 1: 1694, 2: 1694})
    steps = [('o', sm), ('u', under)]
    pipeline = Pipeline(steps=steps)

    # Sample the features:
    train_feats, trains_labels = pipeline.fit_resample(train_feat, train_labels)
    return train_feats, trains_labels