# Import necessary libraries
import numpy as np
import cv2 as cv
import os
from tqdm import tqdm
import pandas as pd
from FeatureExtraction import FeatureExtarction
from Training import Training


# Function to read the dataset
def read_dataset(img_size, subset):
    # Initialize vectors
    vec_img = []
    vec_labels = []
    # Define the path to the dataset
    path_images = os.path.join('/Users/xavibeltranurbano/Desktop/MAIA/GIRONA/CAD/MACHINE LEARNING/BINARY/', subset)
    # Define label-to-number mapping
    label_to_number = {'nevus': 0, 'others': 1}

    for label, label_number in label_to_number.items():
        images_path = os.path.join(path_images, label)
        image_files = os.listdir(images_path)

        with tqdm(total=len(image_files), desc=f'Reading images {subset} ({label})...') as pbar:  # Initialize the progress bar
            for file_name in image_files:
                image = os.path.join(images_path, file_name)
                img = cv.imread(image)
                resized_img = cv.resize(img, img_size)
                vec_img.append(resized_img)
                vec_labels.append(label_number)
                pbar.update(1)  # Update the progress bar

    return np.asarray(vec_img), np.asarray(vec_labels)

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


if __name__ == "__main__":
    """levels = [256, 192, 176, 128, 80, 64, 16, 8]
    img_size = (256, 256)

    for pyramid_level in levels:
        img_size = (pyramid_level, pyramid_level)
        # Read the training dataset
        train, train_gt = read_dataset(img_size=img_size, subset='train')
        print(f"\nTrain set: {train.shape[0]} images")

        # Extract features from the training dataset
        vec_features_train, vec_gt_train = extract_features(train, train_gt, subset='train')

        # Normalise features
        vec_features_train,vec_gt_train=normalise_features(vec_features_train,vec_gt_train)

        # Save train features
        vec_features_train.to_csv(f'/Users/xavibeltranurbano/Desktop/MAIA/GIRONA/CAD/MACHINE LEARNING/BINARY/features/features_train_{pyramid_level}x{pyramid_level}.csv', index=False)  # Set index=False to exclude the index column
        pd.Series(vec_gt_train).to_csv(f'/Users/xavibeltranurbano/Desktop/MAIA/GIRONA/CAD/MACHINE LEARNING/BINARY/features/gt_train_{pyramid_level}x{pyramid_level}.csv', index=False)  # Set index=False to exclude the index column

        # Read the validation dataset
        val, val_gt = read_dataset(img_size=img_size, subset='val')
        print(f"Val set: {val.shape[0]} images")

        # Extract features from the validation dataset
        vec_features_val, vec_gt_val = extract_features(val, val_gt, subset='val')

        # Normalise features
        vec_features_val, vec_gt_val = normalise_features(vec_features_val, vec_gt_val)

        # Save val features
        vec_features_val.to_csv(f'/Users/xavibeltranurbano/Desktop/MAIA/GIRONA/CAD/MACHINE LEARNING/BINARY/features/features_val_{pyramid_level}x{pyramid_level}.csv',index=False)  # Set index=False to exclude the index column
        pd.Series(vec_gt_val).to_csv(f'/Users/xavibeltranurbano/Desktop/MAIA/GIRONA/CAD/MACHINE LEARNING/BINARY/features/gt_val_{pyramid_level}x{pyramid_level}.csv',index=False)  # Set index=False to exclude the index column
"""
    """# Normal approach: Read features
    vec_features_train=pd.read_csv(f'/Users/xavibeltranurbano/Desktop/MAIA/GIRONA/CAD/MACHINE LEARNING/BINARY/features/features_train_128x128.csv')  # Set index=False to exclude the index column
    vec_gt_train=pd.read_csv(f'/Users/xavibeltranurbano/Desktop/MAIA/GIRONA/CAD/MACHINE LEARNING/BINARY/features/gt_train_128x128.csv')  # Set index=False to exclude the index column

    vec_features_val = pd.read_csv( f'/Users/xavibeltranurbano/Desktop/MAIA/GIRONA/CAD/MACHINE LEARNING/BINARY/features/features_val_128x128.csv')  # Set index=False to exclude the index column
    vec_gt_val = pd.read_csv(f'/Users/xavibeltranurbano/Desktop/MAIA/GIRONA/CAD/MACHINE LEARNING/BINARY/features/gt_val_128x128.csv')  # Set index=False to exclude the index column
"""

    #vec_features_val,vec_gt_val=normalise_features(vec_features_val,vec_gt_val)


    #### PYRAMID APPROACH
    levels = [256, 192, 176, 128, 80, 64, 16]#, 8]
    vec_features_train = []
    vec_gt_train = []
    vec_features_val = []
    vec_gt_val = []

    for pyramid_level in levels:
        value = pyramid_level - 1
        # Define file paths with the current level
        train_features_path = f'/Users/xavibeltranurbano/Desktop/MAIA/GIRONA/CAD/MACHINE LEARNING/BINARY/features/features_train_{pyramid_level}x{pyramid_level}.csv'
        train_gt_path = f'/Users/xavibeltranurbano/Desktop/MAIA/GIRONA/CAD/MACHINE LEARNING/BINARY/features/gt_train_{pyramid_level}x{pyramid_level}.csv'
        val_features_path = f'/Users/xavibeltranurbano/Desktop/MAIA/GIRONA/CAD/MACHINE LEARNING/BINARY/features/features_val_{pyramid_level}x{pyramid_level}.csv'
        val_gt_path = f'/Users/xavibeltranurbano/Desktop/MAIA/GIRONA/CAD/MACHINE LEARNING/BINARY/features/gt_val_{pyramid_level}x{pyramid_level}.csv'

        # Read data from CSV files
        vec_features_train.append(pd.read_csv(train_features_path).dropna(axis=1))
        vec_gt_train.append(pd.read_csv(train_gt_path))
        vec_features_val.append(pd.read_csv(val_features_path).dropna(axis=1))
        vec_gt_val.append(pd.read_csv(val_gt_path))

    vec_features_train_ = pd.concat(vec_features_train)
    vec_gt_train_ = pd.Series(np.concatenate(vec_gt_train, axis=0).ravel())
    vec_features_val_ = pd.concat(vec_features_val)
    vec_gt_val_ = pd.Series(np.concatenate(vec_gt_val, axis=0).ravel())

    print("THE FEATURES HAVE BEEN CONCATENATED")
    training = Training(vec_features_train_, vec_features_val_, vec_gt_train_, vec_gt_val_, cv=5)
    training.fit()