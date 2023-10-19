# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from features import Features  # Import your custom features module
import pandas as pd


# Function to read the dataset
def read_dataset(img_size, subset,class_name):
    # Initialize vectors
    vec_img = []
    vec_labels = []
    # Define the path to the dataset
    path_images = os.path.join('/Users/xavibeltranurbano/Desktop/MAIA/GIRONA/CAD/MACHINE LEARNING/BINARY/', subset)
    # Define label-to-number mapping
    label_to_number = {class_name: 0, 'others_name': 1}

    for label, label_number in label_to_number.items():
        images_path = os.path.join(path_images, label)
        image_files = os.listdir(images_path)
        total_images = len(image_files) * 2  # Calculate the total number of images to read

        with tqdm(total=len(image_files), desc=f'Reading images {subset} ({label})...') as pbar:  # Initialize the progress bar
            for file_name in image_files:
                image = os.path.join(images_path, file_name)
                img = cv.imread(image)
                resized_img = cv.resize(img, img_size)
                vec_img.append(resized_img)
                vec_labels.append(label_number)
                pbar.update(1)  # Update the progress bar

    return np.asarray(vec_img), np.asarray(vec_labels)


# Function to extract features
def extract_features(train, train_gt, subset):
    vec_features_train = []
    vec_gt_train = []
    features = Features()  # Initialize your feature extraction class

    with tqdm(total=len(train), desc=f'{subset}: Extracting features... ') as pbar:  # Initialize the progress bar
        for i in range(len(train)):
            vec_features_train.append(features.extract_all(train[i]))
            vec_gt_train.append(train_gt[i])
            pbar.update(1)  # Close the progress bar

    vec_features_train = pd.concat(vec_features_train)
    vec_gt_train_ = pd.Series(vec_gt_train)

    return vec_features_train, vec_gt_train_


# Function to train the SVM model
def train_models(vec_features_train, vec_gt_train, vec_features_val, vec_gt_val):
    # Initialize classifiers
    classifiers = [
        ('SVM', SVC(kernel='linear')),
        ('Random Forest', RandomForestClassifier(max_iter=200)),
        ('Logistic Regression', LogisticRegression())
    ]

    results = {}  # Dictionary to store results
    print("\n------------Results------------")
    all_predictions=[]
    for name, classifier in classifiers:
        # Train the classifier
        classifier.fit(vec_features_train, vec_gt_train.ravel())

        # Make predictions on the validation set
        predictions = classifier.predict(vec_features_val)
        all_predictions.append(predictions)
        # Calculate accuracy
        accuracy = accuracy_score(np.array(vec_gt_val).ravel(), predictions)

        # Store the accuracy in the results dictionary
        results[name] = accuracy

        print(f"{name} Accuracy:", accuracy)

    # We do Voting with all the previous models
    summed_vector = [sum(vector[i] for vector in all_predictions) for i in range(len(all_predictions[0]))]
    voting=np.where(np.array(summed_vector)>1,1,0)
    accuracy = accuracy_score(np.array(vec_gt_val).ravel(), voting)
    print(f"Voting Accuracy:", accuracy)
    all_predictions.append(voting)

    return np.max(all_predictions)


# Function to normalize features using Min-Max scaling
def normalise_features(features):
    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()
    features.columns = features.columns.astype(str)
    features_norm = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)
    features_norm.head()
    return features_norm



if __name__ == "__main__":
    img_size = (256, 256)
    class_names=['scc','mel','bcc']
    vec_accuracies=[]
    for class_name in class_names:
        # Read the training dataset
        train, train_gt = read_dataset(img_size=img_size, subset='train',class_name=class_name)
        print(f"\nTrain set: {train.shape[0]} images")


        # Read the validation dataset
        val, val_gt = read_dataset(img_size=img_size, subset='val',class_name=class_name)
        print(f"Val set: {val.shape[0]} images")

        # Extract features from the training dataset
        vec_features_train, vec_gt_train = extract_features(train, train_gt, subset='train')
        """vec_features_train.to_csv(f'/Users/xavibeltranurbano/Desktop/MAIA/GIRONA/CAD/MACHINE LEARNING/BINARY/features_train.csv', index=False)  # Set index=False to exclude the index column
        pd.Series(vec_gt_train).to_csv(f'/Users/xavibeltranurbano/Desktop/MAIA/GIRONA/CAD/MACHINE LEARNING/BINARY/gt_train.csv', index=False)  # Set index=False to exclude the index column
        """
        # Extract features from the validation dataset
        vec_features_val, vec_gt_val = extract_features(val, val_gt, subset='val')
        """vec_features_val.to_csv(f'/Users/xavibeltranurbano/Desktop/MAIA/GIRONA/CAD/MACHINE LEARNING/BINARY/features_val.csv',index=False)  # Set index=False to exclude the index column
        pd.Series(vec_gt_val).to_csv(f'/Users/xavibeltranurbano/Desktop/MAIA/GIRONA/CAD/MACHINE LEARNING/BINARY/gt_val.csv',index=False)  # Set index=False to exclude the index column
        """
        # Train the SVM model
        accuracy=train_models(normalise_features(vec_features_train), vec_gt_train, normalise_features(vec_features_val), vec_gt_val)
        vec_accuracies.append(accuracy)

    # Compute balanced accuracy score
    bma=np.sum(vec_accuracies)/vec_accuracies.shape[0]


