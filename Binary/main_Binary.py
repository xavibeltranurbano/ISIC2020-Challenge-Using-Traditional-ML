# -----------------------------------------------------------------------------
# Main Binary file
# Author: Xavier Beltran Urbano
# Date Created: 31-10-2023
# -----------------------------------------------------------------------------

from training import Training
from Binary.utils_Binary import *

def run_program():
    img_path = '/Users/xavibeltranurbano/Desktop/MAIA/GIRONA/CAD/MACHINE LEARNING/BINARY/ROI/'
    base_path = '/Users/xavibeltranurbano/Desktop/MAIA/GIRONA/CAD/MACHINE LEARNING/BINARY/prova'
    img_size = (256, 256)

    # Train data
    vec_features_train, vec_gt_train = load_and_extract_features(img_size, img_path, 'train')
    #save_features_to_csv(vec_features_train, vec_gt_train, 'train', base_path,img_size)

    # Validation data
    vec_features_val, vec_gt_val = load_and_extract_features(img_size, img_path, 'val')
    #save_features_to_csv(vec_features_val, vec_gt_val, 'val', base_path,img_size)

    # Initialize and fit training
    training = Training(vec_features_train, vec_features_val, vec_gt_train, vec_gt_val,type_training='Binary', cv=5)
    training.fit()

    # Test data
    #vec_features_test, _ = load_and_extract_features(img_size, img_path,'test')
    #training.predict_test(vec_features_test)


if __name__ == "__main__":
    run_program()
