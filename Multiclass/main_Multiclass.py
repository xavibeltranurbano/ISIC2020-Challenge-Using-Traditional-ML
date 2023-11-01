# -----------------------------------------------------------------------------
# Main Multiclass file
# Author: Xavier Beltran Urbano
# Date Created: 31-10-2023
# -----------------------------------------------------------------------------

from training import Training
import pandas as pd
from utils import Utils

def run_program():
    img_path = '/Users/xavibeltranurbano/Desktop/MAIA/GIRONA/CAD/MACHINE LEARNING/Multiclass/'
    base_path = '/Users/xavibeltranurbano/Desktop/MAIA/GIRONA/CAD/MACHINE LEARNING/Multiclass/features'
    img_size = (256, 256)
    utils=Utils(type_training='Multiclass',img_path=img_path,img_size=img_size)

    # Train data
    vec_features_train, vec_gt_train = utils.load_and_extract_features('train')
    utils.save_features_to_csv(vec_features_train, vec_gt_train, 'train', base_path)

    # Validation data
    vec_features_val, vec_gt_val = utils.load_and_extract_features('val')
    utils.save_features_to_csv(vec_features_val, vec_gt_val, 'val', base_path)

    """vec_features_train = pd.read_csv(f'/Users/xavibeltranurbano/Desktop/MAIA/GIRONA/CAD/MACHINE LEARNING/Multiclass/features/features_train_256x256.csv')  # Set index=False to exclude the index column
    vec_gt_train = pd.read_csv(f'/Users/xavibeltranurbano/Desktop/MAIA/GIRONA/CAD/MACHINE LEARNING/Multiclass/features/gt_features_train_256x256.csv')  # Set index=False to exclude the index column
    vec_features_val = pd.read_csv(f'/Users/xavibeltranurbano/Desktop/MAIA/GIRONA/CAD/MACHINE LEARNING/Multiclass/features/features_val_256x256.csv')  # Set index=False to exclude the index column
    vec_gt_val = pd.read_csv(f'/Users/xavibeltranurbano/Desktop/MAIA/GIRONA/CAD/MACHINE LEARNING/Multiclass/features/gt_features_val_256x256.csv')  # Set index=False to exclude the index column
    """

    # Initialize and fit training
    training = Training(vec_features_train, vec_features_val, vec_gt_train, vec_gt_val,type_training='Multiclass', cv=5)
    training.fit()

    # Test data
    vec_features_test = utils.load_and_extract_features_test('test')
    #vec_features_test = pd.read_csv(f'/Users/xavibeltranurbano/Desktop/MAIA/GIRONA/CAD/MACHINE LEARNING/Multiclass/features/features_test_256x256.csv')  # Set index=False to exclude the index column
    utils.save_features_to_csv(vec_features_test,subset='test', base_path=base_path)
    training.predict_test(vec_features_test)


if __name__ == "__main__":
    run_program()
