# -----------------------------------------------------------------------------
# Main Binary file
# Author: Xavier Beltran Urbano
# Date Created: 31-10-2023
# -----------------------------------------------------------------------------

from training import Training
from utils import Utils
import pandas as pd

def run_program():
    img_path = '/Users/xavibeltranurbano/Desktop/MAIA/GIRONA/CAD/MACHINE LEARNING/Binary/ROI/'
    base_path = '/Users/xavibeltranurbano/Desktop/MAIA/GIRONA/CAD/MACHINE LEARNING/Binary/features'
    img_size = (256, 256)
    utils = Utils(type_training='Binary', img_path=img_path, img_size=img_size)

    # Train data
    vec_features_train, vec_gt_train = utils.load_and_extract_features('train')
    utils.save_features_to_csv(vec_features_train, ground_truth=vec_gt_train, base_path=base_path, subset='train')

    # Validation data
    vec_features_val, vec_gt_val = utils.load_and_extract_features('val')
    utils.save_features_to_csv(vec_features_val, ground_truth=vec_gt_val, base_path=base_path,subset='val')

    # Initialize and fit training
    training = Training(vec_features_train, vec_features_val, vec_gt_train, vec_gt_val,type_training='Binary', cv=5)
    training.fit()

    # Read and predict test data
    vec_features_test = utils.load_and_extract_features_test('test')
    utils.save_features_to_csv(vec_features_test,base_path=base_path,subset='test')
    training.predict_test(vec_features_test)

if __name__ == "__main__":
    run_program()
