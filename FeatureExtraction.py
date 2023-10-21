# Import necessary libraries
import numpy as np
import cv2 as cv
import pandas as pd
from skimage.feature import local_binary_pattern
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy
import mahotas
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


class FeatureExtarction:

  # Global Features
  def variegation(self,Im: np.ndarray):
    # Split Color channels
    lesion_r, lesion_g, lesion_b = np.split(Im, 3, axis=2)

    # Compute the normalized Standard Deviation of each channel
    C_r = np.std(lesion_r) / np.max(lesion_r)
    C_g = np.std(lesion_g) / np.max(lesion_g)
    C_b = np.std(lesion_b) / np.max(lesion_b)

    variegation = np.array([C_r, C_g, C_b]).reshape(1, -1)
    return variegation

  # Color Features
  def color_moments(self,Img: np.ndarray):

    c1, c2, c3 = cv.split(Img)
    color_feature = []  # Initialize the color feature

    # -- The first central moment - average
    c1_mean, c2_mean, c3_mean = np.mean(c1), np.mean(c2), np.mean(c3)
    color_feature.extend([c1_mean, c2_mean, c3_mean])

    # -- The second central moment - standard deviation
    c1_std, c2_std, c3_std = np.std(c1), np.std(c2), np.std(c3)
    color_feature.extend([c1_std, c2_std, c3_std])

    # -- The third central moment - the third root of the skewness
    c1_skewness = np.mean(np.abs(c1 - c1_mean) ** 3)
    c2_skewness = np.mean(np.abs(c2 - c2_mean) ** 3)
    c3_skewness = np.mean(np.abs(c3 - c3_mean) ** 3)
    c1_thirdMoment, c2_thirdMoment, c3_thirdMoment = c1_skewness ** (1. / 3), c2_skewness ** (1. / 3), c3_skewness ** (1. / 3)
    color_feature.extend([c1_thirdMoment, c2_thirdMoment, c3_thirdMoment])

    # -- The fourth central moment - the variance
    c1_var, c2_var, c3_var = c1_std ** 2, c2_std ** 2, c3_std ** 2
    color_feature.extend([c1_var, c2_var, c3_var])

    return np.array(color_feature).reshape(1, -1)

  # Color Histogram Features
  def calculate_normalized_histogram(self,channel, n_bins):
    hist = cv.calcHist([channel], [0], None, [n_bins], [1, 256])
    hist = hist / hist.sum()
    return hist

  def color_histogram(self,Img: np.ndarray, n_bins: int = 256):
    rgb_channels = cv.split(Img)
    hsv_image = cv.cvtColor(Img, cv.COLOR_RGB2HSV)
    hsv_channels = cv.split(hsv_image)
    lab_image = cv.cvtColor(Img, cv.COLOR_RGB2LAB)
    lab_channels = cv.split(lab_image)

    rgb_hist = np.concatenate([self.calculate_normalized_histogram(channel, n_bins) for channel in rgb_channels])
    hsv_hist = np.concatenate([self.calculate_normalized_histogram(channel, n_bins) for channel in hsv_channels])
    lab_hist = np.concatenate([self.calculate_normalized_histogram(channel, n_bins) for channel in lab_channels])

    return rgb_hist.reshape(1, -1), hsv_hist.reshape(1, -1), lab_hist.reshape(1, -1)

  # Texture Features
  def extract_lbp_feature(self,Img: np.ndarray, P: int = 16, R: int = 2):
    blue_Img = Img[:, :, 2]  # Use the blue channel of the Image
    lbp = local_binary_pattern(blue_Img, P, R, method='uniform')
    n_bins = int(lbp.max() + 1)
    lbp_fd, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    lbp_fd = np.array(lbp_fd).reshape(1, -1)
    return lbp_fd

  def extract_haralick_feature(self,Img: np.ndarray):
    blue_Img = Img[:, :, 2]  # Use the blue channel of the Image
    haralick_fd = mahotas.features.haralick(blue_Img).mean(axis=0)
    haralick_fd = np.array(haralick_fd).reshape(1, -1)
    return haralick_fd

  def extract_glcm_feature(self,Img: np.ndarray):
    blue_Img = Img[:, :, 2]  # Use the blue channel of the Image
    distance = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    properties = ['correlation', 'homogeneity', 'contrast', 'energy', 'dissimilarity']
    glcm_fd = []
    glcm_mat = graycomatrix(blue_Img, distances=distance, angles=angles, symmetric=True, normed=True)
    glcm_fd = np.hstack([graycoprops(glcm_mat, props).ravel() for props in properties])
    glcm_fd = np.array(glcm_fd).reshape(1, -1)
    return glcm_fd

  def extract_texture_features(self,Img: np.ndarray, P, R):
    lbp_fd = self.extract_lbp_feature(Img, P, R)
    haralick_fd = self.extract_haralick_feature(Img)
    glcm_fd = self.extract_glcm_feature(Img)
    return lbp_fd, haralick_fd, glcm_fd

  def feature_normalization(self,features):
      # Initialize the MinMaxScaler
      scaler = MinMaxScaler()
      features.columns = features.columns.astype(str)
      features_norm = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)
      return features_norm

  def extract_all(self,image):

      # Initialize an empty DataFrame to store the features
      feature_df = pd.DataFrame()
      variegation_features = self.variegation(image)
      color_moments_features = self.color_moments(image)
      rgb_histogram, hsv_histogram, lab_histogram = self.color_histogram(image, 64)
      lbp_feature = self.extract_lbp_feature(image, 16, 2)
      haralick_feature = self.extract_haralick_feature(image)
      glcm_features = self.extract_glcm_feature(image)

      # Concatenate features into a single feature vector
      features = np.concatenate([
          variegation_features, color_moments_features, rgb_histogram, hsv_histogram, lab_histogram, lbp_feature,
          haralick_feature, glcm_features
      ], axis=1)

      # Create a DataFrame for the current image's features
      image_feature_df = pd.DataFrame(features)

      return image_feature_df

if __name__ == "__main__":
    # Load an image for feature extraction
    image_folder = "/Users/xavibeltranurbano/Desktop/MAIA/GIRONA/CAD/MACHINE LEARNING/BINARY/train/nevus/nev00002.jpg"
    img = cv.imread(image_folder)

    # Create an instance of the Features class
    features = FeatureExtarction()

    # Extract and print color features
    vec_features = features.extract_all(img)
