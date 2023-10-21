# -----------------------------------------------------------------------------
# Features Class
# Author: Xavier Beltran Urbano
# Date Created: 17-10-2023
# -----------------------------------------------------------------------------


# Import necessary libraries
import numpy as np
import cv2 as cv
import pandas as pd
from skimage.feature import local_binary_pattern
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy
from sklearn.preprocessing import MinMaxScaler


# Features Class
class Features:
    # Constructor for the Features class

    def _extract_color_moments(self, img, color_type):
        # Convert the image to the RGB color space
        image_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        # Split the image into color channels
        red_channel = image_rgb[:, :, 0]
        green_channel = image_rgb[:, :, 1]
        blue_channel = image_rgb[:, :, 2]

        # Calculate color moments for each channel
        def calculate_color_moments(channel):
            mean = np.mean(channel)
            std_deviation = np.std(channel)
            skewness = np.mean((channel - mean) ** 3) / (std_deviation ** 3)
            energy = np.sum(channel ** 2)
            return mean, std_deviation, skewness, energy

        # Calculate color moments for each channel
        red_moments = calculate_color_moments(red_channel)
        green_moments = calculate_color_moments(green_channel)
        blue_moments = calculate_color_moments(blue_channel)

        # Return color moments as a dictionary
        return {
            f"Red_Moments_mean_{color_type}": red_moments[0],
            f"Red_Moments_std_{color_type}": red_moments[1],
            f"Red_Moments_skewness_{color_type}": red_moments[2],
            f"Red_Moments_energy_{color_type}": red_moments[3],
            f"Green_Moments_mean_{color_type}": green_moments[0],
            f"Green_Moments_std_{color_type}": green_moments[1],
            f"Green_Moments_skewness_{color_type}": green_moments[2],
            f"Green_Moments_energy_{color_type}": green_moments[3],
            f"Blue_Moments_mean_{color_type}": blue_moments[0],
            f"Blue_Moments_std_{color_type}": blue_moments[1],
            f"Blue_Moments_skewness_{color_type}": blue_moments[2],
            f"Blue_Moments_energy_{color_type}": blue_moments[3]
        }

    def _extract_lbp_features(self, img, P, R):
        # Convert the image to grayscale
        gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Calculate LBP
        n_points = P * R
        lbp_image = local_binary_pattern(gray_img, n_points, R, method='uniform')

        # Calculate the LBP histogram
        lbp_hist, _ = np.histogram(lbp_image.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))

        # Normalize the histogram
        lbp_hist = lbp_hist.astype("float")
        lbp_hist /= (lbp_hist.sum() + 1e-6)

        # Return LBP features as a list
        return lbp_hist.tolist()

    def _extract_glcm_features(self, img, distance=1, angle=np.pi / 4):
        # Convert the image to grayscale
        gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Calculate the GLCM
        glcm = graycomatrix(gray_img, [distance], [angle], symmetric=True, normed=True, levels=256)

        # Calculate GLCM properties (texture features)
        properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
        glcm_features = {}

        for prop in properties:
            feature_value = graycoprops(glcm, prop)[0]
            key = f'{prop}_distance_{distance}_angle_{angle}'
            glcm_features[key] = feature_value

        # Return GLCM features as a dictionary
        return glcm_features

    def color_features(self, img):
        color_moments_vec = []
        color_moments_vec_hsv = []
        color_moments_vec_lab = []
        vec_entropy = []

        # Extract color features in RGB, HSV, and LAB color spaces
        color_moments_vec.append(self._extract_color_moments(img, 'rgb'))  # RGB
        color_moments_vec_hsv.append(self._extract_color_moments(cv.cvtColor(img, cv.COLOR_BGR2HSV), 'hsv'))  # HSV
        color_moments_vec_lab.append(self._extract_color_moments(cv.cvtColor(img, cv.COLOR_BGR2LAB), 'lab'))  # LAB

        # Calculate entropy
        vec_entropy.append({'Entropy': shannon_entropy(img)})

        # Concatenate all color features into a DataFrame
        return pd.concat(
            [pd.DataFrame(color_moments_vec), pd.DataFrame(color_moments_vec_hsv), pd.DataFrame(color_moments_vec_lab),
             pd.DataFrame(vec_entropy)], axis=1)

    def texture_features(self, img):
        lbp_vec_8_1 = []
        lbp_vec_16_2 = []
        glcm_vec = []

        # Extract LBP and GLCM texture features
        lbp_vec_8_1.append(self._extract_lbp_features(img, 8, 1))
        lbp_vec_16_2.append(self._extract_lbp_features(img, 16, 2))
        glcm_vec.append(self._extract_glcm_features(img))

        # Concatenate all texture features into a DataFrame
        return pd.concat([pd.DataFrame(lbp_vec_8_1), pd.DataFrame(lbp_vec_16_2), pd.DataFrame(glcm_vec)], axis=1)

    def normalization(self):
        scaler = MinMaxScaler()


    def extract_all(self, img):
        # Extract and concatenate all color and texture features
        features=pd.concat([self.color_features(img), self.texture_features(img)], axis=1)

        # Normalize the data

        normalized_features=self.normalization(features)
        return normalized_features


# Entry point of the script
if __name__ == "__main__":
    # Load an image for feature extraction
    image_folder = "/Users/xavibeltranurbano/Desktop/MAIA/GIRONA/CAD/MACHINE LEARNING/BINARY/train/nevus/nev00002.jpg"
    img = cv.imread(image_folder)

    # Create an instance of the Features class
    features = Features()

    # Extract and print color features
    vec_features = features.color_features(img)
    print("Color Features:")
    print(vec_features.head())

    # Extract and print texture features
    vec_features = features.texture_features(img)
    print("Texture Features:")
    print(vec_features.head())

    # Extract and print all features
    vec_features = features.extract_all(img)
    print("All Features:")
    print(vec_features.head())
