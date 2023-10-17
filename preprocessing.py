# -----------------------------------------------------------------------------
# Preprocessing Class
# Author: Xavier Beltran Urbano
# Date Created: 17-10-2023
# -----------------------------------------------------------------------------

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from skimage import io, color, segmentation

# Preprocessing Class
class Preprocessing:

    def _roi(self,img, threshold=50):
        # image dimensions
        h,w = img.shape[:2]

        # coordinates of the pixels in the diagonal
        y_coords = list(range(0, h))
        x_coords = list(range(0, w))

        # Mean value of the pixels along the diagonal
        diagonal_values = [np.mean(img[i, i, :]) for i in range(min(h, w))]

        # Find the first and last points where the threshold is crossed
        first_cross = next(i for i, value in enumerate(diagonal_values) if value >= threshold)
        last_cross = len(diagonal_values) - next(i for i, value in enumerate(reversed(diagonal_values)) if value >= threshold)

        # Set the coordinates to crop the image
        y1 = max(0, first_cross)
        y2 = min(h, last_cross)
        x1 = max(0, first_cross)
        x2 = min(w, last_cross)

        # Crop the image using the calculated coordinates
        img_new = img[y1:y2, x1:x2, :]

        if img_new.shape[0] == 0 or img_new.shape[1] == 0:
          img_new = img

        return img_new

    def _color_constancy(self,img, power=6, gamma=None):
        # Get the data type of the input image
        img_dtype = img.dtype

        # Apply gamma correction to the image if gamma is provided
        if gamma is not None:
            img = img.astype('uint8')
            # Create a lookup table for gamma correction
            look_up_table = np.ones((256, 1), dtype='uint8') * 0
            for i in range(256):
                look_up_table[i][0] = 255 * pow(i/255, 1/gamma)
            img = cv.LUT(img, look_up_table)

        # Convert the image to float32 data type for further processing
        img = img.astype('float32')

        # Apply power transformation to the image
        img_power = np.power(img, power)

        # Calculate the mean of img_power along channels (0 and 1)
        rgb_vec = np.power(np.mean(img_power, (0, 1)), 1/power)

        # Calculate the L2 norm of rgb_vec
        rgb_norm = np.sqrt(np.sum(np.power(rgb_vec, 2.0)))

        # Normalize rgb_vec to have unit length
        rgb_vec = rgb_vec / rgb_norm

        # Scale img using the color constancy vector
        rgb_vec = 1 / (rgb_vec * np.sqrt(3))
        img = np.multiply(img, rgb_vec)
        img = np.clip(img, a_min=0, a_max=255)

        return img.astype(img_dtype)

    def _hair_removal(self,img, se_size= 15):
        # Convert the original image to grayscale if it has more than 1 channel
        if (len(img.shape)==3):
          channel = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        else:
          channel = img

        # Structuring Element for the morphological filtering
        se = cv.getStructuringElement(1, (se_size, se_size))
        se2 = np.array(list(reversed(list(zip(*np.eye(se_size)))))) + np.eye(se_size)
        se2[int(se_size/2), int(se_size/2)] = 1

        # Perform the blackHat filtering on the grayscale image to find the hair countours
        blackhat = cv.morphologyEx(channel, cv.MORPH_BLACKHAT, se)
        blackhat2 = cv.morphologyEx(channel, cv.MORPH_BLACKHAT, se2.astype(np.uint8))
        bHat = blackhat + blackhat2

        # Intensify the countours detected in preparation for the hair removal
        ret, thresh = cv.threshold(bHat, 10, 255, cv.THRESH_BINARY)

        # Inpaint the original image depending on the mask
        Inp = cv.inpaint(img, thresh, 1, cv.INPAINT_TELEA)

        return Inp

    def _add_padd_to_image(self,img,padding_size):
        # Get the dimensions of the original image
        height_original, width_original = img.shape

        # Create a new larger image with black padding
        new_height = height_original + 2 * padding_size
        new_width = width_original + 2 * padding_size
        padding_image = np.zeros((new_height, new_width), dtype=np.uint8)

        # Calculate the position to place the original image in the center of the new image
        x_offset = padding_size
        y_offset = padding_size

        # Copy the original image into the center of the new image
        padding_image[y_offset:y_offset+height_original, x_offset:x_offset+width_original] = img

        return padding_image, height_original,width_original

    def _back_original_shape(self,eroded_mask, width_original, height_original,padding_size):
        # Back to the original shape
        crop_x_start = padding_size
        crop_x_end = padding_size + width_original
        crop_y_start = padding_size
        crop_y_end = padding_size + height_original

        # Crop the padded image to the original size
        cropped_mask = 255-eroded_mask[crop_y_start:crop_y_end, crop_x_start:crop_x_end]

        return cropped_mask

    def _plot_results(self,img,image_file, hair_removed_image,roi_image,cn_hair_removed_image, segmented_image, cropped_mask_3_channels,final_image):
        # Create a figure with a larger size
        plt.figure(figsize=(14, 4))

        # Display the original image
        plt.subplot(1, 7, 1)
        plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        plt.title(f'Original: {image_file}', fontsize=5)
        plt.axis('off')

        # Display the padded image
        plt.subplot(1, 7, 2)
        plt.imshow(cv.cvtColor(hair_removed_image, cv.COLOR_BGR2RGB))
        plt.title('Step 1: Hair Removal', fontsize=5)
        plt.axis('off')

        # Display the eroded mask
        plt.subplot(1, 7, 3)
        plt.imshow(cn_hair_removed_image, cmap='binary')
        plt.title('Step 2: Color Normalization', fontsize=5)
        plt.axis('off')

        # Display the ROI image
        plt.subplot(1, 7, 4)
        plt.imshow(cv.cvtColor(roi_image, cv.COLOR_BGR2RGB))
        plt.title('Step 3: ROI Image', fontsize=5)
        plt.axis('off')

        # Display the eroded mask
        plt.subplot(1, 7, 5)
        plt.imshow(255-segmented_image, cmap='binary')
        plt.title('Step 4: Padding Image', fontsize=5)
        plt.axis('off')

        # Display the eroded mask
        plt.subplot(1, 7, 6)
        plt.imshow(cv.cvtColor(cropped_mask_3_channels, cv.COLOR_BGR2RGB))
        plt.title('Step 5: Binary Mask', fontsize=5)
        plt.axis('off')

        # Display the final image
        plt.subplot(1, 7, 7)
        plt.imshow(cv.cvtColor(final_image, cv.COLOR_BGR2RGB))
        plt.title('Final Image', fontsize=5)
        plt.axis('off')

        # Adjust the spacing between subplots
        plt.tight_layout()

        # Show the figure
        plt.show()


    def preprocess_image(self, img,image_file ,plot_results):
            # Call the function to remove the hairs from the image
            hair_removed_image = self._hair_removal(img)
            # Call the function to normalise the colours
            cn_hair_removed_image = self._color_constancy(hair_removed_image)
            # Call the function to crop the images
            roi_image = self._roi(cn_hair_removed_image)

            # Apply Gaussian blur
            kernel_size = (5, 5)  # Adjust the kernel size as needed
            sigma_x = 0  # You can adjust the standard deviation if needed

            # Apply Gaussian blur
            img_blurred = cv.GaussianBlur(roi_image, kernel_size, sigma_x)
            img_gray = cv.cvtColor(img_blurred, cv.COLOR_BGR2GRAY)

            # Define the padding size (in pixels)
            padding_size = 50  # Adjust this value as needed

            # Apply padding
            padding_image, height_original, width_original = self._add_padd_to_image(img_gray,padding_size)

            # Region growing algorithm
            seed = (0, 0)  # Seed point as a single tuple
            segmented_image = segmentation.flood_fill(padding_image, seed_point=seed, new_value=0, tolerance=30)

            # Apply the thresholding to create the binary mask
            mask = np.where(segmented_image == 0, 255, 0).astype(np.uint8)

            # Define the kernel for erosion
            kernel_size = 11
            kernel = np.ones((kernel_size, kernel_size), np.uint8)

            # We remove small elements
            mask = cv.dilate(cv.erode(mask, kernel, iterations=5), kernel, iterations=5)

            # We dilate the mask
            eroded_mask = cv.dilate(mask, kernel, iterations=5)

            # Back to the original shape
            cropped_mask = self._back_original_shape(eroded_mask, width_original, height_original,padding_size)

            # Final output
            cropped_mask_3_channels = cv.cvtColor(cropped_mask, cv.COLOR_GRAY2BGR)
            final_image = 255 - (cropped_mask_3_channels * roi_image)

            # Plot results
            if plot_results:
              self._plot_results(img,image_file, hair_removed_image,roi_image,cn_hair_removed_image, segmented_image, cropped_mask_3_channels,final_image)

            return final_image

# Usage
if __name__ == "__main__":
    """image_folder = "/content/drive/MyDrive/Skin_Lesion_Dataset/Binary_Challenge/train/nevus/"
    preprocessing = Preprocessing(image_folder)
    # Get a list of image files in the folder
    image_files = os.listdir(image_folder)
    # Shuffle the list of image files
    random.shuffle(image_files)
    # Limit the number of images to 50 if there are more
    num_images_to_process = min(50, len(image_files))
    # Choose the first image
    image_file = image_files[0]
    image_path = os.path.join(image_folder, image_file)
    # Read image
    img = cv.imread(image_path)
    preprocessing.preprocess_image(img,image_file, plot_results=True)"""