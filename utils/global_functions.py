import csv
import numpy as np

def csv_writer(file_path, name, action, data):
    """
    Parameters
    ----------
    file_path (Path from pathlib): path where to save the csv file
    name (string): csv name
    action (char): Either 'w' to write a new csv file or 'a' to append a new row
    data (list): Data to be appended to new row

    Returns
    -------
    """
    absolute_path = file_path / name
    with open(absolute_path, action, encoding='UTF8', newline='') as f:  # 'a' to append row
        writer = csv.writer(f)
        writer.writerow(data)
        f.close()


def available_magnifications(mpp, level_downsamples):
	mpp = float(mpp)
	if (mpp<0.26):
		magnification = 40
	else:
		magnification = 20
	
	mags = []
	for l in level_downsamples:
		mags.append(magnification/l)
	
	return mags


def check_corners(img):
    """
    This function checks the corner pixels of an image and returns the pixel value (BGR) of the background.

    Parameters
    ----------
    img (numpy.ndarray): Image data

    Returns
    -------
    background_pixel (numpy.ndarray): pixel value (BGR) for the background
    """
    cropped_image = img.copy()
    width, height, _ = cropped_image.shape
    cropped_image = cropped_image[100:width-100, 100:height-1]
    width, height, _ = cropped_image.shape
    top_left = img[0, 0, :]
    top_right = img[width-1, 0, :]
    bottom_left = img[0, height-1, :]
    bottom_right = img[width-1, height-1, :]
    most_frequent = np.argmax(np.bincount([np.sum(top_left), np.sum(top_right), np.sum(bottom_left), np.sum(bottom_right)]))

    if most_frequent == np.sum(top_left):
        return top_left
    
    elif most_frequent == np.sum(top_right):
        return top_right

    elif most_frequent == np.sum(bottom_left):
        return bottom_left
    
    elif most_frequent == np.sum(bottom_right):
        return bottom_right
    