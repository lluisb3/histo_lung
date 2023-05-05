import csv
import numpy as np


def create_folds(seq, k):
    avg = len(seq) / float(k)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out


def timer(start, end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    message = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)

    return message


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
    for level in level_downsamples:
        mags.append(magnification/level)

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
    copy = img.copy()
    width, height, _ = copy.shape
    if width > 15000 or height > 15000:
        cropped_image = copy[600:width-600, 600:height-600]
    else:
        cropped_image = copy[300:width-300, 300:height-300]
    width, height, _ = cropped_image.shape
    top_left = img[0, 0, :]
    top_right = img[width-1, 0, :]
    bottom_left = img[0, height-1, :]
    bottom_right = img[width-1, height-1, :]
    most_frequent = np.argmax(np.bincount([np.sum(top_left),
                                           np.sum(top_right),
                                           np.sum(bottom_left),
                                           np.sum(bottom_right)]))

    if most_frequent == np.sum(top_left):
        return top_left

    elif most_frequent == np.sum(top_right):
        return top_right

    elif most_frequent == np.sum(bottom_left):
        return bottom_left

    elif most_frequent == np.sum(bottom_right):
        return bottom_right
