import numpy as np


def eval_histogram_threshold(mask, thumb_data):
    
    thumb_data_masked = np.ma.array(thumb_data, mask=np.logical_not(mask))
    mean_thumb_data = thumb_data_masked.mean()
    print(f"Mean image within the mask of: {mean_thumb_data}")

    if mean_thumb_data <= 155:
        upper_thr = 210
        lower_thr = 35
    elif (mean_thumb_data>155 and mean_thumb_data<=180):
        upper_thr = 215
        lower_thr = 40
    elif (mean_thumb_data > 180):
        upper_thr = 220
        lower_thr = 45

    else:
        lower_thr = 40
        upper_thr = 215
    
    return lower_thr, upper_thr


def get_histogram(img, lower, upper):
	
	range_values = np.arange(lower,upper)
	histo_val = np.histogram(img, bins=range_values)[0]
	
	return histo_val
