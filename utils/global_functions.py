import csv

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