import sys, os
import openslide
from PIL import Image
import numpy as np
import pandas as pd 
from collections import Counter
from matplotlib import pyplot as plt
from skimage import io
import threading
import time
from skimage import exposure
import json
import multiprocessing
import PIL
from pathlib import Path

thispath = Path(__file__).resolve()

PIL.Image.MAX_IMAGE_PIXELS = 93312000000

np.random.seed(0)

# folder_files_labels = '/home/niccolo/ExamodePipeline/Multiple_Instance_Learning/Colon/csv_folder/global_labels/'
# csv_labels_filepath = folder_files_labels+'colon_radboudc_v2_multilabel.csv'

# csv_multiclass = pd.read_csv(csv_labels_filepath, sep=',', header=None).values

THREAD_NUMBER = 5
lockList = threading.Lock()
lockGeneralFile = threading.Lock()

def create_output_imgs(img, fname):
	#save file
	patch_size = 224
	img = img.resize((patch_size, patch_size))
	img = np.asarray(img)
	#io.imsave(fname, img)
	print("file " + str(fname) + " saved")

def check_background(glimpse, threshold, GLIMPSE_SIZE_SELECTED_LEVEL):
	b = False

	window_size = GLIMPSE_SIZE_SELECTED_LEVEL
	tot_pxl = window_size * window_size
	white_pxl = np.count_nonzero(glimpse)
	score = white_pxl/tot_pxl
	if (score>=threshold):
		b=True
	return b

def write_coords_local_file(fname, arrays):
		#select path
	output_dir = PATH_OUTPUT+'/'+fname+'/'
	filename_path = output_dir+fname+'_coords_densely.csv'
		#create file
	File = {'filename':arrays[0],'level':arrays[1],'x_top':arrays[2],'y_top':arrays[3]}
	df = pd.DataFrame(File,columns=['filename','level','x_top','y_top'])
		#save file
	np.savetxt(filename_path, df.values, fmt='%s',delimiter=',')

# def write_paths_local_file(fname,listnames):
# 	output_dir = PATH_OUTPUT+'/'+fname+'/'
# 	filename_path = output_dir+fname+'_paths_densely.csv'
# 		#create file
# 	File = {'filenames':listnames}
# 	df = pd.DataFrame(File,columns=['filenames'])
# 		#save file
# 	np.savetxt(filename_path, df.values, fmt='%s',delimiter=',')

# def multi_one_hot_enc(current_labels):
# 	labels = [0,0,0,0,0]
# 	for i in range(len(current_labels)):
# 		labels[current_labels[i]]=1
# 	return labels

def eval_whitish_threshold(mask, thumb):
	a = np.ma.array(thumb, mask=np.logical_not(mask))
	mean_a = a.mean()

	THRESHOLD = 200.0

	if (mean_a<=155):
		THRESHOLD = 195.0
	elif (mean_a>155 and mean_a<=180):
		THRESHOLD = 200.0
	elif (mean_a>180):
		THRESHOLD = 205.0
	return THRESHOLD

def whitish_img(img, THRESHOLD_WHITE):
	b = True
	if (np.mean(img) > THRESHOLD_WHITE):
		b = False
	return b

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

def find_vertical_and_analyze(wsi_np):
	pixel_stride = 10
	THRESHOLD = 0.99
	b = False
	
	half = int(wsi_np.shape[1]/2)
	h1 = half-pixel_stride
	h2 = half+pixel_stride
	   
	central_section = wsi_np[:,h1:h2]
	
	tot_surface = central_section.shape[0]*central_section.shape[1]
	
	unique, counts = np.unique(central_section, return_counts=True)

	if (len(counts)==1):
		b = True
	elif (counts[0]>THRESHOLD*tot_surface):
		b = True
	return b

def left_or_right(img):
	half = int(img.shape[1]/2)
	left_img = img[:,:half]

	right_img = img[:,half:]

	unique, counts_left = np.unique(left_img, return_counts=True)
	unique, counts_right = np.unique(right_img, return_counts=True)

	b = None

	if (len(counts_left)<len(counts_right)):
		b = 'right'
	elif(len(counts_left)>len(counts_right)):
		b = 'left'
	else:
		if (counts_left[1]>counts_right[1]):
			b = 'left'
		else:
			b = 'right'

	return b

def find_horizontal_and_analyze(wsi_np):
	pixel_stride = 10
	THRESHOLD = 0.99
	b = False
	
	half = int(wsi_np.shape[0]/2)
	h1 = half-pixel_stride
	h2 = half+pixel_stride
	
	central_section = wsi_np[h1:h2,:]
	
	#tot_surface = 2*pixel_stride*wsi_np.shape[0]
	tot_surface = central_section.shape[0]*central_section.shape[1]
	
	unique, counts = np.unique(central_section, return_counts=True)

	#return central_section
	
	if (len(counts)==1):
		b = True
	elif (counts[0]>THRESHOLD*tot_surface):
		b = True
	return b

def up_or_down(img):
	
	half = int(img.shape[0]/2)
	up_img = img[:half,:]
	down_img = img[half:,:]

	unique, counts_up = np.unique(up_img, return_counts=True)
	unique, counts_down = np.unique(down_img, return_counts=True)


	b = None

	if (len(counts_up)<len(counts_down)):
		b = 'down'
	elif(len(counts_up)>len(counts_down)):
		b = 'up'
	else:
		if (counts_up[1]>counts_down[1]):
			b = 'up'
		else:
			b = 'down'

	return b

#estrae glimpse e salva metadati relativi al glimpse
def analyze_file(filename, patch_size, PATH_INPUT_MASKS, MAGNIFICATION, PATH_OUTPUT):

	patches = []
	
	file = openslide.OpenSlide(str(filename))
	mpp = file.properties['openslide.mpp-x']

	level_downsamples = file.level_downsamples
	mags = available_magnifications(mpp, level_downsamples)

	level = 0

		#load mask
	fname = os.path.split(filename)[-1]
		#check if exists
	fname_mask = Path(PATH_INPUT_MASKS / filename.parent.stem / filename.stem)

	array_dict = []

		#levels for the conversion
	WANTED_LEVEL = MAGNIFICATION
	MASK_LEVEL = 1.25
	HIGHEST_LEVEL = mags[0]
	#AVAILABLE_LEVEL = select_nearest_magnification(WANTED_LEVEL, mags, level_downsamples)
	
	RATIO_WANTED_MASK = WANTED_LEVEL/MASK_LEVEL
	RATIO_HIGHEST_MASK = HIGHEST_LEVEL/MASK_LEVEL

	WINDOW_WANTED_LEVEL = patch_size

	GLIMPSE_SIZE_SELECTED_LEVEL = WINDOW_WANTED_LEVEL

	GLIMPSE_SIZE_MASK = np.around(GLIMPSE_SIZE_SELECTED_LEVEL/RATIO_WANTED_MASK)
	GLIMPSE_SIZE_MASK = int(GLIMPSE_SIZE_MASK)

	GLIMPSE_HIGHEST_LEVEL = np.around(GLIMPSE_SIZE_MASK*RATIO_HIGHEST_MASK)
	GLIMPSE_HIGHEST_LEVEL = int(GLIMPSE_HIGHEST_LEVEL)
	"""
	print("GLIMPSE_SIZE_MASK " + str(GLIMPSE_SIZE_MASK))
	print("GLIMPSE_HIGHEST_LEVEL " + str(GLIMPSE_HIGHEST_LEVEL))
	print("RATIO_WANTED_MASK " + str(RATIO_WANTED_MASK))
	print("RATIO_HIGHEST_MASK " + str(RATIO_HIGHEST_MASK))
	"""
	STRIDE_SIZE_MASK = 0
	TILE_SIZE_MASK = GLIMPSE_SIZE_MASK+STRIDE_SIZE_MASK

	PIXEL_THRESH = 0.5
	
	if os.path.isfile(fname_mask):
        output_dir = Path(PATH_OUTPUT / fname)
        Path(output_dir).mkdir(exist_ok=True, parents=True)

			#create CSV file structure (local)
		filename_list = []
		level_list = []
		x_list = []
		y_list = []

		img = Image.open(fname_mask)

		thumb = file.get_thumbnail(img.size)
		thumb = thumb.resize(img.size)
		mask_np = np.asarray(thumb)

		img = np.asarray(img)

		mask_3d = np.repeat(img[:, :, np.newaxis], 3, axis=2)
		
		WHITISH_THRESHOLD = eval_whitish_threshold(mask_3d, mask_np)

		mask_np = np.asarray(img)

		cont = 0
		
		unique, counts = np.unique(mask_np, return_counts=True)
		#print(unique, counts, len(unique)>1)

		timeout_start = time.time()
		timeout = 300

		if (len(unique)>1):

			start_X = 0
			start_Y = 0
			end_X = int(mask_np.shape[1])
			end_Y = int(mask_np.shape[0]) 

			n_image = 0
			threshold = PIXEL_THRESH

			y_ini = start_Y + STRIDE_SIZE_MASK
			y_end = y_ini + GLIMPSE_SIZE_MASK

			#print(mask_np.shape, end_X, end_Y)

			a = cont<100000

			while(y_end<end_Y and a==True):
				
				a = cont<1000000

				x_ini = start_X + STRIDE_SIZE_MASK
				x_end = x_ini + GLIMPSE_SIZE_MASK
				
				while(x_end<end_X and a==True ):
					glimpse = mask_np[y_ini:y_ini+GLIMPSE_SIZE_MASK,x_ini:x_ini+GLIMPSE_SIZE_MASK]

					check_flag = check_background(glimpse,threshold,TILE_SIZE_MASK)
					
					if(check_flag):

						fname_patch = output_dir+'/'+fname+'_'+str(n_image)+'.png'
							#change to magnification 40x
						x_coords_0 = int(x_ini*RATIO_HIGHEST_MASK)
						y_coords_0 = int(y_ini*RATIO_HIGHEST_MASK)
							
						file_40x = file.read_region((x_coords_0,y_coords_0),level,(GLIMPSE_HIGHEST_LEVEL,GLIMPSE_HIGHEST_LEVEL))
						file_40x = file_40x.convert("RGB")
						
						save_im = file_40x.resize((patch_size,patch_size))
						save_im = np.asarray(save_im)	

						bool_white = whitish_img(save_im,WHITISH_THRESHOLD)
						#bool_white = True
						bool_exposure = exposure.is_low_contrast(save_im)

						if (bool_white):
							if bool_exposure==False:
							#if (exposure.is_low_contrast(save_im)==False):

								io.imsave(fname_patch, save_im)
								
								#add to arrays (local)
								filename_list.append(fname_patch)
								level_list.append(level)
								x_list.append(x_coords_0)
								y_list.append(y_coords_0)
								n_image = n_image+1
								#save the image
								#create_output_imgs(file_10x,fname)
							else:
								pass
								#print("low_contrast " + str(output_dir))

					x_ini = x_end + STRIDE_SIZE_MASK
					x_end = x_ini + GLIMPSE_SIZE_MASK

					cont = cont + 1
					"""
					if (cont%1000==0):
						print("cont " + str(cont) + ", " + str(fname))
					"""
				y_ini = y_end + STRIDE_SIZE_MASK
				y_end = y_ini + GLIMPSE_SIZE_MASK
			
				#add to general arrays
			if (n_image!=0):
				#lockGeneralFile.acquire()
				#filename_list_general.append(output_dir)

				print("WSI done: " + filename)
				#print("len filename " + str(len(filename_list_general)) + "; WSI done: " + filename)
				print("extracted " + str(n_image) + " patches")
				#lockGeneralFile.release()
				write_coords_local_file(fname,[filename_list,level_list,x_list,y_list])
				# write_paths_local_file(fname,filename_list)
			else:
				print("ZERO OCCURRENCIES " + str(output_dir) + " NO GOOD PATCHES, cont " + str(cont))

		else:
			print("ZERO OCCURRENCIES " + str(output_dir) + " MASK TOTAL BLACK")

	else:
		print("no mask " + str(fname))

def explore_list(list_dirs, patch_size, PATH_INPUT_MASKS, MAGNIFICATION, PATH_OUTPUT):
	global list_dicts, n

	for i in range(len(list_dirs)):
		try:
			analyze_file(list_dirs[i], patch_size, PATH_INPUT_MASKS, MAGNIFICATION, PATH_OUTPUT)
		except Exception as e:

			print(e)
			pass


#list of lists fname-bool
def create_list_dicts(filenames,gs,ps):
	n_list = []
	for (f,g,p)in zip(filenames,gs,ps):
		dic = {"filename":f,"primary_GG":g,"secondary_GG":p,"state":False}
		n_list.append(dic)
	return n_list

def chunker_list(seq, size):
		return (seq[i::size] for i in range(size))

def main():
	#create output dir if not exists
	start_time = time.time()

		#create CSV file structure (global)
	filename_list_general = []

	MAGNIFICATION = 10
	new_patch_size = 224

	datadir = Path("/mnt/nas4/datasets/ToReadme/ExaMode_Dataset1/AOEC")


	PATH_INPUT_MASKS = Path(thispath.parent.parent / "data" / "Mask_PyHIST")

	PATH_OUTPUT = Path(thispath.parent.parent / 'data' / 'Patch_Extractor' / f"magnification {MAGNIFICATION}x")
	Path(PATH_OUTPUT).mkdir(exist_ok=True, parents=True)

	svs_files = [i for i in datadir.rglob("*.svs") if "LungAOEC" in str(i)]

	labels = pd.read_csv(Path(thispath.parent.parent / "data" / "lung_data" / "he_images.csv"))
	names = labels["file_name"].values

	he_svs_files = []
	for name in names:
		for file in svs_files:
			if file.stem in name:
				he_svs_files.append(file)

	#split in chunks for the threads
	he_svs_files = list(chunker_list(he_svs_files, THREAD_NUMBER))
	print(len(he_svs_files))

	threads = []
	for i in range(THREAD_NUMBER):
		#t = multiprocessing.Process(target=explore_list,args=([list_dirs[i]]))
		t = threading.Thread(target=explore_list, args=(he_svs_files[i], new_patch_size, PATH_INPUT_MASKS, MAGNIFICATION, PATH_OUTPUT))
		threads.append(t)

	for t in threads:
		t.start()
		#time.sleep(60)

	for t in threads:
		t.join()
	
	elapsed_time = time.time() - start_time
	print("elapsed time " + str(elapsed_time))


if __name__ == "__main__":
	main()
