#import library
import tensorflow as tf 
import numpy as np 
import os

#image size 
img_width = 208
img_height = 208

file_dir = '/Users/AlexLin/Desktop/EC601/Tensorflow\ miniproject\ 2/data/train'

def get_files(file_dir):
	'''
	Args: 
		file_dir: file directory
	Returns:
		list of images and labels
	'''
	cats = []
	label_cats = []
	dogs = []
	label_dogs = []
	#use os.listdir(file_dir) to return all the images inside the file_dir
	for file in os.listdir(file_dir):
		#since the data image is named as 'cat.0.jpg', we use split(sep='.') to seperate 
		name = file.split(sep='.')
		#if cat, set it to 0, dogs to 1
		if name[0]=='cat':
			cats.append(file_dir + file)
			label_cats.append(0)
		else: 
			dogs.append(file_dir + file)
			label_dogs.append(1)
	print (len(cats), len(dogs))

	image_list = np.hstack((cats, dogs))
	label_list = np.hstack((label_cats, label_dogs))

	#shuffle the images to rearrage the image orders
	temp = np.array([image_list, label_list])
	temp = temp.transpose()
	np.random.shuffle(temp)

	image_list = list(temp[:, 0])
	label_list = list(temp[:, 1])
	label_list = [int(i) for i in label_list]

	return image_list, label_list 

image_list, label_list = get_files(train_dir)




