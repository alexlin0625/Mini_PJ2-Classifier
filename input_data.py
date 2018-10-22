#import library
import tensorflow as tf 
import numpy as np 
import os

#image size 
img_width = 208
img_height = 208

file_dir = "train"
def get_files(file_dir):
	
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

def get_batch(image, label, image_W, image_H, batch_size, capacity):
	
	#use cast to change the format of the data to tensorflow format
	image = tf.cast(image, tf.string)
	label = tf.cast(label, tf.int32)

	#make input queue (slice_input_producer args can be found from tensorflow tutorials)
	input_queue = tf.train.slice_input_producer([image, label])
 
	label = input_queue[1]
	image_contents = tf.read_file(input_queue[0])
	#since all images are in jpeg format, use decode_jpeg
	image = tf.image.decode_jpeg(image_contents, channels=3)

	#now we have to resize the photos using crop or pad (tensorflow API)
	image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
	image = tf.image.per_image_standardlization(image)

	image_batch, label_batch = tf.train.batch([image, label]), batch_size = batch_size, num_threads = 64, capacity = capacity)

	label_batch = tf.reshape(label_batch, [batch_size])

	return image_batch, label_batch
