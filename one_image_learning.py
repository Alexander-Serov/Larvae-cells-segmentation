

### Include packages
import math
import ntpath				# for filename extractions
import numpy as np			# for math. operations
import resource				# for memory estimates
import sys 					# for memory estimates
import tensorflow as tf		# for machine learning
import time    				# to measure elapsed time
import os					# to index images in a folder
# import psutil				# to identify processes and measure memory
from PIL import Image		# to import images in numpy
from PIL import ImageOps	# to flip images
from scipy import interpolate	# used for elastic deformations


### Load constants
execfile("./constants.py")
bl_predict = False


## Define external functions
execfile("./support_functions.py")
execfile("./network_layer_functions.py")


## Initialize
start_time = time.time()
original_image_side = a_in(1)
training_index = -1
print('Running on platform: %s' % (sys.platform))
if bl_apply_elastic_deformations:
	print('This training will use elastic deformations: Yes')
else:
	print('This training will use elastic deformations: No')


### Create a save path if necessary
if not os.path.exists(save_path):
    os.makedirs(save_path)

## Set the same random seed		`
# np.random.seed(3)
# tf.set_random_seed(1)


## Load image files
execfile("./extend_image.py")
execfile("./count_labels.py")
execfile("./rescale_image.py")
execfile("./load_input_files.py")
print original_image_size
print('Original image size: %s px' % (original_image_size, ))
print('Rescaled image size: %s px' % (rescaled_image_size, ))
print('Extended image size: %s px' % (full_image_sizes, ))
print('Input batch size: %ix%i px' % (original_image_side, original_image_side))
print('Output batch size: %ix%i px' % (small_window_side, small_window_side))
print('Non-distorted training samples available: %i' % (training_set_size))
execfile("./get_current_training_pair.py")
execfile("./interpolate_2D.py")
execfile("./apply_random_elastic_deformation.py")
execfile("./convert_mask_to_RGB.py")


### Define the neural network
execfile("./u_net_like_network_structure.py")


## Variable saver
saver = tf.train.Saver()

### Fit the model
# Create a session and initialize the model

# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess = tf.Session()

## Load saved model if necessary. Else initialize variables
bl_loaded = False
if bl_load_saved_model:
	print("Loading saved model...")
	try:
		saver.restore(sess, save_path)
		print("Model loaded successfully!")
		bl_loaded = True
	except:
		print("Model not loaded. Resetting model parameters")
		
if not bl_loaded:
	sess.run(init_op)
sess.graph.finalize()  # Graph is read-only after this statement.

## Start the queue feeders which will supply input files
# queue_threads = tf.train.start_queue_runners(sess)
# print(sess.run(result_convolution_1))
# print([original_file_list[j] for j in range(2)])
for i in range(int(training_steps_number)):
	## Load current training files
	if i < training_set_size or bl_apply_elastic_deformations:
		loaded_original_image, loaded_mask_image, training_index = get_current_training_pair(training_index, bl_predict, training_set_size)
	else:
		print('Training step: %5i/%i. Out of training files!' % (i+1, int(training_steps_number)))
		break
	# loaded_original_image = np.array(Image.open(original_file_list[ind]), dtype = np.float32)
	# loaded_mask_image = np.array(Image.open(mask_file_list[ind]), dtype = np.int32)
	# print(shape(loaded_mask_image))

	# ## == Test section ==
	# [distorted_image, distorted_mask] = apply_random_elastic_deformation(loaded_original_image, loaded_mask_image)
	# # Save original
	# output_full_filename = output_folder + '01_original.png'
	# output_image =  Image.fromarray(loaded_original_image)
	# output_image.save(output_full_filename)
	# # Save distorted original
	# output_full_filename = output_folder + '02_distorted_original.png'
	# output_image =  Image.fromarray(distorted_image)
	# output_image.save(output_full_filename)
	# # Save mask
	# output_full_filename = output_folder + '03_mask.png'
	# output_image =  Image.fromarray(convert_mask_to_RGB(loaded_mask_image))
	# output_image.save(output_full_filename)
	# # Save distorted mask
	# output_full_filename = output_folder + '04_distorted_mask.png'
	# output_image =  Image.fromarray(convert_mask_to_RGB(distorted_mask))
	# output_image.save(output_full_filename)
	# ## == End of the test section ==

	## Training step
	_, cost_eval, cur_mask_out, print_me_eval = sess.run([training_step, cost, cur_mask_image, print_me], feed_dict={
		input_original_image: loaded_original_image.astype(np.int32), 
		input_mask_image: loaded_mask_image.astype(np.int32), 
		keep_prob: keep_probability_train})
	# print(print_me_eval)
	## Visualize
	if i==0 or (not i % visualization_step):
		# mem = get_mem_usage().rss
		_, filename= ntpath.split(original_file_list[ind])
		filename = ntpath.basename(filename)
		# print(print_me_eval)
		print('Time: %.2f s. Training step: %5i/%i. Training index: %i, cost: %g' % (time.time() - start_time, i+1, int(training_steps_number), training_index, cost_eval))
		# print(cur_mask_out)
	
	## Save session
	if not (i+1) % save_step:
		print("Saving session...")
		saver.save(sess, save_path)
		print("Time: %.2f s. Session saved." % (time.time() - start_time))	

	


## Save session
print("Saving session...")
saver.save(sess, save_path)
print("Session saved in %s" % save_path)

# Estimate memory usage
str_platform = sys.platform
if str_platform == 'darwin':
	factor_to_gigabytes = 2.0**30
else:
	factor_to_gigabytes = 2.0**20
memory_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss	

# Print out the final statistics
print("Time: %.2f s. Peak memory usage: %.3f Gb" % (time.time() - start_time, memory_peak/factor_to_gigabytes))
# print(memory_peak, factor_to_gigabytes)
# print (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
## Close session
sess.close()

# ## Make predictions at once
# def predict():
# 	execfile("./predict_mask.py")
# predict()

# print(mask_image)
# print(sess.run(mask_image))
# print(sess.run(
# 	cur_mask_image_reshaped))
# print(sess.run(cur_mask_filename))
# print(mask_file_list)


# print(sess.run(tf.sub(scaled_output, tf.cast(mask_image, tf.float16))))


## Train the model


# print(first_original_file, first_mask_file)

# print(sess.run(cost))
#print(sess.run(result_convolution_1))
#print(sess.run(var))
# print(sess.run(scaled_output))
# print(sess.run(result_convolution_1))


## Populate the filename queue
## originals_queue_coordinator = tf.train.Coordinator()







### Evaluate model accuracy
## Apply the model to one of the files and output the mask

# print(sess.run(result_convolution_1, feed_dict={
# 	original_file_list: [first_original_file],
# 	mask_file_queue: tf.train.string_input_producer([first_mask_file], shuffle = False)}))











