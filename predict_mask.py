

### Include packages
import math
import tensorflow as tf		# for machine learning
import numpy as np			# for math. operations
import os					# to index images in a folder
import resource				# for memory estimates
import sys 					# for memory estimates
import time    				# to measure elapsed time
# from tqdm import tqdm		# Progress bar
from PIL import Image		# to import TIFF images


### Load constants
execfile("./constants.py")
bl_predict = True


## Define external functions
execfile("./support_functions.py")
execfile("./network_layer_functions.py")


## Initialize
start_time = time.time()
original_image_side = a_in(1)
training_index = -1
print('Running on platform: %s' % (sys.platform))


## Load image files
# execfile("./count_labels.py")
execfile("./extend_image.py")
execfile("./rescale_image.py")
execfile("./load_input_files.py")
print('Full image size: %s. Elapsed time: %.2f s' % (full_image_sizes, time.time() - start_time))
execfile("./get_current_training_pair.py")
execfile("./make_predictions_on_a_large_image.py")
execfile("./convert_mask_to_RGB.py")


### Define the neural network
execfile("./u_net_like_network_structure.py")

## Variable saver
saver = tf.train.Saver()


## Start and initialize session
print('Starting a TensorFlow session... Elapsed time: %.2f s' % (time.time() - start_time))
sess = tf.Session()
## Load saved model if necessary. Else initialize variables
try:
	print("Loading saved model...")
	saver.restore(sess, save_path)
	print("Model loaded successfully!")
except:
	raise Exception("Model not loaded. Aborting")
		

## For each file in the input queue produce a prediction and output it
#for i in range(input_queue_length):
i=0
print('Elapsed time: %.2f s. Processing test file %i/%i...' % (time.time() - start_time, i+1, rotated_training_files_count))

# Load current test image
loaded_original_image, loaded_mask_image, training_index = get_current_training_pair(training_index, bl_predict, training_set_size)
print(loaded_original_image.shape)

cur_file = original_file_list[i]
# print(type(cur_prediction))
cur_prediction = make_predictions_on_a_large_image(sess, loaded_original_image, bl_predict, start_time)
cur_prediction = cur_prediction[0]

## Close session
sess.close()


# cur_prediction, cur_coarse_grained_mask, cost_eval, print_me_eval = sess.run((
# 	labels_prediction, saved_cur_mask_image, cost, print_me),
# 	feed_dict={
# 		input_original_image: loaded_original_image.astype(np.int32), 
# 		input_mask_image: loaded_mask_image.astype(np.int32), 
# 		keep_prob: keep_probability_predict})

# Reshape the output
# cur_coarse_grained_mask = cur_coarse_grained_mask[0]

## Extract the filename and extension
cur_filename, cur_file_extension = os.path.splitext(os.path.basename(cur_file))
#cur_filename = cur_filename[0]
# print()
# print(cur_filename)
print('File: %s' % (cur_filename))
# print('File: %s. Cost: %.3g' % (cur_filename, cost_eval))
# if not sum(sum(cur_prediction)) > 0:
# 	print('No features detected')
# np.set_printoptions(threshold = np.nan)
print(cur_prediction)
print(cur_prediction.shape)
# print(print_me_eval.shape)
# print(print_me_eval)
# print(np.sum(np.sum(cur_prediction)))

print('Saving rescaled original...')
# Output the prediction
output_full_filename = output_folder + cur_filename + '_rescaled' + cur_file_extension
output_image = Image.fromarray(loaded_original_image)
output_image.save(output_full_filename)
print('COMPLETED: Saving rescaled original. Elapsed time: %.2f s' % (time.time() - start_time))

print('Saving prediction...')
## Convert mask to an RGB image
cur_prediction_RGB = convert_mask_to_RGB(cur_prediction)
# Output the prediction
output_full_filename = output_folder + cur_filename + '_prediction' + cur_file_extension
output_image = Image.fromarray(cur_prediction_RGB)
output_image.save(output_full_filename)
print('COMPLETED: Saving prediction. Elapsed time: %.2f s' % (time.time() - start_time))
# print((cur_prediction * 255).astype(np.uint8))
# print((cur_coarse_grained_mask * 255).astype(np.uint8))

# Estimate memory usage
str_platform = sys.platform
if str_platform == 'darwin':
	factor_to_gigabytes = 2.0**30
else:
	factor_to_gigabytes = 2.0**20
memory_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss	

# Print out the final statistics
print("Time elapsed: %.2f s. Peak memory usage: %.3f Gb" % (time.time() - start_time, memory_peak/factor_to_gigabytes))

# # Output the mask
# output_full_filename = output_folder + cur_filename + '_coarse-grained_input_mask' + cur_file_extension
# output_image = Image.fromarray((cur_coarse_grained_mask * 255).astype(np.uint8))
# output_image.save(output_full_filename)

#(cur_original_filename.astype(np.string))
# print(os.path.splitext(tf.as_string(cur_original_filename)))











