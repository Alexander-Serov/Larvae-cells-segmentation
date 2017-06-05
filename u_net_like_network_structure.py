
	

# ## Define support functions
# execfile("./support_functions.py")



# ## Make file queues
# pair_file_list = tf.transpose(tf.stack([original_file_list, mask_file_list]))
# print(pair_file_list)

# pair_file_queue = tf.RandomShuffleQueue (
# 	capacity = queue_capacity,
# 	min_after_dequeue = min_after_dequeue, dtypes = ['string'], shapes = [[2]])
# pair_file_queue.enqueue_many(pair_file_list)
# global print_me
print_me = []
# # print_me = pair_file_queue.dequeue()
# # print(print_me)


# ## Get current filenames
# current_file_pair = pair_file_queue.dequeue()
# cur_original_filename = current_file_pair[0]
# cur_mask_filename = current_file_pair[1]


# ## Read files
# file_reader = tf.WholeFileReader()
# _, original_raw = file_reader.read(tf.train.string_input_producer([cur_original_filename]))
# _, mask_raw = file_reader.read(tf.train.string_input_producer([cur_mask_filename]))


# original_file_queue = tf.train.string_input_producer(original_file_list, shuffle = False);
# mask_file_queue = tf.train.string_input_producer(mask_file_list, shuffle = False);

# batch_names = tf.train.shuffle_batch(
# 	tensors = [original_file_list, mask_file_list], 
# 	batch_size = 1,
# 	num_threads = 1,
# 	capacity = queue_capacity,
# 	min_after_dequeue = min_after_dequeue)	


# ## Initialize image reader
# file_reader = tf.WholeFileReader()
#  # = original_file_queue[0]
# cur_original_filename, original_raw = file_reader.read(original_file_queue)
# cur_mask_filename, mask_raw = file_reader.read(mask_file_queue)

# ## Initialize image decoder
# original_image = tf.image.decode_png(original_raw)
# mask_image = tf.image.decode_png(mask_raw)


## Input images into the graph
image_side = expected_image_size_x
input_original_image = tf.placeholder(dtype = tf.float32, shape=(expected_image_size_x, expected_image_size_y), name = 'original')
input_mask_image = tf.placeholder(dtype = tf.int32, shape=(expected_image_size_x, expected_image_size_y), name = 'mask')



# ## The image must be reshaped, so that the tensor dimensions be set
# original_image = tf.reshape(original_image, [image_side ** 2])
# mask_image = tf.reshape(mask_image, [image_side ** 2])
## Convert the original image to float
# original_image = tf.cast(original_image, 'float32')

## Rescale the image to [0, 1]
# mean, var = tf.nn.moments(original_image, axes = [0])
# original_image = tf.sub(original_image, mean)
# image_side = original_image_side;

# Normalize the input image
tmp_mean, tmp_var = tf.nn.moments(input_original_image, axes = [0,1])
original_image = (input_original_image - tmp_mean) / tf.sqrt(tmp_var)


mask_image = input_mask_image


# ## Collecting garbage
# input_original_image = None
# input_mask_image = None


########

### ===== NETWORK LAYERS =====

### ==== DESCENDING PART ====
## Initialize
layer_counter = 0
level_counter = 1
features_count = []
level_out = []
level_out_side = []
# cur_original_image = batch_originals
layer_out = tf.reshape(original_image, [1, image_side, image_side, 1])
# Garbage collection
del original_image
# Save the original
# saved_cur_original_image = central_crop_4d(layer_out, image_side, new_size)


### == LEVEL 1 ==
# -- Convolution outputting 64 fetaure channels
layer_out, layer_counter, image_side, features_count = convolution_layer_3x3_change_features_number(
	layer_out, layer_counter, image_side, 1, features_count_first_layer)
# -- 3x3 convolution
layer_out, layer_counter, image_side = convolution_layer_3x3(layer_out, layer_counter, image_side)
# -- Saving level 1 out
level_out.extend([layer_out])
level_out_side.extend([image_side])
# -- Max. pooling
# print_me = layer_out
layer_out, layer_counter, image_side = max_pool_2x2(layer_out, layer_counter, image_side)
# layer_out, layer_counter, image_side = convolution_layer_2x2(layer_out, layer_counter, image_side)



### == INTERMEDIATE LEVELS ==
for level_counter in range(2, levels_count):
	# -- Convolution, double fetaure channels
	layer_out, layer_counter, image_side, features_count = convolution_layer_3x3_double_features(layer_out, layer_counter, image_side, features_count)

	# -- 3x3 convolution
	layer_out, layer_counter, image_side = convolution_layer_3x3(layer_out, layer_counter, image_side)

	# -- Saving level 2 out
	level_out.extend([layer_out])
	level_out_side.extend([image_side])

	# -- Max. pooling
	layer_out, layer_counter, image_side = max_pool_2x2(layer_out, layer_counter, image_side)
	# layer_out, layer_counter, image_side = convolution_layer_2x2(layer_out, layer_counter, image_side)


### == DEEPEST LEVEL ==
level_counter +=1
# -- Convolution, double fetaure channels)
# print_me = layer_out
layer_out, layer_counter, image_side, features_count = convolution_layer_3x3_double_features(layer_out, layer_counter, image_side, features_count)
# print_me = layer_out

# -- Dropout
layer_out, layer_counter = dropout(layer_out, layer_counter)
# -- 3x3 convolution
layer_out, layer_counter, image_side = convolution_layer_3x3(layer_out, layer_counter, image_side)
# -- Dropout
layer_out, layer_counter = dropout(layer_out, layer_counter)



### ==== ASCENDING PART ====


### == INTERMEDIATE LEVELS ==
for level_counter in range(levels_count - 1, 1, -1):
	# -- Upsample by a factor of 2, spatial dimensions *= 2
	layer_out, image_side = upsample(layer_out, layer_counter, image_side, features_count)
	# -- Convolution, reduce the number of features by two
	layer_out, layer_counter, image_side, features_count = convolution_layer_3x3_halve_features(layer_out, layer_counter, image_side, features_count)	
	# if level_counter is levels_count - 1:
		# global 
	# # print_me = layer_out # layer_out
	# layer_out, layer_counter, image_side, features_count = upconvolution_2x2(layer_out, layer_counter, image_side, features_count)
	# -- Combine result with a layer from the same level of the descending branch
	layer_out, features_count = combine_with_descending_branch(layer_out, level_out, level_out_side, level_counter, features_count)

	# -- Convolution, reduce the number of features by two
	layer_out, layer_counter, image_side, features_count = convolution_layer_3x3_halve_features(layer_out, layer_counter, image_side, features_count)
	# -- 3x3 convolution
	layer_out, layer_counter, image_side = convolution_layer_3x3(layer_out, layer_counter, image_side)
# print(print_me)

### == FIRST LEVEL ==
level_counter -=1
# -- Upsample by a factor of 2, spatial dimensions *= 2
layer_out, image_side = upsample(layer_out, layer_counter, image_side, features_count)
# -- Convolution, reduce the number of features by two
layer_out, layer_counter, image_side, features_count = convolution_layer_3x3_halve_features(layer_out, layer_counter, image_side, features_count)
# -- Combine result with a layer from the same level of the descending branch
layer_out, features_count = combine_with_descending_branch(layer_out, level_out, level_out_side, level_counter, features_count)
# -- Convolution, reduce the number of features by two
layer_out, layer_counter, image_side, features_count = convolution_layer_3x3_halve_features(layer_out, layer_counter, image_side, features_count)
# -- 3x3 convolution
layer_out, layer_counter, image_side = convolution_layer_3x3(layer_out, layer_counter, image_side)
# -- 1x1 convolution, produce the output features
layer_out, layer_counter, image_side, features_out = convolution_layer_1x1_change_features_number(layer_out, layer_counter, image_side, features_count, output_features_count)


# Collecting garbage
del level_out


### ==== OUTPUT IMAGE READY ====


## RESHAPE THE MASK IMAGE
# Crop the mask
cur_mask_image = mask_image
cur_mask_image = tf.reshape(mask_image, [1, original_image_side, original_image_side, 1])
cur_mask_image = central_crop_4d(cur_mask_image, image_side, image_side)
saved_cur_mask_image = tf.reshape(cur_mask_image, [image_side, image_side])
cur_mask_image = tf.reshape(cur_mask_image, [image_side**2])


# Collect garbage
del mask_image


## RESHAPE BOTH IMAGES TO HAVE SPACE IN FIRST DIMENSION AND LABELS IN THE SECOND
# Transpose first to keep the features together for each pixel
layer_out = tf.transpose(layer_out, [3, 1, 2, 0])
layer_out = tf.reshape(layer_out, [output_features_count, image_side**2])
layer_out = tf.transpose(layer_out)	# This gives space in the first dimension and features in the second
# # # layer_out = tf.reshape(layer_out, [image_side**2, output_features_count])



## DEFINE WEIGHTS AS INVERSE TO THE NUMBERS OF EACH LABEL IN THE INPUT IMAGE
## RESHAPE MASK TO HAVE PROBABILITY DISTRIBUTION ALONG THE SECOND DIMENSION

# Prepare weights to construct the weight matrix
max_frequency = np.max(labels_frequency)
bl_zero_frequency = np.equal(labels_frequency, 0)
# Add some ficticial frequency to provide weight for labels that do not exist in the training
tmp_weights = labels_frequency + max_frequency * bl_zero_frequency.astype(np.float)
# Calculate weights as reciprocals of the frequencies. All divisions should now be valid
tmp_weights = 1 / tmp_weights


# tmp_labels_frequency[0] = np.max(tmp_labels_frequency[1:])
# print_me = (labels_frequency, tmp_labels_frequency)
weights = tf.zeros([image_side**2])
mask_with_probabilities = [] # tf.zeros([image_side**2, output_features_count])
for label in range(output_features_count):
	found_labels = tf.cast(tf.equal(cur_mask_image, label), 'float32')
	found_labels_exp = tf.expand_dims(found_labels, 1)
	if label == 0:	
		mask_with_probabilities = found_labels_exp
		# weights = weights + tf.div(found_labels, 1 - average_mask_fraction)
		# print(mask_with_probabilities)
	else:
		mask_with_probabilities = tf.concat([mask_with_probabilities, found_labels_exp], 1)
		# weights = weights + tf.div(found_labels, average_mask_fraction)
	# labels_number = 1e-3 + tf.reduce_sum(found_labels)
	# weights = weights + found_labels / labels_number
	weights = weights + found_labels * tmp_weights[label]
	# Normalize
print('Using label weights: %s' % (tmp_weights))
# print_me = tmp_weights
	

# Collect garbage
del found_labels
del found_labels_exp


## MAKE LABELS PREDICTIONS FOR THE TEST RUN
labels_prediction = tf.nn.softmax(layer_out)
# layer_out_print = labels_prediction
# print_me = labels_prediction
# labels_prediction = tf.slice(labels_prediction, [0, 0], [-1, 1])
labels_prediction = tf.argmax(labels_prediction, axis = 1)
labels_prediction = tf.reshape(labels_prediction, [image_side, image_side])


## COST
cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = layer_out, labels = cur_mask_image)
# cost = tf.nn.softmax_cross_entropy_with_logits(logits = layer_out, labels = mask_with_probabilities)
# Make a weighted sum
cost = tf.multiply(cost, weights)
# cost = tf.reduce_sum(cost)
cost = tf.reduce_mean(cost) 



# Collect garbage
del layer_out
# del cur_mask_image
del weights
del mask_with_probabilities


## TRAINING STEP
training_step = tf.train.AdamOptimizer(learning_rate = learning_rate, beta1 = 0.99, beta2 = 0.999).minimize(cost)	
# training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)	
# training_step = tf.train.AdadeltaOptimizer(learning_rate).minimize(cost)	


## Variable initializer
init_op = tf.global_variables_initializer()














