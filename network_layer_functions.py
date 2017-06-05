


## Initialize
keep_prob = tf.placeholder(tf.float32)


## 3x3 convolution changing the number of features
def convolution_layer_3x3_change_features_number(layer_in, layer_counter, image_side, features_in, features_out):
	layer_counter+=1
	weight_name = "weights_CONV_%i" % (layer_counter)
	N = 3**2 * features_in
	weights_convolution = initialize_weights([3, 3, features_in, features_out], math.sqrt(2/N), weight_name)
	bias_convolution = initialize_bias([1])
	layer_out = tf.nn.conv2d(layer_in, weights_convolution, padding='VALID', strides = [1,1,1,1])	# No padding
	layer_out = layer_out + bias_convolution
	layer_out = tf.nn.relu(layer_out)
	image_side = image_side - 2
	## Weights: [convolution_filter_size_1st_layer, convolution_filter_size_1st_layer, 1, features_count_first_layer]
	## Output: [1, image_side-2, image_side-2, features_count]
	return [layer_out, layer_counter, image_side, features_out]


## 3x3 convolution, double fetaure channels
def convolution_layer_3x3_double_features(layer_in, layer_counter, image_side, features_count):
	return convolution_layer_3x3_change_features_number(layer_in, layer_counter, image_side, features_count, features_count * 2)

## 3x3 convolution, halve fetaure channels
def convolution_layer_3x3_halve_features(layer_in, layer_counter, image_side, features_count):
	return convolution_layer_3x3_change_features_number(layer_in, layer_counter, image_side, features_count, features_count / 2)		


## 3x3 convolution, no feature change
def convolution_layer_3x3(layer_in, layer_counter, image_side):
	layer_out, layer_counter, image_side, _ = convolution_layer_3x3_change_features_number(layer_in, layer_counter, image_side, features_count, features_count)
	return [layer_out, layer_counter, image_side]


## 2x2 convolution, no feature change, reduce image side by 2
def convolution_layer_2x2(layer_in, layer_counter, image_side, features_count):
	layer_counter+=1
	weight_name = "weights_CONV_%i" % (layer_counter)
	# The filter size must be 4 to allow upsampling with stride 2
	N = 2**2 * features_count
	weights_convolution = initialize_weights([2, 2, features_count, features_count], math.sqrt(2/N), weight_name)
	bias_convolution = initialize_bias([1])
	layer_out = tf.nn.conv2d(layer_in, weights_convolution, padding='VALID', strides = [1,2,2,1])	# No padding
	# layer_out = layer_out + bias_convolution
	# layer_out = tf.nn.relu(layer_out)
	image_side = image_side/2
	## Weights: [convolution_filter_size_1st_layer, convolution_filter_size_1st_layer, 1, features_count_first_layer]
	## Output: [1, image_side-2, image_side-2, features_count]
	return [layer_out, layer_counter, image_side]


# ## 2x2 convolution, halve features
# def convolution_layer_2x2_halve_features(layer_in, layer_counter, image_side):
# 	layer_counter+=1
# 	weight_name = "weights_CONV_%i" % (layer_counter)
# 	# The filter size must be 4 to allow upsampling with stride 2
# 	weights_convolution = initialize_weights([4, 4, features_count, features_count], weight_name)
# 	bias_convolution = initialize_bias([1])
# 	layer_out = tf.nn.conv2d(layer_in, weights_convolution, padding='VALID', strides = [1,2,2,1])	# No padding
# 	layer_out = layer_out + bias_convolution
# 	layer_out = tf.nn.relu(layer_out)
# 	image_side = image_side/2
# 	## Weights: [convolution_filter_size_1st_layer, convolution_filter_size_1st_layer, 1, features_count_first_layer]
# 	## Output: [1, image_side-2, image_side-2, features_count]
# 	return [layer_out, layer_counter, image_side]	


## 1x1 convolution changing the number of features
def convolution_layer_1x1_change_features_number(layer_in, layer_counter, image_side, features_in, features_out):
	layer_counter+=1
	weight_name = "weights_CONV_%i" % (layer_counter)
	N = features_in
	weights_convolution = initialize_weights([1, 1, features_in, features_out], math.sqrt(2/N), weight_name)
	bias_convolution = initialize_bias([1])
	layer_out = tf.nn.conv2d(layer_in, weights_convolution, padding='VALID', strides = [1,1,1,1])	# No padding
	# layer_out = layer_out + bias_convolution
	# layer_out = tf.nn.relu(layer_out)
	return [layer_out, layer_counter, image_side, features_out]	


## Max. pool 2x2
def max_pool_2x2(layer_in, layer_counter, image_side):
	## Layer (max. pooling)
	layer_counter+=1
	# weight_name = "weights_CONV_%i" % (layer_counter)
	# weights_convolution = initialize_weights([convolution_filter, convolution_filter, 1, features_count], weight_name)
	# bias_convolution = initialize_bias([1])
	layer_out = tf.nn.max_pool(layer_in, ksize=[1, max_pool_size, max_pool_size, 1], strides=[1, max_pool_size, max_pool_size, 1], padding='VALID')
	image_side /= 2
	## Weights: [convolution_filter_size_1st_layer, convolution_filter_size_1st_layer, 1, features_count_first_layer]
	## Output: [1, image_side-2, image_side-2, features_count]
	return [layer_out, layer_counter, image_side]


## Dropout
def dropout(layer_in, layer_counter):
	layer_counter+=1
	layer_out = tf.nn.dropout(layer_in, keep_prob)
	return [layer_out, layer_counter]


## Upsample by a factor of 2 independently for each feature. No relu
def upsample(layer_in, layer_counter, image_side, features_count):
	layer_counter += 1 
	# First perform the upsampling channel by channel
	filter_size = 4
	upsampling_kernel_2D = get_upconvolution_2x2_filter()
	upsampling_kernel_4D = np.zeros((filter_size, filter_size, features_count, features_count), dtype = 'float32')
	for feature_ind in range(features_count):
		upsampling_kernel_4D[:, :, feature_ind, feature_ind] = upsampling_kernel_2D
	# Upsample the input layer	
	upsampled_layer = tf.nn.conv2d_transpose(layer_in, upsampling_kernel_4D, output_shape = [1, image_side * 2, image_side * 2, features_count], strides = [1, 2, 2, 1], padding = 'SAME')
	image_side *= 2
	return [upsampled_layer, image_side]


## Prepare the upconvolution filter in 2 dimensions
def get_upconvolution_2x2_filter():
	factor = 2
	filter_size = 4    # because the scale factor is 2. 
	center = filter_size / 2.0 - 0.5
	x, y = np.ogrid[0:filter_size, 0:filter_size]
	# Calculate a 4x4 bilinear resampling filter
	weights_per_channel = (1 - abs(x - center) / factor) * (1 - abs(y - center) / factor)
	# weights_per_channel /= sum(sum(weights_per_channel))    # To keep the sum of coefficients to one
	return weights_per_channel


## Combine layer with the same level of the descending branch
def combine_with_descending_branch(layer_in, level_out, level_out_side, level_counter, features_count):
	cropped_descending = level_out[level_counter - 1]
	cur_size = level_out_side[level_counter - 1]
	cropped_descending = central_crop_4d(cropped_descending, cur_size, image_side)
	layer_out = tf.concat([cropped_descending, layer_in], 3)
	features_count *= 2
	return [layer_out, features_count]












