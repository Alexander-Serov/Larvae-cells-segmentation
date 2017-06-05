


### Define support functions

## Define a function to initialize weights with random variables
def initialize_weights(shape, std, name):
	# weights = tf.truncated_normal(shape, stddev = std, dtype = tf.float32)
	# weights = tf.normal(shape, stddev = std, dtype = tf.float32)
	weights = tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer(), dtype = tf.float32)

	# return tf.Variable(weights)
	return weights
	
## Define a function to initialize the bias
def initialize_bias(shape):
	bias = tf.constant(0.1, shape = shape, dtype = tf.float32)
	return tf.Variable(bias)

## Define a 2D convolution function (filter)
def convolution2D(x, W):
	return tf.nn.conv2d(x, W, padding='SAME', strides = [1,1,1,1])	# Adds zero padding


def max_pool(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def central_crop_4d(x, cur_size, new_size):
	crop_size = int(math.floor((cur_size - new_size) / 2.0))
	return tf.slice(x, [0, crop_size, crop_size, 0], [-1, new_size, new_size, -1])


def get_mem_usage():                                                                                                             
    process = psutil.Process(os.getpid())
    return process.memory_info()  


## Image side size at the output of level i in the descending branch
def a_out(i):
	return (2 ** (levels_count - i)) * deepest_level_min_image_size + 2**3 * (2**(levels_count - i) - 1)


## Image side size at the input of level i in the descending branch
def a_in(i):
	return a_out(i) + 4


## Image side size at the output of level i in the ascending branch
def b_in(i):
	return 10 + 2**(levels_count - i) * (deepest_level_min_image_size - 6)


## Image side size at the output of level i in the ascending branch
def b_out(i):
	return b_in(i) - 4








