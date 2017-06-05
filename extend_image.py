


def extend_image(input_image, bl_predict, small_window_side = None, big_window_side = None):
	## Initialize
	if small_window_side is None:
		small_window_side = b_out(1)
		big_window_side = a_in(1)
	input_image_size = input_image.shape
	# print(small_window_side, big_window_side)


	## Calculate the reflections necessary to process the image and extend the image
	x_mesh_length = input_image_size[0] / float(small_window_side)
	y_mesh_length = input_image_size[1] / float(small_window_side)
	# For training stay inside the original data, while for prediction make prediction for the whole domain
	# if bl_predict:
	x_mesh_length = int(math.ceil(x_mesh_length))
	y_mesh_length = int(math.ceil(y_mesh_length))
	# else:
	# 	x_mesh_length = int(math.floor(x_mesh_length))
	# 	y_mesh_length = int(math.floor(y_mesh_length))
	# print(x_mesh_length, y_mesh_length)	

	# x_mesh_length = int(math.ceil(input_image_size[0] / float(small_window_side)))
	# y_mesh_length = int(math.ceil(input_image_size[1] / float(small_window_side)))
	#
	# print(x_mesh_length, small_window_side, big_window_side, input_image_size[0], b_out(1), a_in(1))
	x_in_start = - int(math.floor(((x_mesh_length * small_window_side) - input_image_size[0]) / 2.0))
	y_in_start = - int(math.floor(((y_mesh_length * small_window_side) - input_image_size[1]) / 2.0))
	#
	x_in_end = x_in_start + x_mesh_length * small_window_side
	y_in_end = y_in_start + y_mesh_length * small_window_side
	#
	x_out_start = x_in_start - (big_window_side - small_window_side) / 2
	y_out_start = y_in_start - (big_window_side - small_window_side) / 2
	#
	x_out_end = x_in_end + (big_window_side - small_window_side) / 2
	y_out_end = y_in_end + (big_window_side - small_window_side) / 2

	# print(x_mesh_length, y_mesh_length)
	# print(input_image_size[0], small_window_side, x_in_start)

	## Extend the original image by 8 more around
	# image_ext = [] # np.zeros(input_image_size * 3, dtype = np.uint8)
	image_ext = np.concatenate((np.fliplr(np.flipud(input_image)), np.flipud(input_image), np.fliplr(np.flipud(input_image))), 1)
	image_ext = np.concatenate((image_ext, np.concatenate((np.fliplr(input_image), input_image, np.fliplr(input_image)), 1)), 0)
	image_ext = np.concatenate((image_ext, np.concatenate((np.fliplr(np.flipud(input_image)), np.flipud(input_image), np.fliplr(np.flipud(input_image))), 1)), 0)


	# Cut out only the central part that we will be processing
	# print(small_window_side)
	image_ext = image_ext[(input_image_size[0] + x_out_start):(input_image_size[0] + x_out_end), (input_image_size[1] + y_out_start):(input_image_size[1] + y_out_end)]
	## Extended image is ready
	# print(image_ext.shape)

	return [image_ext, big_window_side, small_window_side, input_image_size, x_mesh_length, y_mesh_length, x_out_start, y_out_start, x_in_start, y_in_start]
