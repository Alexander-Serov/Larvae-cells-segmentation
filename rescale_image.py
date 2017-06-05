


def rescale_image(image, method):
	# Detect original image size
	# print image.size
	# print type(image.size)
	# a = image.size
	# print type((2,3))
	# size_x, size_y = a
	size_x, size_y = image.size


	# Calculate new size
	new_size_x = rescale_new_x_side
	resize_factor = new_size_x / float(size_x)
	new_size_y = int(np.ceil(size_y * resize_factor))

	# Generate error if the requested size is greater than the original. This function only reduces size!
	if new_size_x > size_x:
		raise ValueError('Rescaling error: requested image side is larger than the original!')


	# Resize
	rescaled_image = image.resize((new_size_x, new_size_y), method)
	# print 'ok'


	return rescaled_image














