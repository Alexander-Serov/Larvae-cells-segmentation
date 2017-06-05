


def convert_mask_to_RGB(mask):
	size = mask.shape
	RGB_image = np.zeros([size[0], size[1], 3], dtype = np.uint8)
	for x_ind in range(size[0]):
		for y_ind in range(size[1]):
			RGB_image[x_ind, y_ind, :] = mask_colors[mask[x_ind, y_ind]]

	return RGB_image





