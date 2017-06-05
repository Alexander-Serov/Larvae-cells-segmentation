


def count_labels(input_image):
	labels_count = np.zeros(output_features_count, np.int32)
	# label_mask = np.zeros(input_image.shape[0:2], dtype = np.uint8)
	bl_labelled_pixels = np.zeros(input_image.shape[0:2], dtype = bool)
	for label_ind in range(2, output_features_count):
		# Comparing the 3 channels with 2 ANDs
		bl_color_match = np.logical_and(np.equal(input_image[:, :, 0], mask_colors[label_ind] [0]), np.equal(input_image[:, :, 1], mask_colors[label_ind] [1]))
		bl_color_match = np.logical_and(bl_color_match, np.equal(input_image[:, :, 2], mask_colors[label_ind] [2]))
		# label_mask = label_mask + min([label_ind, 3]) * bl_color_match.astype(np.uint8)
		# Save processed pixels
		bl_labelled_pixels = np.logical_or(bl_labelled_pixels, bl_color_match)
		labels_count[label_ind] += np.sum(bl_color_match.astype(np.uint32))
	# Set remaining (white) pixels to 1
	# label_mask = label_mask + np.logical_not(bl_labelled_pixels).astype(np.uint8)
	labels_count[1] += np.sum(np.logical_not(bl_labelled_pixels).astype(np.uint32))
	return labels_count
