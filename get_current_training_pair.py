


def get_current_training_pair(training_index, bl_predict, training_set_size):

	## Only load images that contain masks starting from number 2
	if bl_predict:
		training_index += 1
		original = full_loaded_rotated_originals[training_index]
		mask = np.zeros(1)

	else:
		bl_has_labels = False
		while(not bl_has_labels):
			training_index += 1
			# Reset the counter if reached the end of files, but using elastic deformations
			if training_index >= training_set_size:
				if not bl_apply_elastic_deformations:
					raise NameError('Error: Training index outside range')
				else:
					training_index -= training_set_size

			im_ind, x_start, y_start = deformation_parameters[training_index]
			mask =  full_loaded_rotated_masks[im_ind][x_start:(x_start + expected_image_size_x), y_start:(y_start + expected_image_size_y)]
			# Check if the masks contains labels larger than 1
			bl_has_labels = np.sum(np.greater(mask, 1).astype(np.uint16)) > 0 or (not bl_get_only_images_with_mask)
			
		original =  full_loaded_rotated_originals[im_ind][x_start:(x_start + expected_image_size_x), y_start:(y_start + expected_image_size_y)]

		## Apply random deformations
		if bl_apply_elastic_deformations:
			[original, mask] = apply_random_elastic_deformation(original, mask)


	return [original, mask, training_index]
















