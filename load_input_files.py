


## Initialize
expected_image_size_x = a_in(1) 
expected_image_size_y = a_in(1) 
expected_image_size = (a_in(1), a_in(1))


if not bl_predict:
	originals_folder = train_originals_folder
	masks_folder = train_masks_folder
else:
	originals_folder = test_originals_folder
	masks_folder = test_masks_folder


#### ==== PREPARE FILE LISTS ====
print('Reading input file list...')
### Load training images
## Read file list of originals
## Make sure the files have the correct extensions
file_list = sorted(os.listdir(originals_folder))
original_file_list = []
first_original_file = []
for file in file_list:
	# print(file)
	filename, file_extension = os.path.splitext(file)
	if not file_extension == FILE_EXTENSION:
		continue
	## Save the filename of the first file 	
	if first_original_file == []:
		first_original_file = file

	# Use repeat to work on the same file
	# for repeat in range(training_steps_number):
	original_file_list.append(originals_folder + file)


# ## Check image size
# test_image = Image.open(original_file_list[0])
# test_image_size = test_image.size
# expected_image_size = (a_in(1), a_in(1))
# if test_image_size != expected_image_size:
# 	print('Error: Wrong input image size. Expected: %ix%i px' % (a_in(1), a_in(1)))
# 	exit()	
	

## Read file list of masks
## Make sure the files have the correct extensions
file_list = sorted(os.listdir(masks_folder))
mask_file_list = []
first_mask_file = []
for file in file_list:
	filename, file_extension = os.path.splitext(file)
	if not file_extension == FILE_EXTENSION:
		continue

	## Save the filename of the first file 	
	if first_mask_file == []:
		first_mask_file = file		
		
	# Use repeat to work on the same file
	# for repeat in range(training_steps_number):
	mask_file_list.append(masks_folder + file)

print('Finished: Reading input file list')


#### ==== LOAD IMAGE FILES ====
training_files_count = len(original_file_list)
# file_order = np.random.permutation(file_list_length)


## All images loaded and stored as image objects
# Assume all images have the same size
full_loaded_originals = []
full_loaded_masks = []
full_loaded_rotated_originals = []
full_loaded_rotated_masks = []
labels_frequency = np.zeros(output_features_count, np.int32)
for ind in range(training_files_count):
	# Originals
	tmp_cur_image = Image.open(original_file_list[ind])
	original_image_size = tmp_cur_image.size
	rescaled_image_size = tmp_cur_image.size
	# Rescale the image if necessary
	if bl_rescale_input_images:
		tmp_cur_image = rescale_image(tmp_cur_image, Image.LANCZOS)
		rescaled_image_size = tmp_cur_image.size
	# Extend the image to be able to process boundary regions
	# Need to convert image to numpy and back to image to perform reflection operations
	if not bl_predict:
		tmp_cur_image = np.array(tmp_cur_image, dtype = np.uint8)
		[tmp_cur_image, big_window_side, small_window_side, input_image_size, x_mesh_length, y_mesh_length, x_out_start, y_out_start, x_in_start, y_in_start] = extend_image(tmp_cur_image, bl_predict)
		tmp_cur_image = Image.fromarray(tmp_cur_image)
	full_loaded_originals.extend([tmp_cur_image])
	# print(tmp_cur_image)
	
	### === Masks ===
	tmp_cur_image = Image.open(mask_file_list[ind])
	# Rescale the image if necessary
	if bl_rescale_input_images:
		tmp_cur_image = rescale_image(tmp_cur_image, Image.NEAREST)
	# Convert to numpy
	tmp_cur_image = np.array(tmp_cur_image, dtype = np.uint8)
	# Extend the image to be able to process boundary regions
	if not bl_predict:
		# Count labels before the image is extended (to get the right statistics)
		labels_frequency += count_labels(tmp_cur_image)
		[tmp_cur_image, _, _, _, _, _, _, _, _, _] = extend_image(tmp_cur_image, bl_predict)
	# Identify colors and replace them with labels
	tmp_label_mask = np.zeros(tmp_cur_image.shape[0:2], dtype = np.uint8)
	if not bl_predict:
		tmp_bl_labelled_pixels = np.zeros(tmp_cur_image.shape[0:2], dtype = bool)
		for label_ind in range(2, output_features_count):
			tmp_bl_color_match = np.logical_and(np.equal(tmp_cur_image[:, :, 0], mask_colors[label_ind] [0]), np.equal(tmp_cur_image[:, :, 1], mask_colors[label_ind] [1]))
			# print(np.sum(tmp_bl_color_match))
			tmp_bl_color_match = np.logical_and(tmp_bl_color_match, np.equal(tmp_cur_image[:, :, 2], mask_colors[label_ind] [2]))
			tmp_label_mask = tmp_label_mask + min([label_ind, 3]) * tmp_bl_color_match.astype(np.uint8)
			# Save processed pixels
			tmp_bl_labelled_pixels = np.logical_or(tmp_bl_labelled_pixels, tmp_bl_color_match)
			# labels_frequency[label_ind] += np.sum(tmp_bl_color_match.astype(np.uint32))
		# Set remaining pixels to 1
		tmp_label_mask = tmp_label_mask + np.logical_not(tmp_bl_labelled_pixels).astype(np.uint8)
		# labels_frequency[1] += np.sum(np.logical_not(tmp_bl_labelled_pixels).astype(np.uint32))
	# np.set_printoptions(threshold = np.nan)	
	# print(tmp_cur_image.shape[0:2])
	# print(tmp_label_mask)
	# print(np.mean(tmp_label_mask == 1))
	# Calculate image statistics
	

	# Store as image because will need to rotate later
	full_loaded_masks.extend([Image.fromarray(tmp_label_mask)])

	tmp_list_origs =[]
	tmp_list_masks = []
	if not bl_predict:
		## Create image rotations for training
		for rot_angle in np.arange(0, 360, rotation_step_dg):
			print('Reading input files. File: %i/%i. Rotation: %i/%i' % (ind + 1, training_files_count, 1 + int(rot_angle/rotation_step_dg), rotations_count))
			#- Originals with two reflections
			tmp_cur_rotation = full_loaded_originals[ind].rotate(rot_angle, resample = Image.BILINEAR, expand = 0)
			tmp_list_origs.extend([np.array(tmp_cur_rotation, dtype = np.uint8)])
			#-- Flip 1
			tmp_cur_rotation = ImageOps.flip(full_loaded_originals[ind]).rotate(rot_angle, resample = Image.BILINEAR, expand = 0)
			tmp_list_origs.extend([np.array(tmp_cur_rotation, dtype = np.uint8)])
			#-- Flip 2
			tmp_cur_rotation = ImageOps.mirror(full_loaded_originals[ind]).rotate(rot_angle, resample = Image.BILINEAR, expand = 0)
			tmp_list_origs.extend([np.array(tmp_cur_rotation, dtype = np.uint8)])
			#- Masks
			tmp_cur_rotation = full_loaded_masks[ind].rotate(rot_angle, resample = Image.NEAREST, expand = 0)
			tmp_list_masks.extend([np.array(tmp_cur_rotation, dtype = np.uint8)])
			#-- Flip 1
			tmp_cur_rotation = ImageOps.flip(full_loaded_masks[ind]).rotate(rot_angle, resample = Image.NEAREST, expand = 0)
			tmp_list_masks.extend([np.array(tmp_cur_rotation, dtype = np.uint8)])
			#-- Flip 2
			tmp_cur_rotation = ImageOps.mirror(full_loaded_masks[ind]).rotate(rot_angle, resample = Image.NEAREST, expand = 0)
			tmp_list_masks.extend([np.array(tmp_cur_rotation, dtype = np.uint8)])		
		full_loaded_rotated_originals.extend(tmp_list_origs)
		full_loaded_rotated_masks.extend(tmp_list_masks)
	else:
		## Keep just one original image
		full_loaded_rotated_originals.extend([np.array(full_loaded_originals[ind], dtype = np.uint8)])
		full_loaded_rotated_masks.extend([np.array(full_loaded_masks[ind], dtype = np.uint8)])
# # Freeing up the memory
# tmp_list_origs = []
# tmp_list_masks = []
print('COMPLETED: Reading input files')
# All rotations are kept at the same size as the originals


# Finish statistics calculations
if not bl_predict:
	labels_frequency = labels_frequency.astype(np.float) / np.sum(labels_frequency)
else:
	labels_frequency = np.ones(output_features_count, np.int32)
print('Frequency of labels: %s' % (labels_frequency))


# print(np.sum(full_loaded_rotated_masks[0][3]==3))


## Determine image size and files count
full_image_sizes = full_loaded_originals[0].size	
rotated_training_files_count = len(full_loaded_rotated_originals)
# print(rotated_training_files_count)


## Use non-overlapping locations of the input window
# x_mesh = [x_in_start]
# y_mesh = [y_in_start]
if not bl_predict:
	window_step_px = small_window_side
else:
	window_step_px = 10	# A dummy constant
x_mesh = np.arange(0, full_image_sizes[0] - expected_image_size_x + 1, window_step_px)
y_mesh = np.arange(0, full_image_sizes[1] - expected_image_size_y + 1, window_step_px)
x_mesh_length = len(x_mesh)
y_mesh_length = len(y_mesh)
# print(x_mesh)



## Create all possible cominations of window positions and files
deformation_parameters = np.mgrid[:rotated_training_files_count, :x_mesh_length, :y_mesh_length]
deformation_parameters = np.reshape(np.transpose(deformation_parameters), [-1, 3])
training_set_size = len(deformation_parameters)
if not bl_predict:
	print('Permutating access indices...')
	deformation_parameters = np.random.permutation(deformation_parameters)
	print('COMPLETED: Permutating access indices')
# print(deformation_parameters)


## Garbage Collection
del tmp_cur_image
del tmp_label_mask
tmp_cur_rotation = None
del tmp_list_masks
del tmp_list_origs
tmp_bl_color_match = None
tmp_bl_labelled_pixels = None
del full_loaded_originals
del full_loaded_masks










