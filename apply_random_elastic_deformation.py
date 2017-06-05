


def apply_random_elastic_deformation(original_image, mask_image):

	## Constants
	init_small_window_side = 10


	# ## Cut the image for checking
	# original_image = original_image[:, 0:-1]
	# mask_image = mask_image[:, 0:-1]


	## Initialize
	## Generate a small deformations field
	small_deformations_field_x = np.random.normal(loc = 0.0, scale = deformations_amplitude, size = (small_deformations_field_side, small_deformations_field_side))
	small_deformations_field_y = np.random.normal(loc = 0.0, scale = deformations_amplitude, size = (small_deformations_field_side, small_deformations_field_side))

	
	# Sizes
	[original_size_x, original_size_y] = original_image.shape
	

	## Generate a small deformations field
	small_deformations_field_x = np.random.normal(loc = 0.0, scale = deformations_amplitude, size = (small_deformations_field_side, small_deformations_field_side))
	small_deformations_field_y = np.random.normal(loc = 0.0, scale = deformations_amplitude, size = (small_deformations_field_side, small_deformations_field_side))


	### === Linearly interpolate the field to a larger scale using scipy ===
	## Prepare the mesh of desired locations
	# 1D
	large_mesh_1D_x = np.arange(original_size_x) / float(original_size_x - 1)
	large_mesh_1D_y = np.arange(original_size_y) / float(original_size_y - 1)
	# 2D
	large_mesh_2D_x, large_mesh_2D_y = np.meshgrid(large_mesh_1D_x, large_mesh_1D_y, indexing = 'ij')
	# 2D flat
	large_mesh_2D_flat_x = np.ndarray.flatten(large_mesh_2D_x)
	large_mesh_2D_flat_y = np.ndarray.flatten(large_mesh_2D_y)
	# 
	large_mesh_2D_flat_xy = np.stack([large_mesh_2D_flat_x, large_mesh_2D_flat_y])
	# Transpose to make pairs
	large_mesh_2D_flat_xy = np.transpose(large_mesh_2D_flat_xy)


	## Interpolate large deformation field
	large_deformations_field_x = interpolate_2D(small_deformations_field_x, large_mesh_2D_flat_xy, 'linear')
	large_deformations_field_y = interpolate_2D(small_deformations_field_y, large_mesh_2D_flat_xy, 'linear')
	# # Reshape
	# large_deformations_field_x = np.reshape(large_deformations_field_x, (extended_size_x, extended_size_y))
	# large_deformations_field_y = np.reshape(large_deformations_field_y, (extended_size_x, extended_size_y))
	# Determine the maximal deformation
	max_deformation = int(np.ceil(np.max([np.max(np.abs(large_deformations_field_x)), np.max(np.abs(large_deformations_field_y))])))
	# Make it even
	max_deformation += max_deformation % 2
	# init_small_window_side = 


	# Generate an extended image to reflect borders and be able to get elastic deformations from the "outside"
	small_window_side = init_small_window_side
	big_window_side = init_small_window_side + 2 * (max_deformation)
	[extended_image, big_window_side, small_window_side, input_image_size, x_mesh_length, y_mesh_length, x_out_start, y_out_start, x_in_start, y_in_start] = extend_image(original_image, True, small_window_side, big_window_side)
	[extended_mask, _, _, _, _, _, _, _, _, _] = extend_image(mask_image, True, small_window_side, big_window_side)
	[extended_size_x, extended_size_y] = extended_image.shape


	## Calculate the normalized locations of the new pixels: the whole image is [0, 1]
	# The minus sign with the start term just referes to the way it is counted. It's still a positive shift
	new_locations_x = (large_mesh_2D_flat_x * (original_size_x - 1) + large_deformations_field_x - x_out_start) / float(extended_size_x - 1)
	new_locations_y = (large_mesh_2D_flat_y * (original_size_y - 1) + large_deformations_field_y - y_out_start) / float(extended_size_y - 1)
	# np.set_printoptions(threshold='nan')
	# print(extended_image.shape)
	# print(x_out_start)
	# print(y_out_start)
	# print(np.min(new_locations_x) * float(extended_size_x - 1), np.max(new_locations_x) * float(extended_size_x - 1))
	# print(np.min(new_locations_y) * float(extended_size_y - 1), np.max(new_locations_y) * float(extended_size_y - 1))
	# print('Original size: %s, extended size: %s' % (original_image.shape, extended_image.shape))
	# print('Small window side: %i, big window side: %i' % (small_window_side, big_window_side))
	# print(x_mesh_length, y_mesh_length)
	# print(new_locations_x)
	# print(new_locations_y)
	# Combine
	new_locations_xy = np.transpose(np.stack([new_locations_x, new_locations_y]))

	# print(new_locations_x)
	# print(new_locations_y)

	## Calculate the distorted original image
	distorted_image = interpolate_2D(extended_image, new_locations_xy, 'linear')
	# Statistics
	# print('Min: %.2f, max: %.2f, max deformation: %i; %.2f' % (np.min(distorted_image), np.max(distorted_image), max_deformation, np.max([np.max(large_deformations_field_x), np.max(large_deformations_field_y)])))
	# Reshape
	distorted_image = np.reshape(distorted_image, (original_size_x, original_size_y))
	# Convert to int
	distorted_image = distorted_image.astype(np.uint8)
	# print(distorted_image)

	## Calculate the distorted mask
	distorted_mask = interpolate_2D(extended_mask, new_locations_xy, 'nearest')
	# Statistics
	# print('Min: %.2f, max: %.2f, max deformation: %i; %.2f' % (np.min(distorted_image), np.max(distorted_image), max_deformation, np.max([np.max(large_deformations_field_x), np.max(large_deformations_field_y)])))
	# Reshape
	distorted_mask = np.reshape(distorted_mask, (original_size_x, original_size_y))
	# Convert to int
	distorted_mask = distorted_mask.astype(np.uint8)


	return [distorted_image, distorted_mask]
	# return [extended_image, extended_mask]







	
	# ## Perform interpolation through the Pillow library (convert there and back)
	# # x
	# large_deformations_field_x = Image.fromarray(small_deformations_field_x)
	# large_deformations_field_x = large_deformations_field_x.resize((extended_size_x, extended_size_y), resample = Image.BILINEAR)
	# large_deformations_field_x = np.array(large_deformations_field_x)
	# # y
	# large_deformations_field_y = Image.fromarray(small_deformations_field_y)
	# large_deformations_field_y = large_deformations_field_y.resize((extended_size_x, extended_size_y), resample = Image.BILINEAR)
	# large_deformations_field_y = np.array(large_deformations_field_y)
	
	# # print(small_deformations_field_x)
	# # print(large_deformations_field_x)


	# ## The distortion field is ready and needs to be applied to the original and mask image
	# # 1D Meshes
	# small_mesh_x = np.arange(small_deformations_field_side)/float(small_deformations_field_side-1)
	# small_mesh_y = small_mesh_x
	# large_mesh_x = np.arange(extended_size_x)/float(extended_size_x-1)
	# large_mesh_y = np.arange(extended_size_y)/float(extended_size_y-1)
	# # 2D Mesh
	# small_mesh_x_2D, small_mesh_y_2D = np.meshgrid(small_mesh_x, small_mesh_y)
	# small_mesh_x_2D_flat = np.ndarray.flatten(small_mesh_x_2D)
	# small_mesh_y_2D_flat = np.ndarray.flatten(small_mesh_y_2D)
	# small_mesh_xy_2D_flat = np.stack([small_mesh_x_2D_flat, small_mesh_y_2D_flat])












	# # 1D Meshes
	# small_mesh_x = np.arange(small_deformations_field_side)/float(small_deformations_field_side-1)
	# small_mesh_y = small_mesh_x
	# large_mesh_x = np.arange(extended_size_x)/float(extended_size_x-1)
	# large_mesh_y = np.arange(extended_size_y)/float(extended_size_y-1)
	# # 2D Mesh
	# small_mesh_x_2D, small_mesh_y_2D = np.meshgrid(small_mesh_x, small_mesh_y)
	# small_mesh_2D = np.stack([small_mesh_x_2D, small_mesh_y_2D])
	# small_mesh_2D = np.transpose(small_mesh_2D,(1,2,0))


		# ## Rescale the deformation field to the the size of the original image
	# # large_deformations_field = np.zeros((extended_size_x, extended_size_y), dtype = np.float)
	# # print(small_mesh_x_2D.shape)
	# print(small_deformations_field_x)
	# # print((np.ndarray.flatten(small_mesh_x_2D), np.ndarray.flatten(small_mesh_y_2D)))
	# # # print(small_deformations_field_x)
	# # # print(large_mesh_x)
	# # print(small_mesh_2D)

	# large_deformations_field_x = interpolate.griddata(points = (np.ndarray.flatten(small_mesh_x_2D), np.ndarray.flatten(small_mesh_y_2D)), values = np.ndarray.flatten(small_deformations_field_x), xi = (large_mesh_x, large_mesh_y), method = 'linear')
	# # Reshape
	# large_deformations_field_x = np.reshape(large_deformations_field_x, (extended_size_x, extended_size_y))
	# print(large_deformations_field_x)


