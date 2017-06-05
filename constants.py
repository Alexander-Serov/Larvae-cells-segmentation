


## INPUT-OUTPUT PARAMETERS
train_originals_folder = './input_originals/'
train_masks_folder = './input_masks/'
test_originals_folder = './test_originals/'
test_masks_folder = './test_masks/'
output_folder = './output/'
save_path = './saved_model/'
FILE_EXTENSION = ".png"


## IMAGE PRE-PROCESSING PARAMETERS
# window_step_px = 5
rotations_count = 4
rotation_step_dg = 360.0 / rotations_count
mask_colors = {
	0: [0, 0, 0], # outside (zero padding from rotation)
	1: [255, 255, 255], # background (unimportant components of the real image)
	2: [0, 0, 255], # Deep blue: flat surface of the microvilli
	3: [0, 255, 255] # Cyan: microvilli
	# 4: [255, 0, 0], # Red: villous interior
	# 5: [0, 255, 0], # Green: basal membrane
	# 6: [255, 0, 255] # Magenta: basement membrane
}
output_features_count = len(mask_colors)


## NETWORK ARCHITECTURE PARAMETERS
levels_count = 7	# 6
features_count_first_layer = 16 #int(2**(10 - level_penalty - levels_count + 1))
# output_features_count = 2
deepest_level_min_image_size = 36	# 54


## INDIVIDUAL LAYER PARAMETERS
max_pool_size = 2
convolution_filter_size = 3
convolution_filter_size_1st_layer = 3
upconvolution_filter_size = 2
# pooling_size = 2


## EXECUTION PARAMETERS
bl_load_saved_model = False
# bl_load_saved_model = True
learning_rate = 1e-3
training_steps_number = 1 # long(2e4)
visualization_step = 1	# 100
save_step = 500
keep_probability_train = 1.0
keep_probability_predict = 1.0
bl_get_only_images_with_mask = False
bl_apply_elastic_deformations = True
bl_rescale_input_images = True
rescale_new_x_side = 2024


## ELASTIC DEFORMATION PARAMETERS
small_deformations_field_side = 5;
deformations_amplitude = 10.0;		# in px


## OTHER PARAMETERS
# original_image_side = 572
# queue_capacity = 1000
# min_after_dequeue = 200
# batch_size = 1
average_mask_fraction = 0.0773









