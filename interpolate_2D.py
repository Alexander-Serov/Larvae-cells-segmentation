## The input should be the following:
#
# image - a 2D ndarray, Nx * Ny
# locations - a 2D ndarray, Nl * 2
# method - either 'nearest' or 'linear'
# 



def interpolate_2D(image, locations, method):
	## Initialize
	[image_size_x, image_size_y] = image.shape
	# [locations_size_x, locations_size_y] = locations.shape
	# print(image.shape, locations.shape)

	## Create x and y meshes
	# 1D
	mesh_1D_x = np.arange(image_size_x)/float(image_size_x - 1)
	mesh_1D_y = np.arange(image_size_y)/float(image_size_y - 1)
	# 2D
	# mesh_2D_x, mesh_2D_y = np.meshgrid(mesh_1D_x, mesh_1D_y)
	# 2D flat
	# mesh_2D_flat_x = np.ndarray.flatten(mesh_2D_x)
	# mesh_2D_flat_y = np.ndarray.flatten(mesh_2D_y)


	## Convert the input image and requested locations to a flat array
	# image_flat = np.ndarray.flatten(image)
	# locations_flat = np.ndarray.flatten(locations)


	## Use scipy to calculate the interpolations
	interpolating_function = interpolate.RegularGridInterpolator(points = (mesh_1D_x, mesh_1D_y), values = image, method = method)
	values = interpolating_function(locations)
	# values = interpolate.griddata(points = (mesh_2D_flat_x, mesh_2D_flat_y), values = image_flat, xi = locations, method = method)


	return values










