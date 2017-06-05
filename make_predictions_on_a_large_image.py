


def make_predictions_on_a_large_image(sess, loaded_original_image, bl_predict, start_time):

	## Extend the image to be able to work with borders
	print('Time: %.2f s. Extending the image...' % (time.time() - start_time))
	[image_ext, big_window_side, small_window_side, input_image_size, x_mesh_length, y_mesh_length, x_out_start, y_out_start, x_in_start, y_in_start] = extend_image(loaded_original_image, bl_predict)
	print('Time: %.2f s. COMPLETED: Extending the image.' % (time.time() - start_time))

	print('Input image size: %ix%i px' % (input_image_size[0], input_image_size[1]))
	print('Small window size: %ix%i px' % (small_window_side, small_window_side))
	print('Big window size: %ix%i px' % (big_window_side, big_window_side))
	print('Extended image size: %ix%i px' % (image_ext.shape[0], image_ext.shape[1]))


	## Make the prediction for the extended image
	print('Calculating predictions...')
	dummy_mask = np.zeros([big_window_side, big_window_side], dtype = np.uint8)
	prediction_out = np.zeros([small_window_side * x_mesh_length, small_window_side * y_mesh_length], dtype = np.uint8)
	for x_ind in range(x_mesh_length):
		cur_x_start = small_window_side * x_ind
		for y_ind in range(y_mesh_length):
			if not y_ind % 5:
				print('Time: %.2f s. Progress: %.2f %% (x: %i/%i, y: %i/%i)' % (time.time() - start_time, 100 * (0.0 + x_ind * y_mesh_length + y_ind)/x_mesh_length/y_mesh_length, x_ind, x_mesh_length, y_ind, y_mesh_length))
			cur_y_start = small_window_side * y_ind
			cur_batch = image_ext[cur_x_start:(cur_x_start + big_window_side), cur_y_start:(cur_y_start + big_window_side)]
			# print((cur_x_start, cur_x_start + small_window_side, cur_y_start, cur_y_start + small_window_side))
			# print(cur_batch.shape)
			# print(dummy_mask.shape)
			# Get a result from the network
			cur_prediction, cur_coarse_grained_mask, cost_eval = sess.run((
				labels_prediction, saved_cur_mask_image, cost),
				feed_dict={
					input_original_image: cur_batch.astype(np.int32), 
					input_mask_image: dummy_mask.astype(np.int32), 
					keep_prob: keep_probability_predict})
			# Combine into the output prediction image
			# print(cur_y_start, cur_y_start + small_window_side)
			prediction_out[cur_x_start:(cur_x_start + small_window_side), cur_y_start:(cur_y_start + small_window_side)] = cur_prediction
			# print(cur_batch)
			# print (cur_prediction)
			
	## Cut the prediction to the input size
	prediction_out = prediction_out[(- x_in_start):(- x_in_start + input_image_size[0]), (- y_in_start):(- y_in_start + input_image_size[1])]
	print('Time: %.2f s. COMPLETED: Making predictions...' % (time.time() - start_time))


	## Return the result
	return [prediction_out]













