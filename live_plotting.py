#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

########################################################################################################################
# Place cell figure setup

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5,8))
fig.canvas.set_window_title('Network Plots')

plot_rates = np.zeros((10,10))
plot_intrinsic_es = np.zeros((10,10))

im1 = ax1.imshow(plot_rates, cmap='YlOrRd', origin='lower', interpolation='nearest', vmin=0, vmax=100, animated=True)
ax1.title.set_text('Network Rates')
ax1.set_xticks(range(0,10))
ax1.set_xticklabels(range(1,11))
ax1.set_yticks(range(0,10))
ax1.set_yticklabels(range(1,11))
cbar = fig.colorbar(im1, ax=ax1)
cbar.set_label('Hz', rotation=0, labelpad=5)

im2 = ax2.imshow(plot_intrinsic_es, cmap='Blues', origin='lower', interpolation='nearest', vmin=0, vmax=4)
ax2.title.set_text('Intrinsic Plasticity')
ax2.set_xticks(range(0,10))
ax2.set_xticklabels(range(1,11))
ax2.set_yticks(range(0,10))
ax2.set_yticklabels(range(1,11))
cbar = fig.colorbar(im2, ax=ax2)
cbar.set_label('$\sigma$', rotation=0, labelpad=5)

def updatefig_live(*args):
	global plot_rates, plot_intrinsic_es
	try:
		rates = np.load('data/rates_data.npy', allow_pickle=True)
		if np.size(rates) == 100:
			plot_rates = np.flip(np.reshape(rates, (10, 10)), 0)
	except:
		pass
	try:
		intrinsic_es = np.load('data/intrinsic_e.npy', allow_pickle=True)
		if np.size(intrinsic_es) == 100:
			plot_intrinsic_es = np.flip(np.reshape(intrinsic_es, (10, 10)), 0)
	except:
		pass
	im1.set_array(plot_rates)
	im2.set_array(plot_intrinsic_es)
	return im1, im2
#
# ########################################################################################################################
# # Action cell figure setup
#
# fig_ac, ax_ac = plt.subplots(1, 1)
# plot_vals_ac = np.zeros((2,2))
# im_ac = ax_ac.imshow(plot_vals_ac, cmap='YlOrRd', origin='lower', interpolation='nearest', vmin=0, vmax=1)
#
# def updatefig_live_ac(*args):
# 	global plot_vals_ac
# 	try:
# 		vals = np.load('data/action_cells_vals.npy', allow_pickle=True)
# 		if np.size(vals) == 4:
# 			east = vals[0]
# 			west = vals[2]
# 			north = vals[1]
# 			south = vals[3]
# 			plot_vals_ac = np.array(((west, south),(north, east)))
#
# 	except:
# 		pass
# 	im_ac.set_array(plot_vals_ac)
# 	return im_ac
#
# ########################################################################################################################
# # Weight vector figure setup
# # action_cells = np.array(e, n, w, s)
# fig_weights, ax_weights = plt.subplots(1, 1)
# ax_weights.title.set_text("Weight vectors")
# weights = np.random.random((4, 100))
# weight_vectors_x_components = np.zeros(100)
# weight_vectors_y_components = np.zeros(100)
#
# x_coords = np.arange(10) / 5 - 0.9
# y_coords = np.flip(np.arange(10) / 5 - 0.9)
# x, y = np.meshgrid(x_coords, y_coords)
# im_weights = ax_weights.quiver(x, y, weight_vectors_x_components, weight_vectors_y_components, scale=20)
#
#
#
# def updatefig_live_weights(*args):
# 	global weight_vectors_x_components
# 	global weight_vectors_y_components
# 	try:
# 		weights = np.load('data/weights.npy', allow_pickle=True)
# 		if np.size(weights) == 400:
# 			for i in range(len(weight_vectors_x_components)):
# 				weight_vectors_x_components[i] = (weights[0, i] - weights[2, i])
# 				weight_vectors_y_components[i] = (weights[1, i] - weights[3, i])
# 				# print(weights[0, i] + weights[1, i] + weights[2, i] + weights[3, i])
# 	except:
# 		pass
# 	im_weights.set_UVC(weight_vectors_x_components, weight_vectors_y_components)
# 	return im_weights

# ########################################################################################################################
# Weight vector figure setup for arbitrary number of action cells
number_action_cells = 72
number_place_cells = 100
fig_weights_arbitrary, ax_weights_arbitrary = plt.subplots(1, 1)
ax_weights_arbitrary.title.set_text("Weight vectors")
weights_arbitrary = np.zeros((number_action_cells, number_place_cells))
weight_vectors_x_components_arbitrary = np.zeros(number_place_cells)
weight_vectors_y_components_arbitrary = np.zeros(number_place_cells)
EE = np.sqrt(weight_vectors_x_components_arbitrary**2 + weight_vectors_y_components_arbitrary**2)
x_coords = np.arange(10) / 5 - 0.9
y_coords = np.flip(np.arange(10) / 5 - 0.9)
x, y = np.meshgrid(x_coords, y_coords)
im_weights_arbitrary = ax_weights_arbitrary.quiver(x, y, weight_vectors_x_components_arbitrary,
                                                   weight_vectors_y_components_arbitrary, EE, scale=20, clim=(0, 4))
cbar = fig_weights_arbitrary.colorbar(im_weights_arbitrary, ax=ax_weights_arbitrary)
# im_weights_arbitrary.set_clim(0, 1)

def updatefig_live_weights_arbitrary(*args):
	global weight_vectors_x_components_arbitrary
	global weight_vectors_y_components_arbitrary
	try:
		# Testing
		# weights_arbitrary = np.zeros((number_action_cells, number_place_cells))
		# weights_arbitrary[45, 50] = 1
		# weights_arbitrary[44, 50] = 0.45
		# weights_arbitrary[46, 50] = 0.32

		weight_vectors_x_components_arbitrary = np.zeros(number_place_cells)
		weight_vectors_y_components_arbitrary = np.zeros(number_place_cells)
		magnitudes = np.zeros(number_place_cells)
		weights_arbitrary = np.load('data/weights.npy', allow_pickle=True)
		if np.size(weights_arbitrary) == number_place_cells * number_action_cells:
			angles = np.radians(np.arange(0, 360, 5))
			for i in range(number_place_cells):
				for j in range(number_action_cells):
					weight_vectors_x_components_arbitrary[i] += weights_arbitrary[j, i] * np.cos(angles[j])
					weight_vectors_y_components_arbitrary[i] += weights_arbitrary[j, i] * np.sin(angles[j])
				# Compute magnitudes and normalise the vectors
				magnitudes[i] = np.sqrt(weight_vectors_x_components_arbitrary[i]**2 +
				                     weight_vectors_y_components_arbitrary[
					i]**2)
				weight_vectors_x_components_arbitrary[i] = weight_vectors_x_components_arbitrary[i] / magnitudes[i]
				weight_vectors_y_components_arbitrary[i] = weight_vectors_y_components_arbitrary[i] / magnitudes[i]
			# print(weight_vectors_x_components_arbitrary)
			# print(magnitudes)
	except:
		pass
	im_weights_arbitrary.set_UVC(weight_vectors_x_components_arbitrary, weight_vectors_y_components_arbitrary,
	                             magnitudes)
	im_weights_arbitrary.set_cmap('Blues')
	# im_weights_arbitrary.set_norm(norm=True)
	return im_weights_arbitrary

########################################################################################################################
# # Eligibility trace vector figure setup for arbitrary number of action cells

# Weight vector figure setup for arbitrary number of action cells
number_action_cells = 72
number_place_cells = 100
fig_elig_arbitrary, ax_elig_arbitrary = plt.subplots(1, 1)
ax_elig_arbitrary.title.set_text("Eligibility trace vectors")
elig_arbitrary = np.zeros((number_action_cells, number_place_cells))
elig_vectors_x_components_arbitrary = np.zeros(number_place_cells)
elig_vectors_y_components_arbitrary = np.zeros(number_place_cells)

x_coords = np.arange(10) / 5 - 0.9
y_coords = np.flip(np.arange(10) / 5 - 0.9)
x, y = np.meshgrid(x_coords, y_coords)
im_elig_arbitrary = ax_elig_arbitrary.quiver(x, y, elig_vectors_x_components_arbitrary,
                                                   elig_vectors_y_components_arbitrary, EE, scale=20, clim=(0, 0.5))
cbar_elig = fig_elig_arbitrary.colorbar(im_elig_arbitrary, ax=ax_elig_arbitrary)
# im_weights_arbitrary.set_clim(0, 1)

def updatefig_live_elig_arbitrary(*args):
	global elig_vectors_x_components_arbitrary
	global elig_vectors_y_components_arbitrary
	try:
		# Testing
		# weights_arbitrary = np.zeros((number_action_cells, number_place_cells))
		# weights_arbitrary[45, 50] = 1
		# weights_arbitrary[44, 50] = 0.45
		# weights_arbitrary[46, 50] = 0.32

		elig_vectors_x_components_arbitrary = np.zeros(number_place_cells)
		elig_vectors_y_components_arbitrary = np.zeros(number_place_cells)
		elig_magnitudes = np.zeros(number_place_cells)
		elig_arbitrary = np.load('data/eligibility_trace.npy', allow_pickle=True)
		if np.size(elig_arbitrary) == number_place_cells * number_action_cells:
			angles = np.radians(np.arange(0, 360, 5))
			for i in range(number_place_cells):
				for j in range(number_action_cells):
					elig_vectors_x_components_arbitrary[i] += elig_arbitrary[j, i] * np.cos(angles[j])
					elig_vectors_y_components_arbitrary[i] += elig_arbitrary[j, i] * np.sin(angles[j])
				# Compute magnitudes and normalise the vectors
				elig_magnitudes[i] = np.sqrt(elig_vectors_x_components_arbitrary[i]**2 +
				                     elig_vectors_y_components_arbitrary[
					i]**2)
				elig_vectors_x_components_arbitrary[i] = elig_vectors_x_components_arbitrary[i] / elig_magnitudes[i]
				elig_vectors_y_components_arbitrary[i] = elig_vectors_y_components_arbitrary[i] / elig_magnitudes[i]
			# print(elig_vectors_x_components_arbitrary)
			# print(elig_magnitudes)
	except:
		pass
	im_elig_arbitrary.set_UVC(elig_vectors_x_components_arbitrary, elig_vectors_y_components_arbitrary,
	                             elig_magnitudes)
	# im_elig_arbitrary.set_cmap('autumn')
	# im_elig_arbitrary.set_norm(norm=True)
	return im_elig_arbitrary




########################################################################################################################
# # Eligibility trace vector figure setup
# # action_cells = np.array(e, n, w, s)
# fig_elig_trace, ax_elig_trace = plt.subplots(1, 1)
# ax_elig_trace.title.set_text("Eligibility trace vectors")
# elig_trace = np.random.random((4, 100))
# elig_trace_vectors_x_components = np.zeros(100)
# elig_trace_vectors_y_components = np.zeros(100)
#
# x_coords = np.arange(10) / 5 - 0.9
# y_coords = np.flip(np.arange(10) / 5 - 0.9)
# x, y = np.meshgrid(x_coords, y_coords)
# im_elig_trace = ax_elig_trace.quiver(x, y, elig_trace_vectors_x_components, elig_trace_vectors_y_components, scale=20)
#
#
#
# def updatefig_live_elig_trace(*args):
# 	global elig_trace_vectors_x_components
# 	global elig_trace_vectors_y_components
# 	try:
# 		elig_trace = np.load('data/eligibility_trace.npy', allow_pickle=True)
# 		if np.size(elig_trace) == 400:
# 			for i in range(len(elig_trace_vectors_x_components)):
# 				elig_trace_vectors_x_components[i] = (elig_trace[0, i] - elig_trace[2, i])
# 				elig_trace_vectors_y_components[i] = (elig_trace[1, i] - elig_trace[3, i])
# 				# print(weight_vectors_x_components[i], weight_vectors_y_components[i])
# 	except:
# 		pass
# 	im_elig_trace.set_UVC(elig_trace_vectors_x_components, elig_trace_vectors_y_components)
# 	return im_elig_trace


########################################################################################################################
# Plot saved figures

# def updatefig_saved(i):
# 	global rates_series, intrinsic_es_series
# 	plot_rates = np.reshape(rates_series[i], (10, 10))
# 	plot_intrinsic_es = np.reshape(intrinsic_es_series[i], (10, 10))
# 	im1.set_array(plot_rates)
# 	im2.set_array(plot_intrinsic_es)
# 	return im1, im2

########################################################################################################################
# Begin plotting

plot_live = True
if plot_live:
	# plot live data
	ani_live = animation.FuncAnimation(fig, updatefig_live, interval=10, blit=True)
	# ani_live_ac = animation.FuncAnimation(fig_ac, updatefig_live_ac)
	# ani_live_weights = animation.FuncAnimation(fig_weights, updatefig_live_weights, blit=False)
	# ani_live_elig_trace = animation.FuncAnimation(fig_elig_trace, updatefig_live_elig_trace, blit=False)
	ani_live_weights_arbitrary = animation.FuncAnimation(fig_weights_arbitrary, updatefig_live_weights_arbitrary)
	ani_live_elig_arbitrary = animation.FuncAnimation(fig_elig_arbitrary, updatefig_live_elig_arbitrary)
# else:
# 	# plots saved data
# 	time_series = np.load('data/time_series_1.npy')
# 	rates_series = np.load('data/rates_series_1.npy')
# 	intrinsic_es_series = np.load('data/intrinsic_e_series_1.npy')
#
# 	# ani_saved = animation.FuncAnimation(fig, updatefig_saved, interval=10, blit=True)
#
# 	# Line plots of the place cell activities over time used for the LM paper.
# 	fig_size = (2,6)
# 	replay_fig, replay_axes = plt.subplots(14, 2, figsize=fig_size, sharey=True)
# 	replay_fig.canvas.set_window_title('Replay Plots')
# 	colours = ['purple', 'blue', 'green', 'orange', 'red']
#
# 	replay_fig.text(0.03, 0.55, 'Neuron Index', rotation=90, fontsize=9, fontstyle='normal', fontweight='light')
# 	replay_fig.text(0.45, 0.03, 'Time (s)', rotation=0, fontsize=9, fontstyle='normal', fontweight='light')
#
# 	axes_i = 0
# 	# The total cell numbers that were active during exploration without their temporal order were [11, 12, 21, 22,
# 	# 23, 33, 34, 41, 44, 45, 51, 54, 55, 65, 75, 85]
# 	cell_indices = [11, 12, 21, 22, 23, 33, 34, 44, 45, 54, 55, 65, 75, 85]
# 	print(np.size(time_series))
# 	for num, i in enumerate(cell_indices):
# 		# if max(rates_series[:,i]) > 10:
# 		# 	print(i, np.argmax(rates_series[:,i]), max(rates_series[:,i]))
# 		replay_axes[axes_i, 0].plot(time_series[:], rates_series[:, i], str(float(axes_i) / 25))
# 		replay_axes[axes_i, 0].set_xlim((22, 33))
# 		replay_axes[axes_i, 0].set_ylim((0, 100))
# 		# line_axes[axes_i].get_yaxis().set_visible(False)
# 		replay_axes[axes_i, 0].get_yaxis().set_ticks([])
# 		if i != cell_indices[-1]:
# 			replay_axes[axes_i, 0].get_xaxis().set_visible(False)
# 		else:
# 			replay_axes[axes_i, 0].set_xticks((22, 27, 32))
# 			replay_axes[axes_i, 0].set_xticklabels((0, 5, 10), size=8)
# 			# replay_axes[axes_i, 0].set_xlabel('Time (s)')
# 		replay_axes[axes_i, 0].set_ylabel(str(num+1), labelpad=5, rotation=0, size=8, va='center')
# 		axes_i += 1
#
# 	axes_i = 0
# 	for num, i in enumerate(cell_indices):
# 		# if max(rates_series[:,i]) > 10:
# 			# print(i, np.argmax(rates_series[:,i]), max(rates_series[:,i]))
# 		replay_axes[axes_i, 1].plot(time_series[:], rates_series[:, i], str(float(axes_i) / 25))
# 		replay_axes[axes_i, 1].set_ylim((0, 100))
# 		replay_axes[axes_i, 1].set_xlim((37.6, 38))
# 		replay_axes[axes_i, 1].get_yaxis().set_ticks([])
# 		if i != cell_indices[-1]:
# 			replay_axes[axes_i, 1].get_xaxis().set_visible(False)
# 		else:
# 			replay_axes[axes_i, 1].set_xticks((37.6, 37.8, 38))
# 			replay_axes[axes_i, 1].set_xticklabels((15.6, 15.8, 16), size=8, position=(1,0))
# 			# replay_axes[axes_i, 1].set_xlabel('Time (s)')
# 		# replay_axes[axes_i, 1].set_ylabel(str(num + 1), labelpad=5, rotation=0, size=8, va='center')
# 		axes_i += 1

plt.show()

# # Plot bar chart of the weights for a place cell
# weights = np.load('data/weights_72_acs.npy')
# chosen_weight_vec = weights[:, 44]
# plt.bar(np.arange(0, 360, 5), chosen_weight_vec)
# plt.show()