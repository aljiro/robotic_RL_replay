import numpy as np
import matplotlib.pyplot as plt

# n = 1.
# s = 0.99
# w = 0.0001
# e = 0.
#
# n_s = n - s
# e_w = e - w
# n_s_norm = n_s / np.sqrt(n_s**2 + e_w**2)
# e_w_norm = e_w / np.sqrt(n_s**2 + e_w**2)
#
# theta_tan = np.arctan(n_s / e_w)
# theta_sin = np.arcsin(n_s_norm)
# theta_cos = np.arccos(e_w_norm)
# if e_w > 0.:
# 	if theta_tan < 0.:
# 		theta_proper_tan = 2. * np.pi - theta_tan
# 	else:
# 		theta_proper_tan = theta_tan
# else:
# 	theta_proper_tan = np.pi + theta_tan
#
# print(theta_tan, theta_sin, theta_cos, theta_proper_tan)

# y = np.arange(0, 1, 0.01)
# de_dt = (1 - y) * y
#
# plt.plot(y, de_dt)
# plt.show()

# testing sigmoids
# inputs = np.arange(0, 150)
# c1 = 0.1
# c2 = 60
# outputs = 1 / (1 + np.exp(-c1 * (inputs - c2)))
# plt.plot(inputs, outputs)
# plt.show()


def compute_place_cell_activities(coord_x, coord_y, reward, movement=False):
	'''

	:param coord_x: float, the x coordinate (m)
	:param coord_y: float, the y coordinate (m)
	:param reward: float, the reward value. If reward != 0, the agent should be resting and the C parameter set
	to 1 Hz
	:param movement: bool, indicates whether the robot moved in the current time step or not
	:return: numpy array, vector of the networks place cell activities
	'''

	d = 0.1  # m
	network_size_pc = 100
	no_cells_per_m = np.sqrt(network_size_pc) / 2  # 5
	no_cell_it = int(np.sqrt(network_size_pc))  # 10, the number of cells along one row of the network
	if movement or reward != 0:
		C = 50  # Hz
	else:
		C = 0  # Hz
	cells_activity = np.zeros((no_cell_it, no_cell_it))
	place = np.array((coord_x, coord_y))
	for x in range(no_cell_it):
		for y in range(no_cell_it):
			place_cell_field_location = np.array(((float(x) / 5) - 0.9, (-float(y) / 5) + 0.9))
			cells_activity[y, x] = C * np.exp(
				-1.0 / (2.0 * d ** 2.0) * np.dot((place - place_cell_field_location),
				                                 (place - place_cell_field_location)))
	cell_activities_array = cells_activity.flatten()
	return cell_activities_array

coord_x = -0.3
coord_y = 0.7

test_output = compute_place_cell_activities(coord_x, coord_y, 0, True)
# print(test_output)

for x in range(10):
	print((float(x) / 5) - 0.9)

