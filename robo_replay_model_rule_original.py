#!/usr/bin/python
'''
This is the full robot replay script using the first learning rule for the weights. It subscribes (
via ROS) to the robot's
coordinates, produces the rate
activities according to the model, and replays once a reward has been reached, which is gathered by subscribing to
the reward topic.
'''

import rospy
from std_msgs.msg import UInt8
from geometry_msgs.msg import Pose2D, TwistStamped
import numpy as np
import os
import signal
import sys
import time
import miro2 as miro


class RoboReplay():

	def __init__(self):
		# system handling
		signal.signal(signal.SIGINT, self.signal_handler)

		# ROS stuff
		rospy.init_node("Robo_Replay")

		topic_base_name = "/" + os.getenv("MIRO_ROBOT_NAME")

		# subscribe to the reward value
		topic = topic_base_name + "/reward_value"
		print("subscribed to", topic)
		self.sub_reward = rospy.Subscriber(topic, UInt8, self.update_reward, queue_size=5, tcp_nodelay=True)
		self.reward_val = 0

		# subscribe to Miro's body pose (x, y, theta)
		topic = topic_base_name + "/sensors/body_pose"
		print("subscribed to", topic)
		self.sub_body_pose = rospy.Subscriber(topic, Pose2D, self.callback_body_pose, queue_size=5, tcp_nodelay=True)
		self.body_pose = np.array((-0.7, 0.0, 0.0))  # pos 0 is x coordinate, pos 1 is y coordinate and pos 2 is theta

		# subscribe to miro's sensors package
		topic = topic_base_name + "/sensors/package"
		self.sub_package = rospy.Subscriber(topic, miro.msg.sensors_package, self.callback_sensors, queue_size=5,
		                                    tcp_nodelay=True)
		self.sonar_val = 1

		# publish to wheels
		topic = topic_base_name + "/control/cmd_vel"
		self.pub_wheels = rospy.Publisher(topic, TwistStamped, queue_size=0)
		self.msg_wheels = TwistStamped()

		# model parameters and variable initial conditions
		self.network_size_pc = 100 # A square number
		self.a = 1
		self.epsilon = 2  # Hz min threshold for rates
		# self.theta = 40  # Hz max threshold for rates #TODO give this a different variable name
		self.w_inh = 0.1

		self.delta_t = 0.01  # s simulation time steps
		self.t = 0  # simulation time
		self.t_last_angle_change = 0 # time stamp used to remember the last time an angle change was made

		# set place cell variables initial conditions
		self.place_cell_rates = np.zeros(self.network_size_pc)
		self.currents = np.zeros(self.network_size_pc)
		self.intrinsic_e_reset = np.ones(self.network_size_pc) * 0.1
		self.intrinsic_e = self.intrinsic_e_reset.copy()
		self.network_weights = self.initialise_weights()
		self.stp_d = np.ones(self.network_size_pc)
		self.stp_f = np.ones(self.network_size_pc) * 0.6
		self.I_place = np.zeros(self.network_size_pc)
		self.I_inh = 0

		# model parameters for action cells
		self.network_size_ac = 4 # one each for north, south, east, west
		self.target_theta = 0
		self.eta = 10 # learning rate
		self.tau_elig = 1 # eligibility trace time constant
		self.sigma = 0.1 # standard deviation in the action cell noise

		# set action cell variables initial conditions
		self.action_cell_vals = np.zeros(self.network_size_ac)
		self.action_cell_vals_noise = np.zeros(self.network_size_ac)
		self.weights_pc_ac = self.normalise_weights_pc_ac(np.random.random((self.network_size_ac,
		                                                                    self.network_size_pc))) # 4 x 100
		# self.weights_pc_ac = self.test_weights()

		# set eligibility trace initial condition
		self.elig_trace = np.zeros((self.network_size_ac, self.network_size_pc))

		# set reward prediction initial condition
		self.reward_pred = 0

		# bools
		self.replay = False
		self.heading_home = True # assume MiRo is not in start position

		self.home_pose = np.array((-0.7, 0, 0))

		# lists for storing network values during trials. Saves to the data folder once script is exited via ctrl-c
		# being pressed.
		self.time_series = []
		self.rates_series = []
		self.intrinsic_e_series = []

	def signal_handler(self, sig, frame):
		# print('\nSaving trial data')
		# np.save('data/time_series.npy', self.time_series)
		# np.save('data/rates_series.npy', self.rates_series)
		# np.save('data/intrinsic_e_series.npy', self.intrinsic_e_series)
		#
		# # clean up the temporary data files
		# os.remove('data/intrinsic_e.npy')
		# os.remove('data/rates_data.npy')
		# os.remove('data/place_data.npy')

		sys.exit(0)

	####################################################################################################################
	# ROS callback functions

	def update_reward(self, msg):
		self.reward_val = msg.data

	def callback_body_pose(self, msg):
		self.body_pose[0] = msg.x
		self.body_pose[1] = msg.y
		# theta ranges from [-pi, pi]. Its altered so it ranges from [0, 2pi] whilst keeping the same pose position
		# for 0 rads
		if msg.theta < 0:
			self.body_pose[2] = msg.theta + 2 * np.pi
		else:
			self.body_pose[2] = msg.theta

	def callback_sensors(self, msg):
		self.sonar_val = msg.sonar.range

	####################################################################################################################
	# Place cell network functions

	def initialise_weights(self):
		# weights are initially all symmetrical, but made to obey the normalisation specification that
		# sum_{k,l} w_{i,j}^{k,l} = 10
		# In addition, as each cell is only connected to 8 others, it would be useless computing the
		# learning rates and activities across a 100x100 weight matrix
		# In weights[i,j], i represents the post-synapse and j the pre-synapse. I.e. for a given row of weights (say
		# weights[i]), the 8 weights in that row will be the incoming weights from its neighbouring pre-synapses in
		# locations [W NW N NE E SE S SW]
		weights = np.zeros((self.network_size_pc, 8))
		for i in range(100):
			for j in range(8):
				if self.is_computable(i, j):
					weights[i, j] = 1.0
			weights[i] = weights[i] / sum(weights[i]) * 8

		return weights

	def is_computable(self, i, j):
		'''

		:param i: integer, indicates which is the selected neuron
		:param j: integer, indicates which neighbour neuron i is receiving something from
		:return: bool
		'''

		# This is a confusing function, so I must describe it in detail. Because of the 2D arrangement of the
		# network, the neurons on the edges will not be connected to any neurons to its side (north most neurons have
		# no connections to its north, etc.). As a result, when performing the computations, such as computing the
		# incoming rates to a neuron from its neighbour neurons, it is worth first determining whether this
		# computation is valid (i.e. there is indeed a neighbour neuron in the specified position). This function
		# therefore determines whether a computation, whether it be incoming rates or updating weights, is valid or
		# not. For simplicity, this is computed for a 50x50 2D neural network.
		# It is important to note here that the order of connections is as follows: [W NW N NE E SE S SW]. So from j
		# = 0 to j = 7, the increment in the index represents a clockwise rotation starting at the point W.

		no_cells_per_row = int(np.sqrt(self.network_size_pc))

		if i % no_cells_per_row == 0 and (j == 0 or j == 1 or j == 7):  # no W connections
			return False
		elif i in range(no_cells_per_row) and (j == 1 or j == 2 or j == 3):  # no N connections
			return False
		elif (i + 1) % no_cells_per_row == 0 and (j == 3 or j == 4 or j == 5):  # no E connections
			return False
		elif i in range(self.network_size_pc - no_cells_per_row, self.network_size_pc) and (j == 5 or j == 6 or j == 7):  # no S
			# connections
			return False
		else:  # it's a valid computation
			return True

	def neighbour_index(self, i, j):
		'''

		:param i: integer, indicates which is the selected neuron
		:param j: integer, indicates which neighbour neuron i is receiving something from
		:return: bool
		'''

		# Due to the 2D structure of the network, it is important to find which index from the vector of neurons
		# should be used as the neighbour neuron. For instance, the 2D network is concatenated by each row. So the
		# first 50 neurons of the vector of neurons represents the first row of the 2D network. The next 50 represent
		# the second row, and so on. Hence, the connection that neuron i receives from its north will be located at
		# i-50. For simplicity, this is computed for a 50x50 2D neural network.
		# It is important to note here that the order of connections is as follows: [W NW N NE E SE S SW]. So from j
		# = 0 to j = 7, the increment in the index represents a clockwise rotation starting at the point W.

		no_cells_per_row = int(np.sqrt(self.network_size_pc))

		if j == 0:  # W connection
			return i - 1
		elif j == 1:  # NW connection
			return i - (no_cells_per_row + 1)
		elif j == 2:  # N connection
			return i - no_cells_per_row
		elif j == 3:  # NE connection
			return i - (no_cells_per_row - 1)
		elif j == 4:  # E connection
			return i + 1
		elif j == 5:  # SE connection
			return i + (no_cells_per_row + 1)
		elif j == 6:  # S connection
			return i + no_cells_per_row
		elif j == 7:  # SW connection
			return i + (no_cells_per_row - 1)
		else:
			return IndexError

	def compute_rates(self, currents):
		rates_update = np.zeros(self.network_size_pc)
		for i in range(self.network_size_pc):
			if currents[i] < self.epsilon:
				rates_update[i] = 0
			else:
				rates_update[i] = min(self.a * (currents[i] - self.epsilon), 100) # upper bound of 200 Hz
				# rates_update[i] = self.a * (currents[i] - self.epsilon)  # no upper bound

		return rates_update

	def update_currents(self, currents, delta_t, intrinsic_e, weights, rates, stp_d, stp_f, I_inh, I_place, replay=False):
		tau_I = 0.05  # s
		currents_update = np.zeros(self.network_size_pc)
		g = 0
		for i in range(self.network_size_pc):
			sum_w_r_df = 0
			if replay:  # g is only nonzero during replays, so if there is no replay it's pointless summing neighbour rates
				for j in range(8):
					if self.is_computable(i, j):
						neighbour = self.neighbour_index(i, j)
						sum_w_r_df += weights[i, j] * rates[neighbour] * stp_d[neighbour] * stp_f[neighbour]
				g = intrinsic_e[i]
			currents_update[i] = currents[i] + (-currents[i] + g * sum_w_r_df - I_inh + I_place[i]) * delta_t  / tau_I

		return currents_update

	def update_intrinsic_e(self, intrinsic_e, delta_t, rates):
		tau_e = 10  # s
		sigma_ss = 0.1
		sigma_max = 4
		r_sigma = 10
		beta = 1
		intrinsic_e_update = np.zeros(self.network_size_pc)
		for i in range(self.network_size_pc):
			sigmoid = (sigma_max - 1) / (1 + np.exp(-beta * (rates[i] - r_sigma)))
			intrinsic_e_update[i] = ((sigma_ss - intrinsic_e[i]) / tau_e + sigmoid) * delta_t + intrinsic_e[i]
			if intrinsic_e_update[i] > sigma_max:
				intrinsic_e_update[i] = sigma_max
		return intrinsic_e_update

	def update_I_inh(self, I_inh, delta_t, w_inh, rates):
		tau_inh = 0.05  # s
		sum_rates = 0
		for i in range(self.network_size_pc):
			sum_rates += rates[i]

		return (-I_inh / tau_inh + w_inh * sum_rates) * delta_t + I_inh

	def update_STP(self, STP_D, STP_F, delta_t, rates):
		'''
		:param STP_F: numpy array, 100x1 current STP_F vector
		:param STP_D: numpy array, 100x1 current STP_D vector
		:param delta_t: float, time step (s)
		:param rates: numpy array, 100x1 of network rates
		:return: two numpy arrays, 100x1 updated vectors of the STP variables
		'''

		tau_f = 1  # s
		tau_d = 1.5  # s
		U = 0.6
		STP_F_next = np.zeros(self.network_size_pc)
		STP_D_next = np.zeros(self.network_size_pc)
		for f in range(self.network_size_pc):
			STP_D_next[f] = ((1.0 - STP_D[f]) / tau_d - rates[f] * STP_D[f] * STP_F[f]) * delta_t + STP_D[f]
			STP_F_next[f] = ((U - STP_F[f]) / tau_f + U * (1 - STP_F[f]) * rates[f]) * delta_t + STP_F[f]
		if max(STP_D_next) > 1 or min(STP_D_next) < 0 or max(STP_F_next) > 1 or min(STP_F_next) < 0:
			print('STP value ouside of bounds')
			print(max(STP_D_next), min(STP_D_next), max(STP_F_next), min(STP_F_next))
		# print(STP_F_next*STP_D_next)
		return STP_D_next, STP_F_next

	def compute_place_cell_activities(self, coord_x, coord_y, reward, movement=False):
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

	####################################################################################################################
	# Action cell functions

	def test_weights(self):
		weights = np.zeros((4, 100))
		for i in list(range(5)) + list(range(10,15))+ list(range(20,25))+ list(range(30,35))+ list(range(40,45)):
			weights[0, i] = np.sqrt(2)
			weights[3, i] = np.sqrt(2)
			weights[2, i+5] = np.sqrt(2)
			weights[3, i+5] = np.sqrt(2)
			weights[1, 50+i] = np.sqrt(2)
			weights[0, 50 + i] = np.sqrt(2)
			weights[1, 55 + i] = np.sqrt(2)
			weights[2, 55 + i] = np.sqrt(2)

		# for weight_vector in range(len(weights[0,:])):
		# 	weights[:, weight_vector] = weights[:, weight_vector] / np.linalg.norm(weights[:, weight_vector])
		return weights

	def normalise_weights_pc_ac(self, weights): # test complete, working well
		'''
		normalises the weight matrix between pc cells and action cells
		:param weights: numpy array, 4x100 (ac x pc)
		:return: numpy array, a 4 x 100 (ac x pc) array of normalised weights
		'''

		for weight_vector in range(len(weights[0,:])):
			weights[:, weight_vector] = weights[:, weight_vector] / np.linalg.norm(weights[:, weight_vector])
		return weights

	def compute_action_cell_outputs(self, weights_pc_ac, place_cell_rates): # TODO need to test with different values for c1 and c2
		'''

		:param weights_pc_ac: numpy array, 4x100 array of the pc to ac weights
		:param place_cell_rates: numpy array, 100x1 array of place cell rates
		:return: numpy array, 4x1 array of action cell values
		'''
		# f_s(x) = 1 / (1 + exp(-c1 * (x - c2)) is the sigmoid function. c1 determines the width of the sigmoid,
		# and c2 the midpoint.
		c1 = 0.1
		c2 = 60
		dot_product = np.dot(weights_pc_ac, place_cell_rates)
		return 1 / (1 + np.exp(-c1 * (dot_product - c2)))


	def add_noise_to_action_cell_outputs(self, action_cell_values, sigma):
		'''

		:param action_cell_values: numpy array, 4x1
		:return: numpy array, 4x1 array of the action cells with added Gaussian white noise
		'''

		noise = np.random.normal(0, sigma, self.network_size_ac)
		action_cell_values_noise = np.zeros(self.network_size_ac)
		for i in range(len(action_cell_values)):
			action_cell_values_noise[i] = min(max(action_cell_values[i] + noise[i], 0), 1) # concatonate between 0 and 1
		return action_cell_values_noise

	def weight_updates(self, weights_current, reward_pred_error, elig, eta, sigma, delta_t):
		'''
		# TODO needs testing
		:param weights_current: numpy array, 4x100
		:param reward_pred_error: float
		:param elig: numpy array, 4x100 of the eligibility trace
		:param eta: float, learning rate
		:param sigma: float, standard deviation in the action cell output noise
		:param delta_t: float, time step
		:return: numpy array, 4x100 updated values for the weights
		'''

		weights_updated = weights_current.copy()
		sigma_squared = sigma**2
		for i in range(len(weights_current[:, 0])): # iterate through rows, four of them, with index i
			for j in range(len(weights_current[0, :])): # iterate through columns, 100 of them, with index j
				weights_updated[i, j] += (eta * reward_pred_error * (1 / sigma_squared) * elig[i, j]) * delta_t

		return weights_updated

	def update_eligibility_trace(self, current_eligibility_trace, place_cells, action_cells, action_cell_noise, tau,
	                             delta_t):
		'''
				#TODO needs testing
				:param current_eligibility_trace: numpy array, 4x100
				:param place_cells: numpy array, 100x1 of the current place cell values
				:param action_cells: numpy array, 4x1 of the current action cell values
				:param action_cell_noise: numpy array, 4x1 of the current action cell values with added noise
				:param tau: float, time constant
				:param delta_t: float, time step
				:return: numpy array, 4x100 updated array of the eligibility trace
				'''

		updated_eligibility_trace = current_eligibility_trace.copy()
		for i in range(len(current_eligibility_trace[:, 0])):  # iterate through rows, four of them, with index i
			for j in range(len(current_eligibility_trace[0, :])):  # iterate through columns, 100 of them, with index j
				Y = (action_cell_noise[i] - action_cells[i]) * (1 - action_cells[i]) * action_cells[i] * place_cells[j]
				updated_eligibility_trace[i, j] += (-current_eligibility_trace[i, j] / tau + Y) * delta_t

		return updated_eligibility_trace

	def theta_to_action_cell(self, theta):
		# converts an angular pose for MiRo into action cell values
		north_south = np.sin(theta)
		east_west = np.cos(theta)
		(e, n, w, s) = (0., 0., 0., 0.)
		if north_south >= 0:
			n = north_south
		else:
			s = -north_south
		if east_west >= 0:
			e = east_west
		else:
			w = -east_west
		return np.array((e, n, w, s))

	def action_cell_to_theta(self, action_cells):
		# converts action cell values into a target theta angle. action_cells = np.array(e, n, w, s)
		north_south = action_cells[1] - action_cells[3]
		east_west = action_cells[0] - action_cells[2]
		if east_west == 0:
			east_west = 0.00001 # just to prevent division by 0
		target_theta = np.arctan(north_south / east_west)
		if east_west < 0:
			target_theta += np.pi
		elif north_south < 0:
			target_theta += 2 * np.pi
		print("action cell vals are: ", action_cells, "; north_south = ", north_south, "; east_west = ", east_west)

		return target_theta


	def update_reward_running_average(self, r_current, reward, m):
		'''
		calculated a running average of the reward for this experiment, using the equation on p. 4 of Vasilaki et al. (2009)
		:param r_current: float, current predicted reward value
		:param reward: int, the received reward at the current time point
		:param m: float, time window in ms
		:return: float, the updated running average reward
		'''

		return (1 - 1/m) * r_current + 1/m * reward

	####################################################################################################################
	# MiRo Controller functions

	def miro_controller(self, target_theta, current_theta):
		# function that sends motor controls to MiRo in an attempt to get MiRo to travel in the direction of the
		# target_theta direction
		# vel = 0  # max = 0.4 m/s, min = -0.4 m/s
		# omega = 0  # pos = anticlockwise; max = +- 5 rad/s
		if self.sonar_val < 0.03:
			self.avoid_wall()
			self.target_theta = self.body_pose[2] # keeps MiRo positioned in the new orientation, rather than turning
		# around to the previous one (which led towards the wall)
		else:
			A = 1
			vel = 0.2
			diff = target_theta - current_theta
			if abs(diff) < np.pi:
				omega = A * diff
			elif diff > 0:
				omega = A * (diff - 2 * np.pi)
			else:
				omega = A * (diff + 2 * np.pi)
			self.msg_wheels.twist.linear.x = vel
			self.msg_wheels.twist.angular.z = omega
			self.pub_wheels.publish(self.msg_wheels)

	def random_walk(self, current_theta, current_target_theta):
		#
		# new angle to aim for will be current_angle +- rand, where rand is in the range [-1, 1]
		if self.t - self.t_last_angle_change > 2:
			self.t_last_angle_change = self.t
			# modulo 2pi in case its outside the range [0, 2pi]
			new_theta = (current_theta + 2 * np.random.rand() - 1) % (2 * np.pi)
			# print("Current angle is: ", current_theta, ". New target theta is: ", new_theta)
			return new_theta
		else:
			return current_target_theta

	def goal_directed_walk(self):
		pass

	def avoid_wall(self):
		t_init = time.time()
		self.msg_wheels.twist.linear.x = -0.1
		self.msg_wheels.twist.angular.z = 0
		while time.time() - t_init < 1:
			self.pub_wheels.publish(self.msg_wheels)
			# try giving a negative reward if MiRo moves into a wall
		self.weights_pc_ac = self.normalise_weights_pc_ac(self.weight_updates(self.weights_pc_ac, -1,
                                                                      self.elig_trace, self.eta, self.sigma,
                                                                              self.delta_t))
		self.elig_trace = np.zeros((self.network_size_ac, self.network_size_pc))  # reset the eligibility
		# trace
		p_anti_clock = 1
		self.msg_wheels.twist.linear.x = 0
		if np.random.rand() < p_anti_clock:
			self.msg_wheels.twist.angular.z = 2
		else:
			self.msg_wheels.twist.angular.z = -2
		while time.time() - t_init < 2.5:
			self.pub_wheels.publish(self.msg_wheels)

	def stop_movement(self):
		self.msg_wheels.twist.linear.x = 0
		self.msg_wheels.twist.angular.z = 0
		self.pub_wheels.publish(self.msg_wheels)

	def head_to_position(self, target_pose):
		A = 1
		vel_max = 0.2
		current_pose = self.body_pose.copy()
		diff_x = target_pose[0] - current_pose[0]
		diff_y = target_pose[1] - current_pose[1]
		distance_from_home = np.sqrt(diff_x ** 2 + diff_y ** 2)
		print("Heading home...")
		while distance_from_home > 0.02:
			# wall avoidance
			if self.sonar_val < 0.03:
				self.avoid_wall()
				self.target_theta = self.body_pose[2]

			current_pose = self.body_pose.copy()
			diff_x = target_pose[0] - current_pose[0]
			diff_y = target_pose[1] - current_pose[1]
			if diff_x == 0:
				diff_x += 0.000001
			distance_from_home = np.sqrt(diff_x ** 2 + diff_y ** 2)
			theta_miro_home = np.arctan(diff_y / diff_x) # angle of vector from miro_position to miro_home_position


			if diff_x > 0 and diff_y < 0: # due to arctan giving same values for different quadrants in unit circle (
				# i.e. arctan(y/x) = arctan(-y/-x), but we need them to be different).
				theta_miro_home += 2 * np.pi
			elif diff_x < 0:
				theta_miro_home += np.pi

			theta_diff = theta_miro_home - current_pose[2]
			if abs(theta_diff) < np.pi:
				omega = A * theta_diff
			elif theta_diff > 0:
				omega = A * (theta_diff - 2 * np.pi)
			else:
				omega = A * (theta_diff + 2 * np.pi)
			vel = min(distance_from_home, vel_max)
			self.msg_wheels.twist.linear.x = vel
			self.msg_wheels.twist.angular.z = omega
			self.pub_wheels.publish(self.msg_wheels)
			# print("Distance from home is: ", distance_from_home)

		# turn around to face the correct angular pose
		self.msg_wheels.twist.linear.x = 0
		angular_diff = target_pose[2] - current_pose[2]
		while abs(angular_diff) > 0.05:
			if abs(angular_diff) < np.pi:
				omega = A * angular_diff
			elif angular_diff > 0:
				omega = A * (angular_diff - 2 * np.pi)
			else:
				omega = A * (angular_diff + 2 * np.pi)
			self.msg_wheels.twist.angular.z = omega
			self.pub_wheels.publish(self.msg_wheels)
			current_pose = self.body_pose.copy()
			angular_diff = target_pose[2] - current_pose[2]

		self.msg_wheels.twist.linear.x = 0
		self.msg_wheels.twist.angular.z = 0
		self.pub_wheels.publish(self.msg_wheels)
		self.heading_home = False
		print("Reached home!")
		time.sleep(2) # pause for a couple of seconds before continuing

########################################################################################################################
# The main portion of the program that starts the loop, runs the MiRo controller and updates all the model variables
# TODO reset the eligibility trace when MiRo has found a reward
	def main(self):

		t_replay = 0 # s
		t_last_command = 0 # s, time since last motor command
		coords_prev = self.body_pose[0:2]
		theta_prev = self.body_pose[2]
		time.sleep(2)  # wait for two seconds for subscriber messages to update at the start
		# self.intrinsic_e = np.ones(self.network_size) # used to test the network with no intrinsic plasticity
		while not rospy.core.is_shutdown():
			rate = rospy.Rate(int(1 / self.delta_t))
			self.t += self.delta_t


			############################################################################################################
			# Network updates

			coords = self.body_pose[0:2]
			theta = self.body_pose[2]
			movement_x = coords[0] - coords_prev[0]
			movement_y = coords[1] - coords_prev[1]
			if movement_x > 0.0 or movement_y > 0.0:  # at least a movement velocity of 0.002 / delta_t is required
				movement = True
			else:
				movement = False
			movement = True
			# Set current variable values to the previous ones
			coords_prev = coords.copy()
			theta_prev = theta.copy()
			place_cell_rates_prev = self.place_cell_rates.copy()
			currents_prev = self.currents.copy()
			intrinsic_e_prev = self.intrinsic_e.copy()
			stp_d_prev = self.stp_d.copy()
			stp_f_prev = self.stp_f.copy()
			I_place_prev = self.I_place.copy()
			I_inh_prev = self.I_inh
			network_weights_prev = self.network_weights.copy()

			action_cell_vals_prev = self.action_cell_vals.copy()
			action_cell_vals_noise_prev = self.action_cell_vals_noise
			weights_pc_ac_prev = self.weights_pc_ac.copy()
			elig_trace_prev = self.elig_trace.copy()


			if self.reward_val == 0:
				self.replay = False
				t_replay = 0
				# Run standard activity during exploration
				# Update the variables
				self.currents = self.update_currents(currents_prev, self.delta_t, intrinsic_e_prev,
				                                     network_weights_prev, place_cell_rates_prev, stp_d_prev, stp_f_prev,
				                                     I_inh_prev, I_place_prev)
				self.place_cell_rates = self.compute_rates(self.currents)
				self.intrinsic_e = self.update_intrinsic_e(intrinsic_e_prev, self.delta_t, place_cell_rates_prev)
				self.stp_d, self.stp_f = self.update_STP(stp_d_prev, stp_f_prev, self.delta_t, place_cell_rates_prev)
				self.I_place = self.compute_place_cell_activities(coords_prev[0], coords_prev[1], 0, movement)
				self.I_inh = self.update_I_inh(I_inh_prev, self.delta_t, self.w_inh, place_cell_rates_prev)

				self.action_cell_vals = self.compute_action_cell_outputs(weights_pc_ac_prev, place_cell_rates_prev)
				self.action_cell_vals_noise = self.theta_to_action_cell(theta_prev)
				# print(self.action_cell_vals_noise, self.action_cell_vals)
				self.elig_trace = self.update_eligibility_trace(elig_trace_prev, place_cell_rates_prev,
				                                                action_cell_vals_prev, action_cell_vals_noise_prev,
				                                                self.tau_elig, self.delta_t) # noise is calculated from the
				# whatever the current theta position of MiRo is determined by the random walk (for now).
				# self.weights_pc_ac = self.normalise_weights_pc_ac(self.weight_updates(weights_pc_ac_prev, -0.1,
				#                                                   elig_trace_prev, self.eta, 0.1, self.delta_t))


			else:
				# Run a reverse replay
				if not self.replay:
					print('Running reverse replay event')
				self.replay = True
				t_replay += self.delta_t
				#
				# if (1 < t_replay < 1.1) or (3 < t_replay < 3.1) or (5 < t_replay < 5.1) or (7 < t_replay < 7.1):
				# 	# if (replay_step < 100):
				# 	I_place = self.I_place
				# else:
				# 	I_place = np.zeros(self.network_size_pc)
				# # set variables at the next time step to the ones now
				# self.currents = self.update_currents(currents_prev, self.delta_t, intrinsic_e_prev,
				#                                      network_weights_prev, place_cell_rates_prev, stp_d_prev, stp_f_prev,
				#                                      I_inh_prev, I_place, replay=self.replay)
				# self.rates = self.compute_rates(self.currents)
				# self.intrinsic_e = self.update_intrinsic_e(intrinsic_e_prev, self.delta_t, place_cell_rates_prev)
				# self.stp_d, self.stp_f = self.update_STP(stp_d_prev, stp_f_prev, self.delta_t, place_cell_rates_prev)
				# self.I_inh = self.update_I_inh(I_inh_prev, self.delta_t, self.w_inh, place_cell_rates_prev)
				self.weights_pc_ac = self.normalise_weights_pc_ac(self.weight_updates(weights_pc_ac_prev,
				                                            self.reward_val, elig_trace_prev, self.eta, 0.1,
				                                                                      self.delta_t))

				if t_replay > 9:
					# finish running the replay event, reset variables that need resetting, and go home
					self.replay = False
					self.intrinsic_e = self.intrinsic_e_reset.copy()
					self.elig_trace = np.zeros((self.network_size_ac, self.network_size_pc)) # reset the eligibility
					# trace
					self.head_to_position(self.home_pose)

			############################################################################################################
			# Miro controller
			if self.heading_home:
				self.head_to_position(self.home_pose)

			if not self.replay:

				if self.t - t_last_command > 2: # send a command once every two seconds
					# self.target_theta = self.action_cell_to_theta(self.action_cell_vals_noise)
					self.target_theta = self.random_walk(theta_prev, self.target_theta)
					t_last_command = self.t

					# np.savetxt('rates.csv', place_cell_rates_prev, delimiter=',')
					# np.savetxt('weights.csv', weights_pc_ac_prev, delimiter=',')
					# np.savetxt('coords.csv', coords_prev, delimiter=',')
					print("The target angle is: ", self.target_theta / 2 / np.pi * 360)
					print("The current position is: ", self.body_pose[0:2])
					print('The max and min values of eligibility trace is: ', np.max(self.elig_trace),
					      np.min(self.elig_trace))
					ac_direction = self.action_cell_to_theta(self.action_cell_vals)
					ac_magnitude = np.sqrt((self.action_cell_vals[0] - self.action_cell_vals[2])**2 + (
							self.action_cell_vals[1] - self.action_cell_vals[3])**2)
					print("The action cell output direction is %.2f rad and maginitude is %.2f"
					      % (ac_direction, ac_magnitude))
					print("-------------------------------------------------------------------------------------------")

				self.miro_controller(self.target_theta, theta_prev)

			else:
				self.stop_movement()

			# TODO ensure it only save the past 1 min of data
			np.save('data/intrinsic_e.npy', self.intrinsic_e)
			np.save('data/rates_data.npy', self.place_cell_rates)
			np.save('data/place_data.npy', self.I_place)
			np.save('data/action_cells_vals.npy', self.action_cell_vals)
			np.save('data/weights.npy', self.weights_pc_ac)

			self.time_series.append(self.t)
			self.rates_series.append(self.place_cell_rates)
			self.intrinsic_e_series.append(self.intrinsic_e)

			rate.sleep()

if __name__ == '__main__':
	robo_replay = RoboReplay()
	robo_replay.main()
