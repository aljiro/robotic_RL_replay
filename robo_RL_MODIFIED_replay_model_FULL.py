#!/usr/bin/python
'''
This is the full robot replay script using the modified learning rule for the weights. It subscribes (
via ROS) to the robot's
coordinates, produces the rate
activities according to the model, and replays once a reward has been reached, which is gathered by subscribing to
the reward topic.
Personal note: see diary entry Wed 6th May 2020 regarding my notes on this modified learning rule
'''

import rospy
from std_msgs.msg import UInt8
from geometry_msgs.msg import Pose2D, TwistStamped
import numpy as np
import os
import signal
import sys
import csv
import time
import miro2 as miro
import robot_reply_RL


class RobotReplayMain(robot_reply_RL.NetworkSetup):
	# Inherits the methods from the NetworkSetup class in the main "robot_replay_RL" module. That provides all the
	# methods for network setup and dynamics. See class for a full list of variable names and methods.

########################################################################################################################
	# Modified methods for updating the eligibility trace and weights
	def update_eligibility_trace_modified(self, current_eligibility_trace, action_cells, action_cell_noise, tau,
	                             delta_t):
		'''
		#TODO needs testing
		:param current_eligibility_trace: numpy array, 4x100
		:param action_cells: numpy array, 4x1 of the current action cell values
		:param action_cell_noise: numpy array, 4x1 of the current action cell values with added noise
		:param tau: float, time constant
		:param delta_t: float, time step
		:return: numpy array, 4x100 updated array of the eligibility trace
		'''

		A = 1 # Constant
		updated_eligibility_trace = current_eligibility_trace.copy()
		for i in range(len(current_eligibility_trace)):  # iterate through rows, 72 of them, with index i
			Y = (action_cell_noise[i] - action_cells[i])
			updated_eligibility_trace[i] += (-current_eligibility_trace[i] / tau + A * Y) * delta_t

		return updated_eligibility_trace

	def weight_updates_modified(self, weights_current, reward_pred_error, elig, action_cells, place_cells, eta, sigma, \
	                                                                                                delta_t):
		'''
		# TODO needs testing
		:param weights_current: numpy array, 72x100
		:param reward_pred_error: float
		:param elig: numpy array, 72x1 of the eligibility trace
		:param eta: float, learning rate
		:param sigma: float, standard deviation in the action cell output noise
		:param delta_t: float, time step
		:return: numpy array, 72x100 updated values for the weights
		'''

		weights_updated = weights_current.copy()
		sigma_squared = sigma ** 2
		for i in range(len(weights_current[:, 0])):  # iterate through rows, four of them, with index i
			for j in range(len(weights_current[0, :])):  # iterate through columns, 100 of them, with index j
				weights_updated[i, j] += (eta * reward_pred_error * (1 / sigma_squared) * action_cells[i] *
				                          place_cells[j] * (1 - action_cells[i]) * elig[i]) * delta_t

		return self.normalise_weights_pc_ac(weights_updated)

	# Overrides the method in the supeclass method in the robot_replay_RL script
	def avoid_wall(self):
		t_init = time.time()
		t_sim_init = self.t
		self.msg_wheels.twist.linear.x = -0.1
		self.msg_wheels.twist.angular.z = 0
		while time.time() - t_init < 1:
			self.pub_wheels.publish(self.msg_wheels)
			# try giving a negative reward if MiRo moves into a wall
			if self.t - t_sim_init < 0.5: # update the weights for 0.5s
				self.weights_pc_ac = self.weight_updates_modified(self.weights_pc_ac, -1,
				                                                  self.elig_trace,
				                                                  self.action_cell_vals,
				                                                  self.place_cell_rates,
				                                                  self.eta, self.sigma, self.delta_t)
				self.t += 0.01

		self.elig_trace = np.zeros(self.network_size_ac)  # reset the eligibility trace
		self.intrinsic_e = self.intrinsic_e_reset # reset the intrinsic excitability (for fairness since the elig
		# trace is reset as well)
		p_anti_clock = 1
		self.msg_wheels.twist.linear.x = 0
		if np.random.rand() < p_anti_clock:
			self.msg_wheels.twist.angular.z = 2
		else:
			self.msg_wheels.twist.angular.z = -2
		while time.time() - t_init < 2.5:
			self.pub_wheels.publish(self.msg_wheels)

	def main(self):
		self.elig_trace = np.zeros(self.network_size_ac) # modifying the eligibility trace shape
		self.t = 0 # s
		t_replay = 0 # s
		t_last_command = 0 # s, time since last motor command
		coords_prev = self.body_pose[0:2]
		theta_prev = self.body_pose[2]
		time.sleep(2)  # wait for two seconds for subscriber messages to update at the start
		# self.intrinsic_e = np.ones(self.network_size) # used to test the network with no intrinsic plasticity

		trial_times = [] # used to store the time taken in a given trial to reach the reward
		t_trial = 0
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
			network_weights_prev = self.network_weights_pc.copy()

			action_cell_vals_prev = self.action_cell_vals.copy()
			action_cell_vals_noise_prev = self.action_cell_vals_noise
			weights_pc_ac_prev = self.weights_pc_ac.copy()
			elig_trace_prev = self.elig_trace.copy()


			if self.reward_val == 0:
				# Run standard activity during exploration. No weights changes here, since R=0

				self.replay = False
				t_replay = 0
				t_trial += self.delta_t

				# Update the variables
				# Place cells
				self.currents = self.update_currents(currents_prev, self.delta_t, intrinsic_e_prev,
				                                     network_weights_prev, place_cell_rates_prev, stp_d_prev, stp_f_prev,
				                                     I_inh_prev, I_place_prev)
				self.place_cell_rates = self.compute_rates(self.currents)
				self.intrinsic_e = self.update_intrinsic_e(intrinsic_e_prev, self.delta_t, place_cell_rates_prev)
				self.stp_d, self.stp_f = self.update_STP(stp_d_prev, stp_f_prev, self.delta_t, place_cell_rates_prev)
				self.I_place = self.compute_place_cell_activities(coords_prev[0], coords_prev[1], 0, movement)
				self.I_inh = self.update_I_inh(I_inh_prev, self.delta_t, self.w_inh, place_cell_rates_prev)

				# Action cells
				self.action_cell_vals = self.compute_action_cell_outputs(weights_pc_ac_prev, place_cell_rates_prev)
				self.action_cell_vals_noise = self.theta_to_action_cell(theta_prev)
				# print(self.action_cell_vals_noise, self.action_cell_vals)
				self.elig_trace = self.update_eligibility_trace_modified(elig_trace_prev,
				                                                action_cell_vals_prev,
				                                                action_cell_vals_noise_prev,
				                                                self.tau_elig, self.delta_t)

			else:
				# Run a reverse replay
				if not self.replay:
					print('Running reverse replay event')
					# Don't need to do this for this learning rule
					# elig_trace_at_reward = np.zeros(self.network_size_ac)
					# for i in range(72):
					# 	for j in range(100):
					# 		if elig_trace_prev[i, j] > 0:
					# 			elig_trace_at_reward[i, j] = 0.1
					# 		elif elig_trace_prev[i, j] < 0:
					# 			elig_trace_at_reward[i, j] = -0.1

					if t_trial != 0 and t_trial > 1:
						trial_times.append(t_trial)
					t_trial = 0
				self.replay = True
				t_replay += self.delta_t

				if (1 < t_replay < 1.1):
					# To run reverse replays
					I_place = 2 * self.I_place
				else:
					I_place = np.zeros(self.network_size_pc)

				# if t_replay == 0.01 or 1.01 > t_replay > 1 or 1.11 > t_replay > 1.1 or 1.21 > t_replay > 1.2 or 1.31 \
				# 		> t_replay > 1.3 or 1.41 > t_replay > 1.4 or 1.51 > t_replay > 1.5:
				# 	print(t_replay)
				# 	np.save('data/eligibility_trace_t_replay=' + str(t_replay) + 's.npy', self.elig_trace)

				# Update variables
				# Place cells (should initiate a reverse replay)
				self.currents = self.update_currents(currents_prev, self.delta_t, intrinsic_e_prev,
				                                     network_weights_prev, place_cell_rates_prev, stp_d_prev, stp_f_prev,
				                                     I_inh_prev, I_place, replay=self.replay)
				self.place_cell_rates = self.compute_rates(self.currents)
				self.intrinsic_e = self.update_intrinsic_e(intrinsic_e_prev, self.delta_t, place_cell_rates_prev)
				self.stp_d, self.stp_f = self.update_STP(stp_d_prev, stp_f_prev, self.delta_t, place_cell_rates_prev)
				self.I_inh = self.update_I_inh(I_inh_prev, self.delta_t, self.w_inh, place_cell_rates_prev)

				# Action cells and weights
				self.weights_pc_ac = self.weight_updates_modified(weights_pc_ac_prev, self.reward_val,
				                                         elig_trace_prev,
				                                         action_cell_vals_prev,
				                                         place_cell_rates_prev,
				                                         self.eta, self.sigma, self.delta_t)
				# self.elig_trace = self.update_eligibility_trace_modified(elig_trace_prev,
				#                                                          action_cell_vals_prev,
				#                                                          action_cell_vals_noise_prev,
				#                                                          self.tau_elig, self.delta_t)
				self.action_cell_vals = self.compute_action_cell_outputs(weights_pc_ac_prev, place_cell_rates_prev)
				# self.action_cell_vals_noise = self.action_cell_vals + self.compute_action_cell_outputs(
				# 	weights_pc_ac_prev + elig_trace_at_reward, place_cell_rates_prev)

				if t_replay > 2:
					# finish running the replay event, reset variables that need resetting, and go to a random position
					self.replay = False
					self.intrinsic_e = self.intrinsic_e_reset.copy()
					self.elig_trace = np.zeros(self.network_size_ac)
					self.head_random_start_position = True
					# self.heading_home = True
					theta_prev = self.body_pose[2] # resets

			############################################################################################################
			# Miro controller
			if self.heading_home:
				print("Heading home...")
				self.head_to_position(self.home_pose)
				print("Reached home!")
				self.heading_home = False
				theta_prev = self.body_pose[2] # need to reset for the controller below
			elif self.head_random_start_position:
				randx = 1.4 * np.random.random() - 0.7 # between -0.7 and 0.7
				randy = -0.7 * np.random.random() # between 0 and -0.7
				randtheta = np.random.random() * (2 * np.pi - 0.0001) # between 0 and 2*pi (minus a little to avoid
				# 2*pi itself
				random_start_position = np.array((randx, randy, randtheta))
				# print("Heading to a random position at location ", random_start_position)
				self.head_to_position(random_start_position)
				print("Reached the start position. Starting experiment-trial number " + str(self.experiment_number) +
				      "-" + str(len(trial_times)) + ".")
				self.head_random_start_position = False
				theta_prev = self.body_pose[2] # need to reset for the controller below

			if not self.replay:

				if self.t - t_last_command > 0.5: # send a command once every half a second
					# self.target_theta, _ = self.action_cell_to_theta_and_magnitude(self.action_cell_vals_noise)
					ac_direction, ac_magnitude = self.action_cell_to_theta_and_magnitude(self.action_cell_vals)
					if ac_magnitude >= 1:
						ac_direction_noise = self.add_noise_to_action_cell_outputs(self.action_cell_vals, self.sigma)
						self.target_theta, _ = self.action_cell_to_theta_and_magnitude(ac_direction_noise)
						# t_last_command = self.t
					else:
						self.target_theta = self.random_walk(theta_prev, self.target_theta)
					t_last_command = self.t

					# np.savetxt('rates.csv', place_cell_rates_prev, delimiter=',')
					# np.savetxt('weights.csv', weights_pc_ac_prev, delimiter=',')
					# np.savetxt('coords.csv', coords_prev, delimiter=',')
					# print("The target angle is: ", self.target_theta / 2 / np.pi * 360)
					# print("The current position is: ", self.body_pose[0:2])
					# print('The max and min values of eligibility trace is: ', np.max(self.elig_trace),
					#       np.min(self.elig_trace))
					# print("The action cell output direction is %.2f rad and maginitude is %.2f"
					#       % (ac_direction, ac_magnitude))
					# print("-------------------------------------------------------------------------------------------")

				# self.target_theta = 0 # For testing purposes
				self.miro_controller(self.target_theta, theta_prev)

			else:
				self.stop_movement()

			np.save('data/intrinsic_e.npy', self.intrinsic_e)
			np.save('data/rates_data.npy', self.place_cell_rates)
			np.save('data/place_data.npy', self.I_place)
			np.save('data/action_cells_vals.npy', self.action_cell_vals)
			np.save('data/weights.npy', self.weights_pc_ac)
			# np.save('data/eligibility_trace.npy', self.elig_trace)

			# TODO ensure it only save the past 1 min of data
			# self.time_series.append(self.t)
			# self.rates_series.append(self.place_cell_rates)
			# self.intrinsic_e_series.append(self.intrinsic_e)

			if len(trial_times) == self.total_number_of_trials:
				with open('data/trial_times/trial_times_MODIFIED_REPLAY_FULL.csv', 'a') as trial_times_file:
					wr = csv.writer(trial_times_file, quoting=csv.QUOTE_ALL)
					wr.writerow([self.experiment_number] + trial_times)
				print("Experiment " + str(self.experiment_number) + " finished. Trial times are \n")
				print(trial_times)
				print("\n -------------------------------------------------------------------------------- \n")
				break # break out of the rospy while loop

			rate.sleep()

if __name__ == '__main__':
	# for tau_elig in [1.0/25, 1.0/5, 1.0, 5.0]:
	for tau_elig in [5.0]:
		# for eta in [0.01, 0.1, 1.0, 10.0]:
		for eta in [0.001]:
			with open('data/trial_times/trial_times_MODIFIED_REPLAY_FULL.csv', 'a') as trial_times_file:
				wr = csv.writer(trial_times_file, quoting=csv.QUOTE_ALL)
				wr.writerow("")
				wr.writerow(["tau_elig=" + str(tau_elig), "eta=" + str(eta)])

			for experiment in range(1, 21):
				robo_replay = RobotReplayMain(tau_elig, eta, experiment)
				robo_replay.main()
