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
import robot_reply_RL
import gc
import json
from rl_logger import RLLogger

logger = RLLogger( True )


class RobotReplayMain(robot_reply_RL.NetworkSetup):
	# Inherits the methods from the NetworkSetup class in the main "robot_replay_RL" module. That provides all the
	# methods for network setup and dynamics. See class for a full list of variable names and methods.

	def signal_handler(self, sig, frame): # handles tasks for when closing the program with ctrl+c
		print('\nSaving trial data')
		np.save('data/time_series_without_inh.npy', self.time_series)
		np.save('data/rates_series_without_inh.npy', self.rates_series)
		np.save('data/intrinsic_e_series_without_inh.npy', self.intrinsic_e_series)

		# clean up the temporary data files
		os.remove('data/intrinsic_e.npy')
		os.remove('data/rates_data.npy')
		os.remove('data/place_data.npy')

		sys.exit(0)
########################################################################################################################
# The main portion of the program that starts the ROS loop, runs the MiRo controller and updates all the model variables
	def main(self):
		self.t = 0 # s
		t_replay = 0 # s
		t_last_command = 0 # s, time since last motor command
		coords_prev = self.body_pose[0:2]
		theta_prev = self.body_pose[2]
		time.sleep(2)  # wait for two seconds for subscriber messages to update at the start
		# self.intrinsic_e = np.ones(self.network_size) # used to test the network with no intrinsic plasticity

		trial_times = [] # used to store the time taken in a given trial to reach the reward
		random_times = []
		hitting_count = []
		ac_activations = []
		total_counts = []
		t_trial = 0
		t_action = 0.0
		t_random = 0.0
		self.wall_hitting = 0
		ac_average = 0.0
		total_counter = 0.0
		


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
				self.currents = self.update_currents(currents_prev, 
													 self.delta_t, 
													 intrinsic_e_prev,
				                                     network_weights_prev, 
													 place_cell_rates_prev, 
													 stp_d_prev, 
													 stp_f_prev,
				                 	                 I_inh_prev, 
													 I_place_prev)

				self.place_cell_rates = self.compute_rates(self.currents)

				self.intrinsic_e = self.update_intrinsic_e(intrinsic_e_prev, 
														   self.delta_t, 
														   place_cell_rates_prev)

				self.stp_d, self.stp_f = self.update_STP(stp_d_prev, 
														 stp_f_prev, 
														 self.delta_t, 
														 place_cell_rates_prev)

				self.I_place = self.compute_place_cell_activities(coords_prev[0], 
																  coords_prev[1], 
																  0, 
																  movement)
				
				self.I_inh = self.update_I_inh(I_inh_prev, 
											   self.delta_t, 
											   self.w_inh, 
											   place_cell_rates_prev)

				# Action cells
				self.action_cell_vals = self.compute_action_cell_outputs(weights_pc_ac_prev, 
																		 place_cell_rates_prev)
				
				self.action_cell_vals_noise = self.theta_to_action_cell(theta_prev)
				# print(self.action_cell_vals_noise, self.action_cell_vals)
				self.elig_trace = self.update_eligibility_trace(elig_trace_prev,
				                                                place_cell_rates_prev,
				                                                action_cell_vals_prev,
				                                                action_cell_vals_noise_prev,
				                                                self.tau_elig, self.delta_t)

			else:
				# Run a reverse replay
				if not self.replay:
					print('Running reverse replay event')
					elig_trace_at_reward = np.zeros((self.network_size_ac, self.network_size_pc))
					# Equation 17
					for i in range(72):
						for j in range(100):
							if elig_trace_prev[i, j] > 0:
								elig_trace_at_reward[i, j] = 0.1
							elif elig_trace_prev[i, j] < 0:
								elig_trace_at_reward[i, j] = -0.1
								# elig_trace_at_reward[i, j] = 0

					if t_trial != 0 and t_trial > 1:
						trial_times.append(t_trial)
						random_times.append(t_random)
						hitting_count.append(self.wall_hitting)
						ac_activations.append(ac_average/total_counter)
						total_counts.append(total_counter)
						print("New trial no. " + str(len(random_times)))
					t_trial = 0 # Maybe this should be inside the if
					t_action = 0.0
					t_random = 0.0
					ac_average = 0.0
					total_counter = 0.0
					self.wall_hitting = 0

				self.replay = True
				t_replay += self.delta_t

				if (1 < t_replay < 1.1):
					# To run reverse replays
					I_place = self.I_place
				else:
					I_place = np.zeros(self.network_size_pc)

				# if t_replay == 0.01 or 1.01 > t_replay > 1 or 1.11 > t_replay > 1.1 or 1.21 > t_replay > 1.2 or 1.31 \
				# 		> t_replay > 1.3 or 1.41 > t_replay > 1.4 or 1.51 > t_replay > 1.5:
				# 	print(t_replay)
				# 	np.save('data/eligibility_trace_t_replay=' + str(t_replay) + 's.npy', self.elig_trace)

				# Update variables
				# Place cells (should initiate a reverse replay)
				self.currents = self.update_currents(currents_prev, 
													 self.delta_t, 
													 intrinsic_e_prev,
				                                     network_weights_prev, 
													 place_cell_rates_prev, 
													 stp_d_prev, 
													 stp_f_prev,
				                                     I_inh_prev, 
													 I_place, 
													 replay=self.replay)

				self.place_cell_rates = self.compute_rates(self.currents)

				self.intrinsic_e = self.update_intrinsic_e(intrinsic_e_prev, 
														   self.delta_t, 
														   place_cell_rates_prev)

				self.stp_d, self.stp_f = self.update_STP(stp_d_prev, 
														 stp_f_prev, 
														 self.delta_t, 
														 place_cell_rates_prev)

				self.I_inh = self.update_I_inh(I_inh_prev, 
											   self.delta_t, 
											   self.w_inh, 
											   place_cell_rates_prev)

				# Action cells and weights
				self.weights_pc_ac = self.weight_updates(weights_pc_ac_prev, 
														 self.reward_val,
				                                         elig_trace_prev,
				                                         self.eta, 
														 self.sigma, 
														 self.delta_t)

				self.elig_trace = self.update_eligibility_trace(elig_trace_prev,
				                                                place_cell_rates_prev,
				                                                action_cell_vals_prev,
				                                                action_cell_vals_noise_prev,
				                                                self.tau_elig, self.delta_t)

				self.action_cell_vals = self.compute_action_cell_outputs(weights_pc_ac_prev, 
																		 place_cell_rates_prev)

				
				self.action_cell_vals_noise = self.action_cell_vals + self.compute_action_cell_outputs(
					weights_pc_ac_prev + elig_trace_at_reward, place_cell_rates_prev)


				if t_replay > 2:
					# finish running the replay event, reset variables that need resetting, and go to a random position
					self.replay = False
					self.intrinsic_e = self.intrinsic_e_reset.copy()
					self.elig_trace = np.zeros((self.network_size_ac, self.network_size_pc))
					self.head_random_start_position = True
					# self.heading_home = True
					theta_prev = self.body_pose[2] # resets

			############################################################################################################

			# If t_trial > 120s, then MiRo has been searching too long. Start a new trial.
			if t_trial > 120:
				print('Took too long')
				self.intrinsic_e = self.intrinsic_e_reset.copy()
				self.elig_trace = np.zeros((self.network_size_ac, self.network_size_pc))
				trial_times.append(120)
				random_times.append(t_random)
				hitting_count.append(self.wall_hitting)
				ac_activations.append(ac_average/total_counter)
				total_counts.append(total_counter)
				t_action = 0.0
				t_random = 0.0
				t_trial = 0
				total_counter = 0.0
				ac_average = 0.0
				self.wall_hitting = 0				
				print("New trial (Time out) no. " + str(len(random_times)))
				self.head_random_start_position = True

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
				print("Heading to a random position at location ", random_start_position)
				self.head_to_position(random_start_position)
				print("Reached the start position. Starting experiment-trial number " + str(self.experiment_number) +
				      "-" + str(len(trial_times)) + ".")
				gc.collect()
				self.head_random_start_position = False
				theta_prev = self.body_pose[2] # need to reset for the controller below

			if not self.replay:

				if self.t - t_last_command > 0.5: # send a command once every half a second
					# self.target_theta, _ = self.action_cell_to_theta_and_magnitude(self.action_cell_vals_noise)
					ac_direction, ac_magnitude = self.action_cell_to_theta_and_magnitude(self.action_cell_vals)
					ac_average += ac_magnitude
					total_counter += 1.0

					if ac_magnitude >= 1:
    					# Choosing action cells
						t_action = t_action + 1.0
						ac_direction_noise = self.add_noise_to_action_cell_outputs(self.action_cell_vals, self.sigma)
						self.target_theta, _ = self.action_cell_to_theta_and_magnitude(ac_direction_noise)
						# t_last_command = self.t
					else:
    					# Random walk
						t_random = t_random + 1.0
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
			np.save('data/eligibility_trace.npy', self.elig_trace)

			# TODO ensure it only save the past 1 min of data
			# self.time_series.append(self.t)
			# self.rates_series.append(self.place_cell_rates)
			# self.intrinsic_e_series.append(self.intrinsic_e)

			if len(trial_times) == self.total_number_of_trials:
				data = {"trial_times": trial_times, 
						"random_times": random_times, 
						"hitting": hitting_count, 
						"activations": ac_activations, 
						"total_counts": total_counts}

				logger.saveAll( self.experiment_number, data )				
					
				print("Experiment " + str(self.experiment_number) + " finished. Trial times are \n")
				# print("Trial times: " + str(trial_times))
				# print("Percentage spent in random walk: " + str(random_times))
				print("\n -------------------------------------------------------------------------------- \n")
				break # break out of the rospy while loop

			rate.sleep()

if __name__ == '__main__':
	
	with open('parameters.json', 'r') as f:
  		parameters = json.load(f)

	no_trials = parameters["no_trials"]
	no_exps = parameters["no_experiments"]

	for tau_elig in [parameters["figure4"]["with_replay"]["tau_elig"]]:
		for eta in [parameters["figure4"]["with_replay"]["eta"]]:
			logger.initialize(tau_elig, eta)

			for experiment in range(1, no_exps):
				robo_replay = RobotReplayMain(tau_elig, eta, no_trials, experiment)
				robo_replay.main()
