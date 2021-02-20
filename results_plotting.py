#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as colors
import csv
import copy

# Storing the raw data
result_parameters = {'tau=0.04_eta=0.001': [],
	'tau=0.04_eta=0.01': [],
    'tau=0.04_eta=0.1': [],
    'tau=0.04_eta=1': [],
    'tau=0.04_eta=10': [],
	'tau=0.2_eta=0.001': [],
    'tau=0.2_eta=0.01': [],
    'tau=0.2_eta=0.1': [],
    'tau=0.2_eta=1': [],
    'tau=0.2_eta=10': [],
    'tau=1_eta=0.001': [],
    'tau=1_eta=0.002': [],
    'tau=1_eta=0.005': [],
    'tau=1_eta=0.01': [],
    'tau=1_eta=0.1': [],
    'tau=1_eta=1': [],
    'tau=1_eta=10': [],
    'tau=5_eta=0.001': [],
    'tau=5_eta=0.01': [],
    'tau=5_eta=0.1': [],
    'tau=5_eta=1': [],
    'tau=5_eta=10': [],}

results_full_nonreplay = copy.deepcopy(result_parameters)
results_full_replay = copy.deepcopy(result_parameters)
results_modified = copy.deepcopy(result_parameters)

with open('data/trial_times/trial_times_NON_REPLAY_plot.csv', newline='') as file:
	data = csv.reader(file, delimiter=',')
	step = 42
	for i, row in enumerate(data):
		if (2 <= i < step):
			results_full_nonreplay['tau=0.04_eta=0.001'].append([float(j) for j in row[1:]])
		if (step + 2 <= i < 2 * step):
			results_full_nonreplay['tau=0.04_eta=0.01'].append([float(j) for j in row[1:]])
		if (2 * step + 2 <= i < 3 * step):
			results_full_nonreplay['tau=0.04_eta=0.1'].append([float(j) for j in row[1:]])
		if (3 * step + 2 <= i < 4 * step):
			results_full_nonreplay['tau=0.04_eta=1'].append([float(j) for j in row[1:]])
		if (4 * step + 2 <= i < 5 * step):
			results_full_nonreplay['tau=0.04_eta=10'].append([float(j) for j in row[1:]])

		if (5 * step + 2 <= i < 6 * step):
			results_full_nonreplay['tau=0.2_eta=0.001'].append([float(j) for j in row[1:]])
		if (6 * step + 2 <= i < 7 * step):
			results_full_nonreplay['tau=0.2_eta=0.01'].append([float(j) for j in row[1:]])
		if (7 * step + 2 <= i < 8 * step):
			results_full_nonreplay['tau=0.2_eta=0.1'].append([float(j) for j in row[1:]])
		if (8 * step + 2 <= i < 9 * step):
			results_full_nonreplay['tau=0.2_eta=1'].append([float(j) for j in row[1:]])
		if (9 * step + 2 <= i < 10 * step):
			results_full_nonreplay['tau=0.2_eta=10'].append([float(j) for j in row[1:]])

		if (10 * step + 2 <= i < 11 * step):
			results_full_nonreplay['tau=1_eta=0.001'].append([float(j) for j in row[1:]])
		if (11 * step + 2 <= i < 12 * step):
			results_full_nonreplay['tau=1_eta=0.01'].append([float(j) for j in row[1:]])
		if (12 * step + 2 <= i < 13 * step):
			results_full_nonreplay['tau=1_eta=0.1'].append([float(j) for j in row[1:]])
		if (13 * step + 2 <= i < 14 * step):
			results_full_nonreplay['tau=1_eta=1'].append([float(j) for j in row[1:]])
		if (14 * step + 2 <= i < 15 * step):
			results_full_nonreplay['tau=1_eta=10'].append([float(j) for j in row[1:]])

		if (15 * step + 2 <= i < 16 * step):
			results_full_nonreplay['tau=5_eta=0.001'].append([float(j) for j in row[1:]])
		if (16 * step + 2 <= i < 17 * step):
			results_full_nonreplay['tau=5_eta=0.01'].append([float(j) for j in row[1:]])
		if (17 * step + 2 <= i < 18 * step):
			results_full_nonreplay['tau=5_eta=0.1'].append([float(j) for j in row[1:]])
		if (18 * step + 2 <= i < 19 * step):
			results_full_nonreplay['tau=5_eta=1'].append([float(j) for j in row[1:]])
		if (19 * step + 2 <= i < 20 * step):
			results_full_nonreplay['tau=5_eta=10'].append([float(j) for j in row[1:]])

with open('data/trial_times/trial_times_WITH_REPLAY_plot.csv', newline='') as file:
	data = csv.reader(file, delimiter=',')
	for i, row in enumerate(data):
		if (2 <= i < step):
			results_full_replay['tau=0.04_eta=0.001'].append([float(j) for j in row[1:]])
		if (step + 2 <= i < 2 * step):
			results_full_replay['tau=0.04_eta=0.01'].append([float(j) for j in row[1:]])
		if (2 * step + 2 <= i < 3 * step):
			results_full_replay['tau=0.04_eta=0.1'].append([float(j) for j in row[1:]])
		if (3 * step + 2 <= i < 4 * step):
			results_full_replay['tau=0.04_eta=1'].append([float(j) for j in row[1:]])
		if (4 * step + 2 <= i < 5 * step):
			results_full_replay['tau=0.04_eta=10'].append([float(j) for j in row[1:]])

		if (5 * step + 2 <= i < 6 * step):
			results_full_replay['tau=0.2_eta=0.001'].append([float(j) for j in row[1:]])
		if (6 * step + 2 <= i < 7 * step):
			results_full_replay['tau=0.2_eta=0.01'].append([float(j) for j in row[1:]])
		if (7 * step + 2 <= i < 8 * step):
			results_full_replay['tau=0.2_eta=0.1'].append([float(j) for j in row[1:]])
		if (8 * step + 2 <= i < 9 * step):
			results_full_replay['tau=0.2_eta=1'].append([float(j) for j in row[1:]])
		if (9 * step + 2 <= i < 10 * step):
			results_full_replay['tau=0.2_eta=10'].append([float(j) for j in row[1:]])

		if (10 * step + 2 <= i < 11 * step):
			results_full_replay['tau=1_eta=0.001'].append([float(j) for j in row[1:]])
		if (11 * step + 2 <= i < 12 * step):
			results_full_replay['tau=1_eta=0.01'].append([float(j) for j in row[1:]])
		if (12 * step + 2 <= i < 13 * step):
			results_full_replay['tau=1_eta=0.1'].append([float(j) for j in row[1:]])
		if (13 * step + 2 <= i < 14 * step):
			results_full_replay['tau=1_eta=1'].append([float(j) for j in row[1:]])
		if (14 * step + 2 <= i < 15 * step):
			results_full_replay['tau=1_eta=10'].append([float(j) for j in row[1:]])

		if (15 * step + 2 <= i < 16 * step):
			results_full_replay['tau=5_eta=0.001'].append([float(j) for j in row[1:]])
		if (16 * step + 2 <= i < 17 * step):
			results_full_replay['tau=5_eta=0.01'].append([float(j) for j in row[1:]])
		if (17 * step + 2 <= i < 18 * step):
			results_full_replay['tau=5_eta=0.1'].append([float(j) for j in row[1:]])
		if (18 * step + 2 <= i < 19 * step):
			results_full_replay['tau=5_eta=1'].append([float(j) for j in row[1:]])
		if (19 * step + 2 <= i < 20 * step):
			results_full_replay['tau=5_eta=10'].append([float(j) for j in row[1:]])

with open('data/trial_times/trial_times_MODIFIED_REPLAY_FULL.csv', newline='') as file:
	data = csv.reader(file, delimiter=',')
	for i, row in enumerate(data):
		if (2 <= i < step):
			results_modified['tau=5_eta=0.001'].append([float(j) for j in row[1:]])

# Getting the averages and standard deviations for each
averages_nonreplay = copy.deepcopy(result_parameters)
averages_replay = copy.deepcopy(result_parameters)
averages_modified = copy.deepcopy(result_parameters)

std_dev_nonreplay = copy.deepcopy(result_parameters)
std_dev_replay = copy.deepcopy(result_parameters)
std_dev_modified = copy.deepcopy(result_parameters)

no_experiments = 40
for key in averages_nonreplay:
	for trial_no in range(20):
		average = 0
		for experiment_no in range(len(results_full_nonreplay[key])):
			average += results_full_nonreplay[key][experiment_no][trial_no]
		average /= no_experiments
		averages_nonreplay[key].append(average)

		standard_dev = 0
		for experiment_no in range(len(results_full_nonreplay[key])):
			standard_dev += (results_full_nonreplay[key][experiment_no][trial_no] - averages_nonreplay[key][trial_no])**2
		standard_dev = np.sqrt(standard_dev / no_experiments)
		std_dev_nonreplay[key].append(standard_dev)

for key in averages_replay:
	for trial_no in range(20):
		average = 0
		for experiment_no in range(len(results_full_replay[key])):
			average += results_full_replay[key][experiment_no][trial_no]
		average /= no_experiments
		averages_replay[key].append(average)

		standard_dev = 0
		for experiment_no in range(len(results_full_replay[key])):
			standard_dev += (results_full_replay[key][experiment_no][trial_no] - averages_replay[key][trial_no])**2
		standard_dev = np.sqrt(standard_dev / no_experiments)
		std_dev_replay[key].append(standard_dev)

for key in ['tau=5_eta=0.001']:
	for trial_no in range(20):
		average = 0
		for experiment_no in range(len(results_modified[key])):
			average += results_modified[key][experiment_no][trial_no]
		average /= no_experiments
		averages_modified[key].append(average)

		standard_dev = 0
		for experiment_no in range(len(results_modified[key])):
			standard_dev += (results_modified[key][experiment_no][trial_no] - averages_modified[key][trial_no])**2
		standard_dev = np.sqrt(standard_dev / no_experiments)
		std_dev_modified[key].append(standard_dev)

#######################################################
# Function for generating a moving average
def generate_moving_average_data(tau_e, eta, modified=False):
	'''

	:param tau_e: string
	:param eta: string
	:return: numpy array, moving average data
	'''
	time_const = 'tau=' + tau_e + '_'
	eta = 'eta=' + eta

	if not modified:
		# calculate moving averages
		moving_average_nonreplay = np.zeros(20)
		moving_average_replay = np.zeros(20)
		moving_average_std_devs_nonreplay = np.zeros(20)
		moving_average_std_devs_replay = np.zeros(20)

		for i in range(20):
			if i == 0:
				moving_average_nonreplay[i] = averages_nonreplay[time_const + eta][i] + \
				                              averages_nonreplay[time_const + eta][i
				                                                                   + 1]
				moving_average_nonreplay[i] /= 2
				moving_average_std_devs_nonreplay[i] = std_dev_nonreplay[time_const + eta][i] + \
				                                       std_dev_nonreplay[time_const +
				                                                         eta][i + 1]
				moving_average_std_devs_nonreplay[i] /= 2

				moving_average_replay[i] = averages_replay[time_const + eta][i] + averages_replay[time_const + eta][i + 1]
				moving_average_replay[i] /= 2
				moving_average_std_devs_replay[i] = std_dev_replay[time_const + eta][i] + std_dev_replay[time_const + eta][
					i + 1]
				moving_average_std_devs_replay[i] /= 2
			elif i == 19:
				moving_average_nonreplay[i] = averages_nonreplay[time_const + eta][i] + \
				                              averages_nonreplay[time_const + eta][i
				                                                                   - 1]
				moving_average_nonreplay[i] /= 2
				moving_average_std_devs_nonreplay[i] = std_dev_nonreplay[time_const + eta][i] + \
				                                       std_dev_nonreplay[time_const +
				                                                         eta][i - 1]
				moving_average_std_devs_nonreplay[i] /= 2

				moving_average_replay[i] = averages_replay[time_const + eta][i] + averages_replay[time_const + eta][i - 1]
				moving_average_replay[i] /= 2
				moving_average_std_devs_replay[i] = std_dev_replay[time_const + eta][i] + std_dev_replay[time_const + eta][
					i - 1]
				moving_average_std_devs_replay[i] /= 2
			else:
				moving_average_nonreplay[i] = averages_nonreplay[time_const + eta][i] + \
				                              averages_nonreplay[time_const + eta][
					                              i - 1] + averages_nonreplay[time_const + eta][i + 1]
				moving_average_nonreplay[i] /= 3
				moving_average_std_devs_nonreplay[i] = std_dev_nonreplay[time_const + eta][i] + \
				                                       std_dev_nonreplay[time_const +
				                                                         eta][i - 1] + std_dev_nonreplay[time_const + eta][
					                                       i + 1]
				moving_average_std_devs_nonreplay[i] /= 3

				moving_average_replay[i] = averages_replay[time_const + eta][i] + averages_replay[time_const + eta][i - 1] + \
				                           averages_replay[time_const + eta][i + 1]
				moving_average_replay[i] /= 3
				moving_average_std_devs_replay[i] = std_dev_replay[time_const + eta][i] + std_dev_replay[time_const + eta][
					i - 1] + \
				                                    std_dev_replay[time_const + eta][i + 1]
				moving_average_std_devs_replay[i] /= 3

		return moving_average_replay, moving_average_nonreplay, moving_average_std_devs_replay, moving_average_std_devs_nonreplay

	else:
		moving_average_modified = np.zeros(20)
		moving_average_std_devs_modified = np.zeros(20)

		for i in range(20):
			if i == 0:
				moving_average_modified[i] = averages_modified[time_const + eta][i] + averages_modified[time_const + eta][i + 1]
				moving_average_modified[i] /= 2
				moving_average_std_devs_modified[i] = std_dev_modified[time_const + eta][i] + std_dev_modified[time_const +
				                                                                                       eta][
					i + 1]
				moving_average_std_devs_modified[i] /= 2
			elif i == 19:
				moving_average_modified[i] = averages_modified[time_const + eta][i] + averages_modified[time_const + eta][i - 1]
				moving_average_modified[i] /= 2
				moving_average_std_devs_modified[i] = std_dev_modified[time_const + eta][i] + std_dev_modified[time_const + eta][
					i - 1]
				moving_average_std_devs_modified[i] /= 2
			else:
				moving_average_modified[i] = averages_modified[time_const + eta][i] + averages_modified[time_const + eta][i - 1] + \
				                           averages_replay[time_const + eta][i + 1]
				moving_average_modified[i] /= 3
				moving_average_std_devs_modified[i] = std_dev_modified[time_const + eta][i] + std_dev_modified[time_const + eta][
					i - 1] + \
				                                    std_dev_modified[time_const + eta][i + 1]
				moving_average_std_devs_modified[i] /= 3

		return moving_average_modified, moving_average_std_devs_modified

def generate_moving_average_best_cases():
	'''

	:return: numpy array, moving average data
	'''

	# collect the raw data
	no_experiments = 40
	no_trials = 30

	best_case_with_replay = []
	best_case_non_replay = []
	with open('data/trial_times/best_case_NON_REPLAY.csv', newline='') as file:
		data = csv.reader(file, delimiter=',')
		for i, row in enumerate(data):
			if (2 <= i < 42):
				best_case_non_replay.append([float(j) for j in row[1:]])

	with open('data/trial_times/best_case_WITH_REPLAY.csv', newline='') as file:
		data = csv.reader(file, delimiter=',')
		for i, row in enumerate(data):
			if (2 <= i < 42):
				best_case_with_replay.append([float(j) for j in row[1:]])

	# compute the averages
	best_case_average_replay = np.zeros(no_trials)
	best_case_average_non_replay = np.zeros(no_trials)
	best_case_std_dev_replay = np.zeros(no_trials)
	best_case_std_dev_non_replay = np.zeros(no_trials)

	for trial_no in range(no_trials):
		average_replay = 0
		average_nonreplay = 0
		for experiment_no in range(no_experiments):
			average_replay += best_case_with_replay[experiment_no][trial_no]
			average_nonreplay += best_case_non_replay[experiment_no][trial_no]
		average_replay /= no_experiments
		average_nonreplay /= no_experiments
		best_case_average_replay[trial_no] = average_replay
		best_case_average_non_replay[trial_no] = average_nonreplay

		standard_dev_replay = 0
		standard_dev_non_replay = 0
		for experiment_no in range(no_experiments):
			standard_dev_replay += (best_case_with_replay[experiment_no][trial_no] - best_case_average_replay[
				trial_no]) ** 2
			standard_dev_non_replay += (best_case_non_replay[experiment_no][trial_no] - best_case_average_non_replay[
				trial_no]) ** 2
		standard_dev_replay = np.sqrt(standard_dev_replay / no_experiments)
		standard_dev_non_replay = np.sqrt(standard_dev_non_replay / no_experiments)
		best_case_std_dev_replay[trial_no] = standard_dev_replay
		best_case_std_dev_non_replay[trial_no] = standard_dev_non_replay

	print(best_case_std_dev_non_replay)
	print(best_case_std_dev_replay)

	# calculate moving averages
	moving_average_nonreplay = np.zeros(30)
	moving_average_replay = np.zeros(30)
	moving_average_std_devs_nonreplay = np.zeros(30)
	moving_average_std_devs_replay = np.zeros(30)

	for i in range(30):
		if i == 0:
			moving_average_nonreplay[i] = best_case_average_non_replay[i] + best_case_average_non_replay[i + 1]
			moving_average_nonreplay[i] /= 2
			moving_average_std_devs_nonreplay[i] = best_case_std_dev_non_replay[i] + best_case_std_dev_non_replay[i + 1]
			moving_average_std_devs_nonreplay[i] /= 2

			moving_average_replay[i] = best_case_average_replay[i] + best_case_average_replay[i + 1]
			moving_average_replay[i] /= 2
			moving_average_std_devs_replay[i] = best_case_std_dev_replay[i] + best_case_std_dev_replay[i + 1]
			moving_average_std_devs_replay[i] /= 2
		elif i == 29:
			moving_average_nonreplay[i] = best_case_average_non_replay[i] + best_case_average_non_replay[i - 1]
			moving_average_nonreplay[i] /= 2
			moving_average_std_devs_nonreplay[i] = best_case_std_dev_non_replay[i] + best_case_std_dev_non_replay[i - 1]
			moving_average_std_devs_nonreplay[i] /= 2

			moving_average_replay[i] = best_case_average_replay[i] + best_case_average_replay[i - 1]
			moving_average_replay[i] /= 2
			moving_average_std_devs_replay[i] = best_case_std_dev_replay[i] + best_case_std_dev_replay[i - 1]
			moving_average_std_devs_replay[i] /= 2
		else:
			moving_average_nonreplay[i] = best_case_average_non_replay[i] + best_case_average_non_replay[i - 1] + \
			                              best_case_average_non_replay[i + 1]
			moving_average_nonreplay[i] /= 3
			moving_average_std_devs_nonreplay[i] = best_case_std_dev_non_replay[i] + best_case_std_dev_non_replay[i - \
			                                                1] + best_case_std_dev_non_replay[i + 1]
			moving_average_std_devs_nonreplay[i] /= 3

			moving_average_replay[i] = best_case_average_replay[i] + best_case_average_replay[i - 1] + \
			                              best_case_average_replay[i + 1]
			moving_average_replay[i] /= 3
			moving_average_std_devs_replay[i] = best_case_std_dev_replay[i] + best_case_std_dev_replay[i - 1] + \
			                                       best_case_std_dev_replay[i + 1]
			moving_average_std_devs_replay[i] /= 3

	return moving_average_replay, moving_average_nonreplay, moving_average_std_devs_replay, moving_average_std_devs_nonreplay


# ########################################################################################################################
# # Plotting the non-replay best case
# tau_e = '1'
# eta = '0.01'
# fig, ax = plt.subplots(1,1)
# ax.set_title('Non-replay case, ' + '$\\tau_e =$ ' + tau_e + '$, \eta =$ ' + eta)
# _, average_data_moving, _, std_dev_moving = generate_moving_average_data(tau_e, eta)
# ax.plot(np.arange(1, 21), average_data_moving, color='#ff7f0e', label='$\\tau_e = 1s$ \n $\eta = '
#                                                                              '0.01$')
# ax.set_ylim(0,60)
# ax.set_xlim(1, 20)
# ax.set_xticks(np.arange(1,21))
# ax.fill_between(np.arange(1, 21), average_data_moving - std_dev_moving, average_data_moving + std_dev_moving,
#                        alpha=0.2, color='#ff7f0e')
# ax.set_ylabel('Time (s)', fontsize=14)
# ax.set_xlabel('Trial no.', fontsize=14)
# ax.legend()
# fig.savefig('figs/Non_replay_case_' + 'tau=' + tau_e + '_eta=' + eta + '.png')

########################################################################################################################
# Plotting the best cases for replay and non-replay
#replay

average_data_moving_replay, average_data_moving_nonreplay, std_dev_moving_replay, std_dev_moving_nonreplay \
	= generate_moving_average_best_cases()

fig, ax = plt.subplots(1,1)
#replay
tau_e = '0.04'
eta = '1'
ax.set_title('Replay versus non-replay')
ax.plot(np.arange(1, 31), average_data_moving_replay, label='With Replay')
ax.fill_between(np.arange(1, 31), average_data_moving_replay - std_dev_moving_replay,
                average_data_moving_replay + std_dev_moving_replay, alpha=0.4)

#nonreplay
tau_e = '1'
eta = '0.01'
ax.plot(np.arange(1, 31), average_data_moving_nonreplay, label='Without Replay')
ax.fill_between(np.arange(1, 31), average_data_moving_nonreplay - std_dev_moving_nonreplay,
                average_data_moving_nonreplay + std_dev_moving_nonreplay, alpha=0.2)

ax.set_ylim(0,60)
ax.set_xlim(1, 30)
ax.set_xticks(np.arange(1,31))
ax.set_ylabel('Time (s)', fontsize=14)
ax.set_xlabel('Trial no.', fontsize=14)
ax.legend()
fig.savefig('figs/replay_vs_nonreplay_best_cases.png')

# ########################################################################################################################
# # Plotting replay vs non-replay for tau_e=1s and eta=0.002
# fig, ax = plt.subplots(1,1)
# ax.set_title('Replay versus non-replay, $\\tau_e = 1s$; $\eta = 0.002$')
#
# ax.set_ylim(0,40)
# ax.set_xlim(1, 20)
# ax.set_xticks(np.arange(1,21))
#
# parameters = 'tau=1_eta=0.002'
# ax.plot(np.arange(1, 21), averages_replay[parameters], label='With Replay')
# ax.fill_between(np.arange(1, 21), np.array(averages_replay[parameters]) -
#                        np.array(std_dev_replay[parameters]),
#                        np.array(averages_replay[parameters]) + np.array(std_dev_replay[
# 	                                                                                   parameters]),
#                        alpha=0.4)
#
# ax.plot(np.arange(1, 21), averages_nonreplay[parameters], label='Without Replay')
# ax.fill_between(np.arange(1, 21), np.array(averages_nonreplay[parameters]) -
#                        np.array(std_dev_nonreplay[parameters]),
#                        np.array(averages_nonreplay[parameters]) + np.array(std_dev_nonreplay[
# 	                                                                                   parameters]),
#                        alpha=0.2)
#
# ax.legend()
# ax.set_ylabel('Time (s)', fontsize=14)
# ax.set_xlabel('Trial no.', fontsize=14)
# fig.savefig('figs/replay_vs_non_replay_'+ parameters + '.png')
#
# ########################################################################################################################
# # Plotting replay vs non-replay for tau_e=1s and eta=0.001
# fig, ax = plt.subplots(1,1)
# ax.set_title('Replay versus non-replay, $\\tau_e = 1s$; $\eta = 0.001$')
#
# ax.set_ylim(0,60)
# ax.set_xlim(1, 20)
# ax.set_xticks(np.arange(1,21))
#
# parameters = 'tau=1_eta=0.001'
# ax.plot(np.arange(1, 21), averages_replay[parameters], label='With Replay')
# ax.fill_between(np.arange(1, 21), np.array(averages_replay[parameters]) -
#                        np.array(std_dev_replay[parameters]),
#                        np.array(averages_replay[parameters]) + np.array(std_dev_replay[
# 	                                                                                   parameters]),
#                        alpha=0.4)
#
# ax.plot(np.arange(1, 21), averages_nonreplay[parameters], label='Without Replay')
# ax.fill_between(np.arange(1, 21), np.array(averages_nonreplay[parameters]) -
#                        np.array(std_dev_nonreplay[parameters]),
#                        np.array(averages_nonreplay[parameters]) + np.array(std_dev_nonreplay[
# 	                                                                                   parameters]),
#                        alpha=0.2)
#
# ax.legend()
# ax.set_ylabel('Time (s)', fontsize=14)
# ax.set_xlabel('Trial no.', fontsize=14)
# fig.savefig('figs/replay_vs_non_replay_'+ parameters + '.png')
#
# ########################################################################################################################
# # Plotting the weight population vectors before and after an experiment
# fig, ax = plt.subplots(1,2, figsize=(11.5,5))
# number_place_cells = 100
# number_action_cells = 72
#
# # used to generate a new colormap
# def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
# 	new_cmap = colors.LinearSegmentedColormap.from_list(
#         'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
#         cmap(np.linspace(minval, maxval, n)))
# 	return new_cmap
#
# # Before the experiment
# ax[0].set_title('Trial = 0')
# weight_vectors_x_components_arbitrary = np.zeros(number_place_cells)
# weight_vectors_y_components_arbitrary = np.zeros(number_place_cells)
# magnitudes = np.zeros(number_place_cells)
# weights_arbitrary = np.load('data/weight_at_start.npy', allow_pickle=True)
# if np.size(weights_arbitrary) == number_place_cells * number_action_cells:
# 	angles = np.radians(np.arange(0, 360, 5))
# 	for i in range(number_place_cells):
# 		for j in range(number_action_cells):
# 			weight_vectors_x_components_arbitrary[i] += weights_arbitrary[j, i] * np.cos(angles[j])
# 			weight_vectors_y_components_arbitrary[i] += weights_arbitrary[j, i] * np.sin(angles[j])
# 		# Compute magnitudes and normalise the vectors
# 		magnitudes[i] = np.sqrt(weight_vectors_x_components_arbitrary[i]**2 +
# 		                     weight_vectors_y_components_arbitrary[
# 			i]**2)
# 		weight_vectors_x_components_arbitrary[i] = 1.2 * weight_vectors_x_components_arbitrary[i] / magnitudes[i]
# 		weight_vectors_y_components_arbitrary[i] = 1.2 * weight_vectors_y_components_arbitrary[i] / magnitudes[i]
# x_coords = np.arange(10) / 5 - 0.9
# y_coords = np.flip(np.arange(10) / 5 - 0.9)
# x, y = np.meshgrid(x_coords, y_coords)
# cmap = plt.get_cmap('copper_r')
# new_cmap = truncate_colormap(cmap, 0.4, 1)
# im = ax[0].quiver(x, y, weight_vectors_x_components_arbitrary,
#                                                    weight_vectors_y_components_arbitrary, magnitudes, scale=20, clim=(0,
#                                                                                                                    3.5),
#                   width=0.008, cmap=new_cmap)
# ax[0].set_yticks([])
# ax[0].set_xticks([])
# ax[0].set_aspect('equal', 'box')
#
# # After the experiment
# ax[1].set_title('Trial = 20')
# weight_vectors_x_components_arbitrary = np.zeros(number_place_cells)
# weight_vectors_y_components_arbitrary = np.zeros(number_place_cells)
# magnitudes = np.zeros(number_place_cells)
# weights_arbitrary = np.load('data/weight_at_end.npy', allow_pickle=True)
# if np.size(weights_arbitrary) == number_place_cells * number_action_cells:
# 	angles = np.radians(np.arange(0, 360, 5))
# 	for i in range(number_place_cells):
# 		for j in range(number_action_cells):
# 			weight_vectors_x_components_arbitrary[i] += weights_arbitrary[j, i] * np.cos(angles[j])
# 			weight_vectors_y_components_arbitrary[i] += weights_arbitrary[j, i] * np.sin(angles[j])
# 		# Compute magnitudes and normalise the vectors
# 		magnitudes[i] = np.sqrt(weight_vectors_x_components_arbitrary[i]**2 +
# 		                     weight_vectors_y_components_arbitrary[
# 			i]**2)
# 		weight_vectors_x_components_arbitrary[i] = 1.2 * weight_vectors_x_components_arbitrary[i] / magnitudes[i]
# 		weight_vectors_y_components_arbitrary[i] = 1.2 * weight_vectors_y_components_arbitrary[i] / magnitudes[i]
# x_coords = np.arange(10) / 5 - 0.9
# y_coords = np.flip(np.arange(10) / 5 - 0.9)
# x, y = np.meshgrid(x_coords, y_coords)
# im = ax[1].quiver(x, y, weight_vectors_x_components_arbitrary,
#                                                    weight_vectors_y_components_arbitrary, magnitudes, scale=20,
#                   clim=(0, 3.5),
#                   width=0.008, cmap=new_cmap)
# ax[1].set_yticks([])
# ax[1].set_xticks([])
# ax[1].set_aspect('equal', 'box')
# cbar = fig.colorbar(im, ax=ax[1], label='Magnitude')
# fig.savefig('figs/non_replay_before_after_weights.png')
#
# ########################################################################################################################
# # Plotting all the cases for comparison between replay vs non-replay
#
# fig1, ax1 = plt.subplots(2, 2)
# time_const = 'tau=0.04_'
#
# ax1[0][0].plot(np.arange(1, 21), averages_replay[time_const + 'eta=0.01'])
# ax1[0][0].plot(np.arange(1, 21), averages_nonreplay[time_const + 'eta=0.01'])
# ax1[0][0].set_title('$\eta=0.01$')
# ax1[0][0].set_ylim(0,60)
# ax1[0][0].set_xlim(1,20)
# ax1[0][0].set_xticklabels([1, 5, 10, 15, 20])
# ax1[0][0].fill_between(np.arange(1, 21), np.array(averages_replay[time_const + 'eta=0.01']) -
#                        np.array(std_dev_replay[time_const + 'eta=0.01']),
#                        np.array(averages_replay[time_const + 'eta=0.01']) + np.array(std_dev_replay[time_const + 'eta=0.01']),
#                        alpha=0.4)
# ax1[0][0].fill_between(np.arange(1, 21), np.array(averages_nonreplay[time_const + 'eta=0.01']) -
#                        np.array(std_dev_nonreplay[time_const + 'eta=0.01']),
#                        np.array(averages_nonreplay[time_const + 'eta=0.01']) + np.array(std_dev_nonreplay[
# 	                                                                                   time_const + 'eta=0.01']),
#                        alpha=0.2)
#
# ax1[0][1].plot(np.arange(1, 21), averages_replay[time_const + 'eta=0.1'])
# ax1[0][1].plot(np.arange(1, 21), averages_nonreplay[time_const + 'eta=0.1'])
# ax1[0][1].set_title('$\eta=0.1$')
# ax1[0][1].set_ylim(0,60)
# ax1[0][1].set_xlim(1,20)
# ax1[0][1].set_xticklabels([1, 5, 10, 15, 20])
# ax1[0][1].fill_between(np.arange(1, 21), np.array(averages_replay[time_const + 'eta=0.1']) -
#                        np.array(std_dev_replay[time_const + 'eta=0.1']),
#                        np.array(averages_replay[time_const + 'eta=0.1']) + np.array(std_dev_replay[time_const + 'eta=0.1']),
#                        alpha=0.4)
# ax1[0][1].fill_between(np.arange(1, 21), np.array(averages_nonreplay[time_const + 'eta=0.1']) -
#                        np.array(std_dev_nonreplay[time_const + 'eta=0.1']),
#                        np.array(averages_nonreplay[time_const + 'eta=0.1']) + np.array(std_dev_nonreplay[
# 	                                                                                   time_const + 'eta=0.1']),
#                        alpha=0.2)
#
#
# ax1[1][0].plot(np.arange(1, 21), averages_replay[time_const + 'eta=1'])
# ax1[1][0].plot(np.arange(1, 21), averages_nonreplay[time_const + 'eta=1'])
# ax1[1][0].set_title('$\eta=1$')
# ax1[1][0].set_ylim(0,60)
# ax1[1][0].set_xlim(1,20)
# ax1[1][0].set_xticklabels([1, 5, 10, 15, 20])
# ax1[1][0].fill_between(np.arange(1, 21), np.array(averages_replay[time_const + 'eta=1']) -
#                        np.array(std_dev_replay[time_const + 'eta=1']),
#                        np.array(averages_replay[time_const + 'eta=1']) + np.array(std_dev_replay[time_const + 'eta=1']),
#                        alpha=0.4)
# ax1[1][0].fill_between(np.arange(1, 21), np.array(averages_nonreplay[time_const + 'eta=1']) -
#                        np.array(std_dev_nonreplay[time_const + 'eta=1']),
#                        np.array(averages_nonreplay[time_const + 'eta=1']) + np.array(std_dev_nonreplay[
# 	                                                                                   time_const + 'eta=1']),
#                        alpha=0.2)
#
#
# ax1[1][1].plot(np.arange(1, 21), averages_replay[time_const + 'eta=10'], label='With replay')
# ax1[1][1].plot(np.arange(1, 21), averages_nonreplay[time_const + 'eta=10'], label='Without replay')
# ax1[1][1].set_title('$\eta=10$')
# ax1[1][1].set_ylim(0,60)
# ax1[1][1].set_xlim(1,20)
# ax1[1][1].set_xticklabels([1, 5, 10, 15, 20])
# ax1[1][1].legend()
# ax1[1][1].fill_between(np.arange(1, 21), np.array(averages_replay[time_const + 'eta=10']) -
#                        np.array(std_dev_replay[time_const + 'eta=10']),
#                        np.array(averages_replay[time_const + 'eta=10']) + np.array(std_dev_replay[time_const + 'eta=10']),
#                        alpha=0.4)
# ax1[1][1].fill_between(np.arange(1, 21), np.array(averages_nonreplay[time_const + 'eta=10']) -
#                        np.array(std_dev_nonreplay[time_const + 'eta=10']),
#                        np.array(averages_nonreplay[time_const + 'eta=10']) + np.array(std_dev_nonreplay[
# 	                                                                                   time_const + 'eta=10']),
#                        alpha=0.2)
#
# fig1.tight_layout()
# fig1.savefig('figs/replay_vs_non_replay_' + time_const + '.png')
#
# ########################################################################################################################
#
# fig1, ax1 = plt.subplots(2, 2)
# time_const = 'tau=0.2_'
#
# ax1[0][0].plot(np.arange(1, 21), averages_replay[time_const + 'eta=0.01'])
# ax1[0][0].plot(np.arange(1, 21), averages_nonreplay[time_const + 'eta=0.01'])
# ax1[0][0].set_title('$\eta=0.01$')
# ax1[0][0].set_ylim(0,60)
# ax1[0][0].set_xlim(1,20)
# ax1[0][0].set_xticklabels([1, 5, 10, 15, 20])
# ax1[0][0].fill_between(np.arange(1, 21), np.array(averages_replay[time_const + 'eta=0.01']) -
#                        np.array(std_dev_replay[time_const + 'eta=0.01']),
#                        np.array(averages_replay[time_const + 'eta=0.01']) + np.array(std_dev_replay[time_const + 'eta=0.01']),
#                        alpha=0.4)
# ax1[0][0].fill_between(np.arange(1, 21), np.array(averages_nonreplay[time_const + 'eta=0.01']) -
#                        np.array(std_dev_nonreplay[time_const + 'eta=0.01']),
#                        np.array(averages_nonreplay[time_const + 'eta=0.01']) + np.array(std_dev_nonreplay[
# 	                                                                                   time_const + 'eta=0.01']),
#                        alpha=0.2)
#
# ax1[0][1].plot(np.arange(1, 21), averages_replay[time_const + 'eta=0.1'])
# ax1[0][1].plot(np.arange(1, 21), averages_nonreplay[time_const + 'eta=0.1'])
# ax1[0][1].set_title('$\eta=0.1$')
# ax1[0][1].set_ylim(0,60)
# ax1[0][1].set_xlim(1,20)
# ax1[0][1].set_xticklabels([1, 5, 10, 15, 20])
# ax1[0][1].fill_between(np.arange(1, 21), np.array(averages_replay[time_const + 'eta=0.1']) -
#                        np.array(std_dev_replay[time_const + 'eta=0.1']),
#                        np.array(averages_replay[time_const + 'eta=0.1']) + np.array(std_dev_replay[time_const + 'eta=0.1']),
#                        alpha=0.4)
# ax1[0][1].fill_between(np.arange(1, 21), np.array(averages_nonreplay[time_const + 'eta=0.1']) -
#                        np.array(std_dev_nonreplay[time_const + 'eta=0.1']),
#                        np.array(averages_nonreplay[time_const + 'eta=0.1']) + np.array(std_dev_nonreplay[
# 	                                                                                   time_const + 'eta=0.1']),
#                        alpha=0.2)
#
#
# ax1[1][0].plot(np.arange(1, 21), averages_replay[time_const + 'eta=1'])
# ax1[1][0].plot(np.arange(1, 21), averages_nonreplay[time_const + 'eta=1'])
# ax1[1][0].set_title('$\eta=1$')
# ax1[1][0].set_ylim(0,60)
# ax1[1][0].set_xlim(1,20)
# ax1[1][0].set_xticklabels([1, 5, 10, 15, 20])
# ax1[1][0].fill_between(np.arange(1, 21), np.array(averages_replay[time_const + 'eta=1']) -
#                        np.array(std_dev_replay[time_const + 'eta=1']),
#                        np.array(averages_replay[time_const + 'eta=1']) + np.array(std_dev_replay[time_const + 'eta=1']),
#                        alpha=0.4)
# ax1[1][0].fill_between(np.arange(1, 21), np.array(averages_nonreplay[time_const + 'eta=1']) -
#                        np.array(std_dev_nonreplay[time_const + 'eta=1']),
#                        np.array(averages_nonreplay[time_const + 'eta=1']) + np.array(std_dev_nonreplay[
# 	                                                                                   time_const + 'eta=1']),
#                        alpha=0.2)
#
#
# ax1[1][1].plot(np.arange(1, 21), averages_replay[time_const + 'eta=10'], label='With replay')
# ax1[1][1].plot(np.arange(1, 21), averages_nonreplay[time_const + 'eta=10'], label='Without replay')
# ax1[1][1].set_title('$\eta=10$')
# ax1[1][1].set_ylim(0,60)
# ax1[1][1].set_xlim(1,20)
# ax1[1][1].set_xticklabels([1, 5, 10, 15, 20])
# ax1[1][1].legend()
# ax1[1][1].fill_between(np.arange(1, 21), np.array(averages_replay[time_const + 'eta=10']) -
#                        np.array(std_dev_replay[time_const + 'eta=10']),
#                        np.array(averages_replay[time_const + 'eta=10']) + np.array(std_dev_replay[time_const + 'eta=10']),
#                        alpha=0.4)
# ax1[1][1].fill_between(np.arange(1, 21), np.array(averages_nonreplay[time_const + 'eta=10']) -
#                        np.array(std_dev_nonreplay[time_const + 'eta=10']),
#                        np.array(averages_nonreplay[time_const + 'eta=10']) + np.array(std_dev_nonreplay[
# 	                                                                                   time_const + 'eta=10']),
#                        alpha=0.2)
#
# fig1.tight_layout()
# fig1.savefig('figs/replay_vs_non_replay_' + time_const + '.png')

########################################################################################################################
#
# fig1, ax1 = plt.subplots(2, 2)
# time_const = 'tau=1_'
#
# ax1[0][0].plot(np.arange(1, 21), averages_replay[time_const + 'eta=0.005'])
# ax1[0][0].plot(np.arange(1, 21), averages_nonreplay[time_const + 'eta=0.005'])
# ax1[0][0].set_title('$\eta=0.005$')
# ax1[0][0].set_ylim(0,60)
# ax1[0][0].set_xlim(1,20)
# ax1[0][0].set_xticklabels([1, 5, 10, 15, 20])
# ax1[0][0].fill_between(np.arange(1, 21), np.array(averages_replay[time_const + 'eta=0.005']) -
#                        np.array(std_dev_replay[time_const + 'eta=0.005']),
#                        np.array(averages_replay[time_const + 'eta=0.005']) + np.array(std_dev_replay[time_const + 'eta=0.005']),
#                        alpha=0.4)
# ax1[0][0].fill_between(np.arange(1, 21), np.array(averages_nonreplay[time_const + 'eta=0.005']) -
#                        np.array(std_dev_nonreplay[time_const + 'eta=0.005']),
#                        np.array(averages_nonreplay[time_const + 'eta=0.005']) + np.array(std_dev_nonreplay[
# 	                                                                                   time_const + 'eta=0.005']),
#                        alpha=0.2)
#
# ax1[0][1].plot(np.arange(1, 21), averages_replay[time_const + 'eta=0.01'])
# ax1[0][1].plot(np.arange(1, 21), averages_nonreplay[time_const + 'eta=0.01'])
# ax1[0][1].set_title('$\eta=0.01$')
# ax1[0][1].set_ylim(0,60)
# ax1[0][1].set_xlim(1,20)
# ax1[0][1].set_xticklabels([1, 5, 10, 15, 20])
# ax1[0][1].fill_between(np.arange(1, 21), np.array(averages_replay[time_const + 'eta=0.01']) -
#                        np.array(std_dev_replay[time_const + 'eta=0.01']),
#                        np.array(averages_replay[time_const + 'eta=0.01']) + np.array(std_dev_replay[time_const + 'eta=0.01']),
#                        alpha=0.4)
# ax1[0][1].fill_between(np.arange(1, 21), np.array(averages_nonreplay[time_const + 'eta=0.01']) -
#                        np.array(std_dev_nonreplay[time_const + 'eta=0.01']),
#                        np.array(averages_nonreplay[time_const + 'eta=0.01']) + np.array(std_dev_nonreplay[
# 	                                                                                   time_const + 'eta=0.01']),
#                        alpha=0.2)
#
# ax1[1][0].plot(np.arange(1, 21), averages_replay[time_const + 'eta=0.1'])
# ax1[1][0].plot(np.arange(1, 21), averages_nonreplay[time_const + 'eta=0.1'])
# ax1[1][0].set_title('$\eta=0.1$')
# ax1[1][0].set_ylim(0,60)
# ax1[1][0].set_xlim(1,20)
# ax1[1][0].set_xticklabels([1, 5, 10, 15, 20])
# ax1[1][0].fill_between(np.arange(1, 21), np.array(averages_replay[time_const + 'eta=0.1']) -
#                        np.array(std_dev_replay[time_const + 'eta=0.1']),
#                        np.array(averages_replay[time_const + 'eta=0.1']) + np.array(std_dev_replay[time_const + 'eta=0.1']),
#                        alpha=0.4)
# ax1[1][0].fill_between(np.arange(1, 21), np.array(averages_nonreplay[time_const + 'eta=0.1']) -
#                        np.array(std_dev_nonreplay[time_const + 'eta=0.1']),
#                        np.array(averages_nonreplay[time_const + 'eta=0.1']) + np.array(std_dev_nonreplay[
# 	                                                                                   time_const + 'eta=0.1']),
#                        alpha=0.2)
#
# fig1.tight_layout()
# fig1.savefig('figs/replay_vs_non_replay_' + time_const + '.png')

#######################################################################
# Plotting the eligibility trace values for replay vs nonreplay for tau_e = 0.04s and eta = 0.1. Ignoring the
# variables having the name 'weights' in them, I'm just reusing code from above for the eligibility traces
#
# number_place_cells = 100
# number_action_cells = 72
#
# # used to generate a new colormap
# def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
# 	new_cmap = colors.LinearSegmentedColormap.from_list(
#         'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
#         cmap(np.linspace(minval, maxval, n)))
# 	return new_cmap
#
# # replay, t=0s
# fig, ax = plt.subplots(1)
# ax.set_title('Replay, t=0s')
# weight_vectors_x_components_arbitrary = np.zeros(number_place_cells)
# weight_vectors_y_components_arbitrary = np.zeros(number_place_cells)
# magnitudes = np.zeros(number_place_cells)
# weights_arbitrary = np.load('data/eligibility_trace_t_replay=0.01s.npy', allow_pickle=True)
# if np.size(weights_arbitrary) == number_place_cells * number_action_cells:
# 	angles = np.radians(np.arange(0, 360, 5))
# 	for i in range(number_place_cells):
# 		for j in range(number_action_cells):
# 			weight_vectors_x_components_arbitrary[i] += weights_arbitrary[j, i] * np.cos(angles[j])
# 			weight_vectors_y_components_arbitrary[i] += weights_arbitrary[j, i] * np.sin(angles[j])
# 		# Compute magnitudes and normalise the vectors
# 		magnitudes[i] = np.sqrt(weight_vectors_x_components_arbitrary[i]**2 +
# 		                     weight_vectors_y_components_arbitrary[
# 			i]**2)
# 		weight_vectors_x_components_arbitrary[i] = 1.2 * weight_vectors_x_components_arbitrary[i] / magnitudes[i]
# 		weight_vectors_y_components_arbitrary[i] = 1.2 * weight_vectors_y_components_arbitrary[i] / magnitudes[i]
# x_coords = np.arange(10) / 5 - 0.9
# y_coords = np.flip(np.arange(10) / 5 - 0.9)
# x, y = np.meshgrid(x_coords, y_coords)
# # cmap = plt.get_cmap('copper_r')
# # new_cmap = truncate_colormap(cmap, 0.4, 1)
# new_cmap = 'coolwarm'
# im = ax.quiver(x, y, weight_vectors_x_components_arbitrary,
#                                                    weight_vectors_y_components_arbitrary, magnitudes, scale=20,
#                clim=(-0.05, 0.05), width=0.008, cmap=new_cmap)
# ax.set_yticks([])
# ax.set_xticks([])
# ax.set_aspect('equal', 'box')
# fig.savefig('figs/eligibility_trace_replay_0s.png')
#
# # replay, t=1.2s
# fig, ax = plt.subplots(1)
# ax.set_title('Replay, t=1.2s')
# weight_vectors_x_components_arbitrary = np.zeros(number_place_cells)
# weight_vectors_y_components_arbitrary = np.zeros(number_place_cells)
# magnitudes = np.zeros(number_place_cells)
# weights_arbitrary = np.load('data/eligibility_trace_t_replay=1.2s.npy', allow_pickle=True)
# if np.size(weights_arbitrary) == number_place_cells * number_action_cells:
# 	angles = np.radians(np.arange(0, 360, 5))
# 	for i in range(number_place_cells):
# 		for j in range(number_action_cells):
# 			weight_vectors_x_components_arbitrary[i] += weights_arbitrary[j, i] * np.cos(angles[j])
# 			weight_vectors_y_components_arbitrary[i] += weights_arbitrary[j, i] * np.sin(angles[j])
# 		# Compute magnitudes and normalise the vectors
# 		magnitudes[i] = np.sqrt(weight_vectors_x_components_arbitrary[i]**2 +
# 		                     weight_vectors_y_components_arbitrary[
# 			i]**2)
# 		weight_vectors_x_components_arbitrary[i] = 1.2 * weight_vectors_x_components_arbitrary[i] / magnitudes[i]
# 		weight_vectors_y_components_arbitrary[i] = 1.2 * weight_vectors_y_components_arbitrary[i] / magnitudes[i]
# x_coords = np.arange(10) / 5 - 0.9
# y_coords = np.flip(np.arange(10) / 5 - 0.9)
# x, y = np.meshgrid(x_coords, y_coords)
# # cmap = plt.get_cmap('copper_r')
# # new_cmap = truncate_colormap(cmap, 0.4, 1)
# new_cmap = 'coolwarm'
# im = ax.quiver(x, y, weight_vectors_x_components_arbitrary,
#                                                    weight_vectors_y_components_arbitrary, magnitudes, scale=20,
#                clim=(-0.05, 0.05), width=0.008, cmap=new_cmap)
# ax.set_yticks([])
# ax.set_xticks([])
# ax.set_aspect('equal', 'box')
# fig.savefig('figs/eligibility_trace_replay_1.2s.png')
#
# # replay, t=1.4s
# fig, ax = plt.subplots(1)
# ax.set_title('Replay, t=1.4s')
# weight_vectors_x_components_arbitrary = np.zeros(number_place_cells)
# weight_vectors_y_components_arbitrary = np.zeros(number_place_cells)
# magnitudes = np.zeros(number_place_cells)
# weights_arbitrary = np.load('data/eligibility_trace_t_replay=1.4s.npy', allow_pickle=True)
# if np.size(weights_arbitrary) == number_place_cells * number_action_cells:
# 	angles = np.radians(np.arange(0, 360, 5))
# 	for i in range(number_place_cells):
# 		for j in range(number_action_cells):
# 			weight_vectors_x_components_arbitrary[i] += weights_arbitrary[j, i] * np.cos(angles[j])
# 			weight_vectors_y_components_arbitrary[i] += weights_arbitrary[j, i] * np.sin(angles[j])
# 		# Compute magnitudes and normalise the vectors
# 		magnitudes[i] = np.sqrt(weight_vectors_x_components_arbitrary[i]**2 +
# 		                     weight_vectors_y_components_arbitrary[
# 			i]**2)
# 		weight_vectors_x_components_arbitrary[i] = 1.2 * weight_vectors_x_components_arbitrary[i] / magnitudes[i]
# 		weight_vectors_y_components_arbitrary[i] = 1.2 * weight_vectors_y_components_arbitrary[i] / magnitudes[i]
# x_coords = np.arange(10) / 5 - 0.9
# y_coords = np.flip(np.arange(10) / 5 - 0.9)
# x, y = np.meshgrid(x_coords, y_coords)
# # cmap = plt.get_cmap('copper_r')
# # new_cmap = truncate_colormap(cmap, 0.4, 1)
# new_cmap = 'coolwarm'
# im = ax.quiver(x, y, weight_vectors_x_components_arbitrary,
#                                                    weight_vectors_y_components_arbitrary, magnitudes, scale=20,
#                clim=(-0.05, 0.05), width=0.008, cmap=new_cmap)
# ax.set_yticks([])
# ax.set_xticks([])
# ax.set_aspect('equal', 'box')
# fig.savefig('figs/eligibility_trace_replay_1.4s.png')
#
# # nonreplay, t=0s
# fig, ax = plt.subplots(1)
# ax.set_title('Non-replay, t=0s')
# weight_vectors_x_components_arbitrary = np.zeros(number_place_cells)
# weight_vectors_y_components_arbitrary = np.zeros(number_place_cells)
# magnitudes = np.zeros(number_place_cells)
# weights_arbitrary = np.load('data/eligibility_trace_t_nonreplay=0.01s.npy', allow_pickle=True)
# if np.size(weights_arbitrary) == number_place_cells * number_action_cells:
# 	angles = np.radians(np.arange(0, 360, 5))
# 	for i in range(number_place_cells):
# 		for j in range(number_action_cells):
# 			weight_vectors_x_components_arbitrary[i] += weights_arbitrary[j, i] * np.cos(angles[j])
# 			weight_vectors_y_components_arbitrary[i] += weights_arbitrary[j, i] * np.sin(angles[j])
# 		# Compute magnitudes and normalise the vectors
# 		magnitudes[i] = np.sqrt(weight_vectors_x_components_arbitrary[i]**2 +
# 		                     weight_vectors_y_components_arbitrary[
# 			i]**2)
# 		weight_vectors_x_components_arbitrary[i] = 1.2 * weight_vectors_x_components_arbitrary[i] / magnitudes[i]
# 		weight_vectors_y_components_arbitrary[i] = 1.2 * weight_vectors_y_components_arbitrary[i] / magnitudes[i]
# x_coords = np.arange(10) / 5 - 0.9
# y_coords = np.flip(np.arange(10) / 5 - 0.9)
# x, y = np.meshgrid(x_coords, y_coords)
# # cmap = plt.get_cmap('copper_r')
# # new_cmap = truncate_colormap(cmap, 0.4, 1)
# new_cmap = 'coolwarm'
# im = ax.quiver(x, y, weight_vectors_x_components_arbitrary,
#                                                    weight_vectors_y_components_arbitrary, magnitudes, scale=20,
#                clim=(-0.05, 0.05), width=0.008, cmap=new_cmap)
# ax.set_yticks([])
# ax.set_xticks([])
# ax.set_aspect('equal', 'box')
# fig.savefig('figs/eligibility_trace_nonreplay_0s.png')
#
# # nonreplay, t=1.2s
# fig, ax = plt.subplots(1)
# ax.set_title('Non-replay, t=1.2s')
# weight_vectors_x_components_arbitrary = np.zeros(number_place_cells)
# weight_vectors_y_components_arbitrary = np.zeros(number_place_cells)
# magnitudes = np.zeros(number_place_cells)
# weights_arbitrary = np.load('data/eligibility_trace_t_nonreplay=1.2s.npy', allow_pickle=True)
# if np.size(weights_arbitrary) == number_place_cells * number_action_cells:
# 	angles = np.radians(np.arange(0, 360, 5))
# 	for i in range(number_place_cells):
# 		for j in range(number_action_cells):
# 			weight_vectors_x_components_arbitrary[i] += weights_arbitrary[j, i] * np.cos(angles[j])
# 			weight_vectors_y_components_arbitrary[i] += weights_arbitrary[j, i] * np.sin(angles[j])
# 		# Compute magnitudes and normalise the vectors
# 		magnitudes[i] = np.sqrt(weight_vectors_x_components_arbitrary[i]**2 +
# 		                     weight_vectors_y_components_arbitrary[
# 			i]**2)
# 		weight_vectors_x_components_arbitrary[i] = 1.2 * weight_vectors_x_components_arbitrary[i] / magnitudes[i]
# 		weight_vectors_y_components_arbitrary[i] = 1.2 * weight_vectors_y_components_arbitrary[i] / magnitudes[i]
# x_coords = np.arange(10) / 5 - 0.9
# y_coords = np.flip(np.arange(10) / 5 - 0.9)
# x, y = np.meshgrid(x_coords, y_coords)
# # cmap = plt.get_cmap('copper_r')
# # new_cmap = truncate_colormap(cmap, 0.4, 1)
# new_cmap = 'coolwarm'
# im = ax.quiver(x, y, weight_vectors_x_components_arbitrary,
#                                                    weight_vectors_y_components_arbitrary, magnitudes, scale=20,
#                clim=(-0.05, 0.05), width=0.008, cmap=new_cmap)
# ax.set_yticks([])
# ax.set_xticks([])
# ax.set_aspect('equal', 'box')
# fig.savefig('figs/eligibility_trace_nonreplay_1.2s.png')
#
# # nonreplay, t=1.4s
# fig, ax = plt.subplots(1)
# ax.set_title('Non-replay, t=1.4s')
# weight_vectors_x_components_arbitrary = np.zeros(number_place_cells)
# weight_vectors_y_components_arbitrary = np.zeros(number_place_cells)
# magnitudes = np.zeros(number_place_cells)
# weights_arbitrary = np.load('data/eligibility_trace_t_nonreplay=1.4s.npy', allow_pickle=True)
# if np.size(weights_arbitrary) == number_place_cells * number_action_cells:
# 	angles = np.radians(np.arange(0, 360, 5))
# 	for i in range(number_place_cells):
# 		for j in range(number_action_cells):
# 			weight_vectors_x_components_arbitrary[i] += weights_arbitrary[j, i] * np.cos(angles[j])
# 			weight_vectors_y_components_arbitrary[i] += weights_arbitrary[j, i] * np.sin(angles[j])
# 		# Compute magnitudes and normalise the vectors
# 		magnitudes[i] = np.sqrt(weight_vectors_x_components_arbitrary[i]**2 +
# 		                     weight_vectors_y_components_arbitrary[
# 			i]**2)
# 		weight_vectors_x_components_arbitrary[i] = 1.2 * weight_vectors_x_components_arbitrary[i] / magnitudes[i]
# 		weight_vectors_y_components_arbitrary[i] = 1.2 * weight_vectors_y_components_arbitrary[i] / magnitudes[i]
# x_coords = np.arange(10) / 5 - 0.9
# y_coords = np.flip(np.arange(10) / 5 - 0.9)
# x, y = np.meshgrid(x_coords, y_coords)
# # cmap = plt.get_cmap('copper_r')
# # new_cmap = truncate_colormap(cmap, 0.4, 1)
# new_cmap = 'coolwarm'
# im = ax.quiver(x, y, weight_vectors_x_components_arbitrary,
#                                                    weight_vectors_y_components_arbitrary, magnitudes, scale=20,
#                clim=(-0.05, 0.05), width=0.008, cmap=new_cmap)
# ax.set_yticks([])
# ax.set_xticks([])
# ax.set_aspect('equal', 'box')
# fig.savefig('figs/eligibility_trace_nonreplay_1.4s.png')
#############################################################

# ########################################################################################################################
# # Plotting the results of the modified learning rule
# tau_e = '5'
# eta = '0.001'
# fig, ax = plt.subplots(1,1)
# # ax.set_title('Modified learning rule, ' + '$\\tau_e =$ ' + tau_e + '$, \eta =$ ' + eta)
# average_data_moving, std_dev_moving = generate_moving_average_data(tau_e, eta, modified=True)
# print(average_data_moving)
# ax.plot(np.arange(1, 21), average_data_moving, color='green', label='$\\tau_e = 5s$ \n $\eta = '
#                                                                              '0.001$')
# ax.fill_between(np.arange(1, 21), average_data_moving - std_dev_moving, average_data_moving + std_dev_moving,
#                        alpha=0.2, color='green')
#
# ax.set_ylim(0,60)
# ax.set_xlim(1, 20)
# ax.set_xticks(np.arange(1,21))
# ax.set_ylabel('Time (s)', fontsize=14)
# ax.set_xlabel('Trial no.', fontsize=14)
# # ax.legend()
# fig.savefig('figs/Modified_learning_rule_' + 'tau=' + tau_e + '_eta=' + eta + '.png')

########################################################################################################################

plt.show()