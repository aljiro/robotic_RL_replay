#!/usr/bin/python3

'''
Comparisons for all parameters with and without reverse replays.
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as colors
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D
import csv
import copy

# Storing the raw data
result_parameters = {'tau=0.04_eta=0.01': [],
                          'tau=0.04_eta=0.1': [],
                          'tau=0.04_eta=1': [],
                          'tau=0.04_eta=10': [],
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
                          'tau=5_eta=0.01': [],
                          'tau=5_eta=0.1': [],}

results_full_nonreplay = copy.deepcopy(result_parameters)
results_full_replay = copy.deepcopy(result_parameters)

with open('data/trial_times/trial_times_NON_REPLAY_FULL.csv', newline='') as file:
	data = csv.reader(file, delimiter=',')
	for i, row in enumerate(data):
		if (2 <= i < 42):
			results_full_nonreplay['tau=0.04_eta=0.01'].append([float(j) for j in row[1:]])
		if (44 <= i < 84):
			results_full_nonreplay['tau=0.04_eta=0.1'].append([float(j) for j in row[1:]])
		if (86 <= i < 126):
			results_full_nonreplay['tau=0.04_eta=1'].append([float(j) for j in row[1:]])
		if (128 <= i < 168):
			results_full_nonreplay['tau=0.04_eta=10'].append([float(j) for j in row[1:]])
		if (170 <= i < 210):
			results_full_nonreplay['tau=0.2_eta=0.01'].append([float(j) for j in row[1:]])
		if (212 <= i < 252):
			results_full_nonreplay['tau=0.2_eta=0.1'].append([float(j) for j in row[1:]])
		if (254 <= i < 294):
			results_full_nonreplay['tau=0.2_eta=1'].append([float(j) for j in row[1:]])
		if (296 <= i < 336):
			results_full_nonreplay['tau=0.2_eta=10'].append([float(j) for j in row[1:]])
		if (338 <= i < 358):
			results_full_nonreplay['tau=1_eta=0.01'].append([float(j) for j in row[1:]])
		if (360 <= i < 380):
			results_full_nonreplay['tau=1_eta=0.1'].append([float(j) for j in row[1:]])
		if (382 <= i < 402):
			results_full_nonreplay['tau=1_eta=1'].append([float(j) for j in row[1:]])
		if (404 <= i < 424):
			results_full_nonreplay['tau=5_eta=0.01'].append([float(j) for j in row[1:]])
		if (426 <= i < 446):
			results_full_nonreplay['tau=5_eta=0.1'].append([float(j) for j in row[1:]])
		if (448 <= i < 468):
			results_full_nonreplay['tau=1_eta=0.005'].append([float(j) for j in row[1:]])
		if (470 <= i < 490):
			results_full_nonreplay['tau=1_eta=0.002'].append([float(j) for j in row[1:]])
		if (492 <= i < 512):
			results_full_nonreplay['tau=1_eta=0.001'].append([float(j) for j in row[1:]])

with open('data/trial_times/trial_times_WITH_REPLAY_FULL.csv', newline='') as file:
	data = csv.reader(file, delimiter=',')
	for i, row in enumerate(data):
		if (2 <= i < 42):
			results_full_replay['tau=0.04_eta=0.01'].append([float(j) for j in row[1:]])
		if (44 <= i < 84):
			results_full_replay['tau=0.04_eta=0.1'].append([float(j) for j in row[1:]])
		if (86 <= i < 126):
			results_full_replay['tau=0.04_eta=1'].append([float(j) for j in row[1:]])
		if (128 <= i < 168):
			results_full_replay['tau=0.04_eta=10'].append([float(j) for j in row[1:]])
		if (170 <= i < 210):
			results_full_replay['tau=0.2_eta=0.01'].append([float(j) for j in row[1:]])
		if (212 <= i < 252):
			results_full_replay['tau=0.2_eta=0.1'].append([float(j) for j in row[1:]])
		if (254 <= i < 294):
			results_full_replay['tau=0.2_eta=1'].append([float(j) for j in row[1:]])
		if (296 <= i < 336):
			results_full_replay['tau=0.2_eta=10'].append([float(j) for j in row[1:]])
		if (338 <= i < 358):
			results_full_replay['tau=1_eta=0.01'].append([float(j) for j in row[1:]])
		if (360 <= i < 380):
			results_full_replay['tau=1_eta=0.1'].append([float(j) for j in row[1:]])
		if (382 <= i < 402):
			results_full_replay['tau=1_eta=1'].append([float(j) for j in row[1:]])
		if (404 <= i < 424):
			results_full_replay['tau=5_eta=0.01'].append([float(j) for j in row[1:]])
		if (426 <= i < 446):
			results_full_replay['tau=5_eta=0.1'].append([float(j) for j in row[1:]])
		if (448 <= i < 468):
			results_full_replay['tau=1_eta=0.005'].append([float(j) for j in row[1:]])
		if (470 <= i < 490):
			results_full_replay['tau=1_eta=0.002'].append([float(j) for j in row[1:]])
		if (492 <= i < 512):
			results_full_replay['tau=1_eta=0.001'].append([float(j) for j in row[1:]])

# Getting the averages and standard deviations for each
averages_nonreplay = copy.deepcopy(result_parameters)

averages_replay = copy.deepcopy(result_parameters)

std_dev_nonreplay = copy.deepcopy(result_parameters)

std_dev_replay = copy.deepcopy(result_parameters)

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
		for experiment_no in range(len(results_full_nonreplay[key])):
			average += results_full_replay[key][experiment_no][trial_no]
		average /= no_experiments
		averages_replay[key].append(average)

		standard_dev = 0
		for experiment_no in range(len(results_full_nonreplay[key])):
			standard_dev += (results_full_replay[key][experiment_no][trial_no] - averages_replay[key][trial_no])**2
		standard_dev = np.sqrt(standard_dev / no_experiments)
		std_dev_replay[key].append(standard_dev)

################################################################################

def generate_moving_average_data(tau_e, eta):
	time_const = 'tau=' + tau_e + '_'
	eta = 'eta=' + eta
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

def produce_plots(tau_e, list_eta, plt_show=False):
	'''
	produces the comparison plots
	:param tau_e: string
	:param eta: list of strings, for now only 4 allowed
	:param plt_show: bool
	:return: none
	'''

	fig1, ax1 = plt.subplots(2, 2)

	eta = list_eta[0]
	moving_average_data = generate_moving_average_data(tau_e, eta)
	moving_average_replay = moving_average_data[0]
	moving_average_nonreplay = moving_average_data[1]
	moving_average_std_devs_replay = moving_average_data[2]
	moving_average_std_devs_nonreplay = moving_average_data[3]

	# plot averages
	ax1[0][0].plot(np.arange(1, 21), moving_average_replay)
	ax1[0][0].plot(np.arange(1, 21), moving_average_nonreplay)
	ax1[0][0].set_title('$\\tau_e = $' + tau_e + ', $\eta = $' + eta)
	ax1[0][0].set_ylim(0, 60)
	ax1[0][0].set_xlim(1, 20)
	ax1[0][0].set_xticklabels([1, 5, 10, 15, 20])

	# plot standard deviations
	ax1[0][0].fill_between(np.arange(1, 21), moving_average_replay - moving_average_std_devs_replay,
	                       moving_average_replay + moving_average_std_devs_replay,
	                       alpha=0.4)
	ax1[0][0].fill_between(np.arange(1, 21), moving_average_nonreplay - moving_average_std_devs_nonreplay,
	                       moving_average_nonreplay + moving_average_std_devs_nonreplay,
	                       alpha=0.2)

	eta = list_eta[1]
	moving_average_data = generate_moving_average_data(tau_e, eta)
	moving_average_replay = moving_average_data[0]
	moving_average_nonreplay = moving_average_data[1]
	moving_average_std_devs_replay = moving_average_data[2]
	moving_average_std_devs_nonreplay = moving_average_data[3]

	# plot averages
	ax1[0][1].plot(np.arange(1, 21), moving_average_replay)
	ax1[0][1].plot(np.arange(1, 21), moving_average_nonreplay)
	ax1[0][1].set_title('$\\tau_e = $' + tau_e + ', $\eta = $' + eta)
	ax1[0][1].set_ylim(0, 60)
	ax1[0][1].set_xlim(1, 20)
	ax1[0][1].set_xticklabels([1, 5, 10, 15, 20])

	# plot standard deviations
	ax1[0][1].fill_between(np.arange(1, 21), moving_average_replay - moving_average_std_devs_replay,
	                       moving_average_replay + moving_average_std_devs_replay,
	                       alpha=0.4)
	ax1[0][1].fill_between(np.arange(1, 21), moving_average_nonreplay - moving_average_std_devs_nonreplay,
	                       moving_average_nonreplay + moving_average_std_devs_nonreplay,
	                       alpha=0.2)

	eta = list_eta[2]
	moving_average_data = generate_moving_average_data(tau_e, eta)
	moving_average_replay = moving_average_data[0]
	moving_average_nonreplay = moving_average_data[1]
	moving_average_std_devs_replay = moving_average_data[2]
	moving_average_std_devs_nonreplay = moving_average_data[3]

	# plot averages
	ax1[1][0].plot(np.arange(1, 21), moving_average_replay)
	ax1[1][0].plot(np.arange(1, 21), moving_average_nonreplay)
	ax1[1][0].set_title('$\\tau_e = $' + tau_e + ', $\eta = $' + eta)
	ax1[1][0].set_ylim(0, 60)
	ax1[1][0].set_xlim(1, 20)
	ax1[1][0].set_xticklabels([1, 5, 10, 15, 20])

	# plot standard deviations
	ax1[1][0].fill_between(np.arange(1, 21), moving_average_replay - moving_average_std_devs_replay,
	                       moving_average_replay + moving_average_std_devs_replay,
	                       alpha=0.4)
	ax1[1][0].fill_between(np.arange(1, 21), moving_average_nonreplay - moving_average_std_devs_nonreplay,
	                       moving_average_nonreplay + moving_average_std_devs_nonreplay,
	                       alpha=0.2)

	eta = list_eta[3]
	moving_average_data = generate_moving_average_data(tau_e, eta)
	moving_average_replay = moving_average_data[0]
	moving_average_nonreplay = moving_average_data[1]
	moving_average_std_devs_replay = moving_average_data[2]
	moving_average_std_devs_nonreplay = moving_average_data[3]

	# plot averages
	ax1[1][1].plot(np.arange(1, 21), moving_average_replay)
	ax1[1][1].plot(np.arange(1, 21), moving_average_nonreplay)
	ax1[1][1].set_title('$\\tau_e = $' + tau_e + ', $\eta = $' + eta)
	ax1[1][1].set_ylim(0, 60)
	ax1[1][1].set_xlim(1, 20)
	ax1[1][1].set_xticklabels([1, 5, 10, 15, 20])

	# plot standard deviations
	ax1[1][1].fill_between(np.arange(1, 21), moving_average_replay - moving_average_std_devs_replay,
	                       moving_average_replay + moving_average_std_devs_replay,
	                       alpha=0.4)
	ax1[1][1].fill_between(np.arange(1, 21), moving_average_nonreplay - moving_average_std_devs_nonreplay,
	                       moving_average_nonreplay + moving_average_std_devs_nonreplay,
	                       alpha=0.2)

	fig1.tight_layout()
	fig1.savefig('figs/replay_vs_non_replay_tau_e=' + tau_e + '.png')
	if plt_show == True:
		plt.show()

###############################################################################
# Plots for tau_e = 0.04s
tau_e = '0.04'
list_eta = ['0.01', '0.1', '1', '10']
produce_plots(tau_e, list_eta, plt_show=True)

###############################################################################
# Plots for tau_e = 0.2s
tau_e = '0.2'
list_eta = ['0.01', '0.1', '1', '10']
produce_plots(tau_e, list_eta, plt_show=True)


###############################################################################
# Producing the box charts
# setup the figure and axes
fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')
eta = np.array([0.001, 0.01, 0.1, 1, 10])
tau_e = np.array([0.04, 0.2, 1, 5])

# get the bar chart positions
width = 0.4
start = 0.1
_x = np.array([start, start + width,
               1 + start, 1 + start + width,
               2 + start, 2 + start + width,
               3 + start, 3 + start + width,
               4 + start, 4 + start + width
               ])
_y = np.array([0.1, 0.6, 1.1, 1.6])
_xx, _yy = np.meshgrid(_x, _y)
x, y = _xx.ravel(), _yy.ravel()

replay_data = np.ones(20)
nonreplay_data = 2 * np.ones(20)

# putting the data together
full_data = np.zeros(40)
for i in range(40):
	if i % 2 == 0:
		full_data[i] = replay_data[int(i / 2)]
	else:
		full_data[i] = replay_data[int(i / 2)]


bottom = np.zeros(len(eta) * len(tau_e) * 2)
width = depth = 0.5
top = np.ones(len(eta) * len(tau_e) * 2)



ax1.get_proj = lambda: np.dot(Axes3D.get_proj(ax1), np.diag([1, 0.5, 1, 1]))
colors = ['blue', 'red'] * 20
ax1.bar3d(x, y, bottom, width, depth, full_data, shade=True, color=colors)

x = x + width


ax1.set_xticks(np.arange(len(eta)) + 0.5)
ax1.set_yticks(_y + width / 2)
ax1.set_xticklabels(eta)
ax1.set_yticklabels(tau_e)

ax1.set_xlabel('$\eta$')
ax1.set_ylabel('$\\tau_e$')
ax1.set_zlabel('Time (s)')

plt.show()
