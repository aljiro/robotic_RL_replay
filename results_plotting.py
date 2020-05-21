#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import csv

# Storing the raw data
results_full_nonreplay = {'tau=0.04_eta=0.01': [],
                          'tau=0.04_eta=0.1': [],
                          'tau=0.04_eta=1': [],
                          'tau=0.04_eta=10': [],
                          'tau=0.2_eta=0.01': [],
                          'tau=0.2_eta=0.1': [],
                          'tau=0.2_eta=1': [],
                          'tau=0.2_eta=10': [],
                          'tau=1_eta=0.01': [],
                          'tau=1_eta=0.1': [],
                          'tau=1_eta=1': [],
                          'tau=5_eta=0.01': [],
                          'tau=5_eta=0.1': [],}

results_full_replay = {'tau=0.04_eta=0.01': [],
                          'tau=0.04_eta=0.1': [],
                          'tau=0.04_eta=1': [],
                          'tau=0.04_eta=10': [],
                          'tau=0.2_eta=0.01': [],
                          'tau=0.2_eta=0.1': [],
                          'tau=0.2_eta=1': [],
                          'tau=0.2_eta=10': [],
                          'tau=1_eta=0.01': [],
                          'tau=1_eta=0.1': [],
                          'tau=1_eta=1': [],
                          'tau=5_eta=0.01': [],
                          'tau=5_eta=0.1': [],}

with open('data/trial_times/trial_times_NON_REPLAY_FULL.csv', newline='') as file:
	data = csv.reader(file, delimiter=',')
	for i, row in enumerate(data):
		if (2 <= i < 22):
			results_full_nonreplay['tau=0.04_eta=0.01'].append([float(j) for j in row[1:]])
		if (24 <= i < 44):
			results_full_nonreplay['tau=0.04_eta=0.1'].append([float(j) for j in row[1:]])
		if (46 <= i < 66):
			results_full_nonreplay['tau=0.04_eta=1'].append([float(j) for j in row[1:]])
		if (68 <= i < 88):
			results_full_nonreplay['tau=0.04_eta=10'].append([float(j) for j in row[1:]])
		if (90 <= i < 110):
			results_full_nonreplay['tau=0.2_eta=0.01'].append([float(j) for j in row[1:]])
		if (112 <= i < 132):
			results_full_nonreplay['tau=0.2_eta=0.1'].append([float(j) for j in row[1:]])
		if (134 <= i < 154):
			results_full_nonreplay['tau=0.2_eta=1'].append([float(j) for j in row[1:]])
		if (156 <= i < 176):
			results_full_nonreplay['tau=0.2_eta=10'].append([float(j) for j in row[1:]])
		if (178 <= i < 198):
			results_full_nonreplay['tau=1_eta=0.01'].append([float(j) for j in row[1:]])
		if (200 <= i < 220):
			results_full_nonreplay['tau=1_eta=0.1'].append([float(j) for j in row[1:]])
		if (222 <= i < 242):
			results_full_nonreplay['tau=1_eta=1'].append([float(j) for j in row[1:]])
		if (244 <= i < 264):
			results_full_nonreplay['tau=5_eta=0.01'].append([float(j) for j in row[1:]])
		if (266 <= i < 286):
			results_full_nonreplay['tau=5_eta=0.1'].append([float(j) for j in row[1:]])

with open('data/trial_times/trial_times_WITH_REPLAY_FULL.csv', newline='') as file:
	data = csv.reader(file, delimiter=',')
	for i, row in enumerate(data):
		if (2 <= i < 22):
			results_full_replay['tau=0.04_eta=0.01'].append([float(j) for j in row[1:]])
		if (24 <= i < 44):
			results_full_replay['tau=0.04_eta=0.1'].append([float(j) for j in row[1:]])
		if (46 <= i < 66):
			results_full_replay['tau=0.04_eta=1'].append([float(j) for j in row[1:]])
		if (68 <= i < 88):
			results_full_replay['tau=0.04_eta=10'].append([float(j) for j in row[1:]])
		if (90 <= i < 110):
			results_full_replay['tau=0.2_eta=0.01'].append([float(j) for j in row[1:]])
		if (112 <= i < 132):
			results_full_replay['tau=0.2_eta=0.1'].append([float(j) for j in row[1:]])
		if (134 <= i < 154):
			results_full_replay['tau=0.2_eta=1'].append([float(j) for j in row[1:]])
		if (156 <= i < 176):
			results_full_replay['tau=0.2_eta=10'].append([float(j) for j in row[1:]])
		if (178 <= i < 198):
			results_full_replay['tau=1_eta=0.01'].append([float(j) for j in row[1:]])
		if (200 <= i < 220):
			results_full_replay['tau=1_eta=0.1'].append([float(j) for j in row[1:]])
		if (222 <= i < 242):
			results_full_replay['tau=1_eta=1'].append([float(j) for j in row[1:]])
		if (244 <= i < 264):
			results_full_replay['tau=5_eta=0.01'].append([float(j) for j in row[1:]])
		if (266 <= i < 286):
			results_full_replay['tau=5_eta=0.1'].append([float(j) for j in row[1:]])

# Getting the averages and standard deviations for each
averages_nonreplay = {'tau=0.04_eta=0.01': [],
                          'tau=0.04_eta=0.1': [],
                          'tau=0.04_eta=1': [],
                          'tau=0.04_eta=10': [],
                          'tau=0.2_eta=0.01': [],
                          'tau=0.2_eta=0.1': [],
                          'tau=0.2_eta=1': [],
                          'tau=0.2_eta=10': [],
                          'tau=1_eta=0.01': [],
                          'tau=1_eta=0.1': [],
                          'tau=1_eta=1': [],
                          'tau=5_eta=0.01': [],
                          'tau=5_eta=0.1': [],}

averages_replay = {'tau=0.04_eta=0.01': [],
                          'tau=0.04_eta=0.1': [],
                          'tau=0.04_eta=1': [],
                          'tau=0.04_eta=10': [],
                          'tau=0.2_eta=0.01': [],
                          'tau=0.2_eta=0.1': [],
                          'tau=0.2_eta=1': [],
                          'tau=0.2_eta=10': [],
                          'tau=1_eta=0.01': [],
                          'tau=1_eta=0.1': [],
                          'tau=1_eta=1': [],
                          'tau=5_eta=0.01': [],
                          'tau=5_eta=0.1': [],}

std_dev_nonreplay = {'tau=0.04_eta=0.01': [],
                          'tau=0.04_eta=0.1': [],
                          'tau=0.04_eta=1': [],
                          'tau=0.04_eta=10': [],
                          'tau=0.2_eta=0.01': [],
                          'tau=0.2_eta=0.1': [],
                          'tau=0.2_eta=1': [],
                          'tau=0.2_eta=10': [],
                          'tau=1_eta=0.01': [],
                          'tau=1_eta=0.1': [],
                          'tau=1_eta=1': [],
                          'tau=5_eta=0.01': [],
                          'tau=5_eta=0.1': [],}

std_dev_replay = {'tau=0.04_eta=0.01': [],
                          'tau=0.04_eta=0.1': [],
                          'tau=0.04_eta=1': [],
                          'tau=0.04_eta=10': [],
                          'tau=0.2_eta=0.01': [],
                          'tau=0.2_eta=0.1': [],
                          'tau=0.2_eta=1': [],
                          'tau=0.2_eta=10': [],
                          'tau=1_eta=0.01': [],
                          'tau=1_eta=0.1': [],
                          'tau=1_eta=1': [],
                          'tau=5_eta=0.01': [],
                          'tau=5_eta=0.1': [],}

for key in averages_nonreplay:
	for trial_no in range(20):
		average = 0
		for experiment_no in range(20):
			average += results_full_nonreplay[key][experiment_no][trial_no]
		average /= 20.0
		averages_nonreplay[key].append(average)

		standard_dev = 0
		for experiment_no in range(20):
			standard_dev += (results_full_nonreplay[key][experiment_no][trial_no] - averages_nonreplay[key][trial_no])**2
		standard_dev = np.sqrt(standard_dev / 20.0)
		std_dev_nonreplay[key].append(standard_dev)

for key in averages_replay:
	for trial_no in range(20):
		average = 0
		for experiment_no in range(20):
			average += results_full_replay[key][experiment_no][trial_no]
		average /= 20.0
		averages_replay[key].append(average)

		standard_dev = 0
		for experiment_no in range(20):
			standard_dev += (results_full_replay[key][experiment_no][trial_no] - averages_replay[key][trial_no])**2
		standard_dev = np.sqrt(standard_dev / 20.0)
		std_dev_replay[key].append(standard_dev)

######################################################

fig1, ax1 = plt.subplots(2, 2)

ax1[0][0].plot(averages_replay['tau=0.04_eta=0.01'])
ax1[0][0].plot(averages_nonreplay['tau=0.04_eta=0.01'])
ax1[0][0].set_title('$\\tau_e = 0.04s$; $\eta = 0.01$')
ax1[0][0].set_ylim(0,60)
ax1[0][0].fill_between(np.arange(20), np.array(averages_replay['tau=0.04_eta=0.01']) -
                       np.array(std_dev_replay['tau=0.04_eta=0.01']),
                       np.array(averages_replay['tau=0.04_eta=0.01']) + np.array(std_dev_replay['tau=0.04_eta=0.01']),
                       alpha=0.4)
ax1[0][0].fill_between(np.arange(20), np.array(averages_nonreplay['tau=0.04_eta=0.01']) -
                       np.array(std_dev_nonreplay['tau=0.04_eta=0.01']),
                       np.array(averages_nonreplay['tau=0.04_eta=0.01']) + np.array(std_dev_nonreplay[
	                                                                                   'tau=0.04_eta=0.01']),
                       alpha=0.2)

ax1[0][1].plot(averages_replay['tau=0.04_eta=0.1'])
ax1[0][1].plot(averages_nonreplay['tau=0.04_eta=0.1'])
ax1[0][1].set_title('$\\tau_e = 0.04s$; $\eta = 0.1$')
ax1[0][1].set_ylim(0,60)
ax1[0][1].fill_between(np.arange(20), np.array(averages_replay['tau=0.04_eta=0.1']) -
                       np.array(std_dev_replay['tau=0.04_eta=0.1']),
                       np.array(averages_replay['tau=0.04_eta=0.1']) + np.array(std_dev_replay['tau=0.04_eta=0.1']),
                       alpha=0.4)
ax1[0][1].fill_between(np.arange(20), np.array(averages_nonreplay['tau=0.04_eta=0.1']) -
                       np.array(std_dev_nonreplay['tau=0.04_eta=0.1']),
                       np.array(averages_nonreplay['tau=0.04_eta=0.1']) + np.array(std_dev_nonreplay[
	                                                                                   'tau=0.04_eta=0.1']),
                       alpha=0.2)


ax1[1][0].plot(averages_replay['tau=0.04_eta=1'])
ax1[1][0].plot(averages_nonreplay['tau=0.04_eta=1'])
ax1[1][0].set_title('$\\tau_e = 0.04s$; $\eta = 1$')
ax1[1][0].set_ylim(0,60)
ax1[1][0].fill_between(np.arange(20), np.array(averages_replay['tau=0.04_eta=1']) -
                       np.array(std_dev_replay['tau=0.04_eta=1']),
                       np.array(averages_replay['tau=0.04_eta=1']) + np.array(std_dev_replay['tau=0.04_eta=1']),
                       alpha=0.4)
ax1[1][0].fill_between(np.arange(20), np.array(averages_nonreplay['tau=0.04_eta=1']) -
                       np.array(std_dev_nonreplay['tau=0.04_eta=1']),
                       np.array(averages_nonreplay['tau=0.04_eta=1']) + np.array(std_dev_nonreplay[
	                                                                                   'tau=0.04_eta=1']),
                       alpha=0.2)


ax1[1][1].plot(averages_replay['tau=0.04_eta=10'], label='With replay')
ax1[1][1].plot(averages_nonreplay['tau=0.04_eta=10'], label='Without replay')
ax1[1][1].set_title('$\\tau_e = 0.04s$; $\eta = 10$')
ax1[1][1].set_ylim(0,60)
ax1[1][1].legend()
ax1[1][1].fill_between(np.arange(20), np.array(averages_replay['tau=0.04_eta=10']) -
                       np.array(std_dev_replay['tau=0.04_eta=10']),
                       np.array(averages_replay['tau=0.04_eta=10']) + np.array(std_dev_replay['tau=0.04_eta=10']),
                       alpha=0.4)
ax1[1][1].fill_between(np.arange(20), np.array(averages_nonreplay['tau=0.04_eta=10']) -
                       np.array(std_dev_nonreplay['tau=0.04_eta=10']),
                       np.array(averages_nonreplay['tau=0.04_eta=10']) + np.array(std_dev_nonreplay[
	                                                                                   'tau=0.04_eta=10']),
                       alpha=0.2)

############################################################

fig1, ax1 = plt.subplots(2, 2)

ax1[0][0].plot(averages_replay['tau=0.2_eta=0.01'])
ax1[0][0].plot(averages_nonreplay['tau=0.2_eta=0.01'])
ax1[0][0].set_title('$\\tau_e = 0.2s$; $\eta = 0.01$')
ax1[0][0].set_ylim(0,60)
ax1[0][0].fill_between(np.arange(20), np.array(averages_replay['tau=0.2_eta=0.01']) -
                       np.array(std_dev_replay['tau=0.2_eta=0.01']),
                       np.array(averages_replay['tau=0.2_eta=0.01']) + np.array(std_dev_replay['tau=0.2_eta=0.01']),
                       alpha=0.4)
ax1[0][0].fill_between(np.arange(20), np.array(averages_nonreplay['tau=0.2_eta=0.01']) -
                       np.array(std_dev_nonreplay['tau=0.2_eta=0.01']),
                       np.array(averages_nonreplay['tau=0.2_eta=0.01']) + np.array(std_dev_nonreplay[
	                                                                                   'tau=0.2_eta=0.01']),
                       alpha=0.2)

ax1[0][1].plot(averages_replay['tau=0.2_eta=0.1'])
ax1[0][1].plot(averages_nonreplay['tau=0.2_eta=0.1'])
ax1[0][1].set_title('$\\tau_e = 0.2s$; $\eta = 0.1$')
ax1[0][1].set_ylim(0,60)
ax1[0][1].fill_between(np.arange(20), np.array(averages_replay['tau=0.2_eta=0.1']) -
                       np.array(std_dev_replay['tau=0.2_eta=0.1']),
                       np.array(averages_replay['tau=0.2_eta=0.1']) + np.array(std_dev_replay['tau=0.2_eta=0.1']),
                       alpha=0.4)
ax1[0][1].fill_between(np.arange(20), np.array(averages_nonreplay['tau=0.2_eta=0.1']) -
                       np.array(std_dev_nonreplay['tau=0.2_eta=0.1']),
                       np.array(averages_nonreplay['tau=0.2_eta=0.1']) + np.array(std_dev_nonreplay[
	                                                                                   'tau=0.2_eta=0.1']),
                       alpha=0.2)


ax1[1][0].plot(averages_replay['tau=0.2_eta=1'])
ax1[1][0].plot(averages_nonreplay['tau=0.2_eta=1'])
ax1[1][0].set_title('$\\tau_e = 0.2s$; $\eta = 1$')
ax1[1][0].set_ylim(0,60)
ax1[1][0].fill_between(np.arange(20), np.array(averages_replay['tau=0.2_eta=1']) -
                       np.array(std_dev_replay['tau=0.2_eta=1']),
                       np.array(averages_replay['tau=0.2_eta=1']) + np.array(std_dev_replay['tau=0.2_eta=1']),
                       alpha=0.4)
ax1[1][0].fill_between(np.arange(20), np.array(averages_nonreplay['tau=0.2_eta=1']) -
                       np.array(std_dev_nonreplay['tau=0.2_eta=1']),
                       np.array(averages_nonreplay['tau=0.2_eta=1']) + np.array(std_dev_nonreplay[
	                                                                                   'tau=0.2_eta=1']),
                       alpha=0.2)


ax1[1][1].plot(averages_replay['tau=0.2_eta=10'], label='With replay')
ax1[1][1].plot(averages_nonreplay['tau=0.2_eta=10'], label='Without replay')
ax1[1][1].set_title('$\\tau_e = 0.2_eta; $\eta = 10$')
ax1[1][1].set_ylim(0,60)
ax1[1][1].legend()
ax1[1][1].fill_between(np.arange(20), np.array(averages_replay['tau=0.2_eta=10']) -
                       np.array(std_dev_replay['tau=0.2_eta=10']),
                       np.array(averages_replay['tau=0.2_eta=10']) + np.array(std_dev_replay['tau=0.2_eta=10']),
                       alpha=0.4)
ax1[1][1].fill_between(np.arange(20), np.array(averages_nonreplay['tau=0.2_eta=10']) -
                       np.array(std_dev_nonreplay['tau=0.2_eta=10']),
                       np.array(averages_nonreplay['tau=0.2_eta=10']) + np.array(std_dev_nonreplay[
	                                                                                   'tau=0.2_eta=10']),
                       alpha=0.2)

#################################################################

fig1, ax1 = plt.subplots(2, 2)

ax1[0][0].plot(averages_replay['tau=1_eta=0.01'])
ax1[0][0].plot(averages_nonreplay['tau=1_eta=0.01'])
ax1[0][0].set_title('$\\tau_e = 1s$; $\eta = 0.01$')
ax1[0][0].set_ylim(0,60)
ax1[0][0].fill_between(np.arange(20), np.array(averages_replay['tau=1_eta=0.01']) -
                       np.array(std_dev_replay['tau=1_eta=0.01']),
                       np.array(averages_replay['tau=1_eta=0.01']) + np.array(std_dev_replay['tau=1_eta=0.01']),
                       alpha=0.4)
ax1[0][0].fill_between(np.arange(20), np.array(averages_nonreplay['tau=1_eta=0.01']) -
                       np.array(std_dev_nonreplay['tau=1_eta=0.01']),
                       np.array(averages_nonreplay['tau=1_eta=0.01']) + np.array(std_dev_nonreplay[
	                                                                                   'tau=1_eta=0.01']),
                       alpha=0.2)

ax1[0][1].plot(averages_replay['tau=1_eta=0.1'])
ax1[0][1].plot(averages_nonreplay['tau=1_eta=0.1'])
ax1[0][1].set_title('$\\tau_e = 1s$; $\eta = 0.1$')
ax1[0][1].set_ylim(0,60)
ax1[0][1].fill_between(np.arange(20), np.array(averages_replay['tau=1_eta=0.1']) -
                       np.array(std_dev_replay['tau=1_eta=0.1']),
                       np.array(averages_replay['tau=1_eta=0.1']) + np.array(std_dev_replay['tau=1_eta=0.1']),
                       alpha=0.4)
ax1[0][1].fill_between(np.arange(20), np.array(averages_nonreplay['tau=1_eta=0.1']) -
                       np.array(std_dev_nonreplay['tau=1_eta=0.1']),
                       np.array(averages_nonreplay['tau=1_eta=0.1']) + np.array(std_dev_nonreplay[
	                                                                                   'tau=1_eta=0.1']),
                       alpha=0.2)


ax1[1][0].plot(averages_replay['tau=1_eta=1'])
ax1[1][0].plot(averages_nonreplay['tau=1_eta=1'])
ax1[1][0].set_title('$\\tau_e = 1s$; $\eta = 1$')
ax1[1][0].set_ylim(0,60)
ax1[1][0].fill_between(np.arange(20), np.array(averages_replay['tau=1_eta=1']) -
                       np.array(std_dev_replay['tau=1_eta=1']),
                       np.array(averages_replay['tau=1_eta=1']) + np.array(std_dev_replay['tau=1_eta=1']),
                       alpha=0.4)
ax1[1][0].fill_between(np.arange(20), np.array(averages_nonreplay['tau=1_eta=1']) -
                       np.array(std_dev_nonreplay['tau=1_eta=1']),
                       np.array(averages_nonreplay['tau=1_eta=1']) + np.array(std_dev_nonreplay[
	                                                                                   'tau=1_eta=1']),
                       alpha=0.2)

#############################################################

fig1, ax1 = plt.subplots(2, 2)

ax1[0][0].plot(averages_replay['tau=5_eta=0.01'])
ax1[0][0].plot(averages_nonreplay['tau=5_eta=0.01'])
ax1[0][0].set_title('$\\tau_e = 5s$; $\eta = 0.01$')
ax1[0][0].set_ylim(0,60)
ax1[0][0].fill_between(np.arange(20), np.array(averages_replay['tau=5_eta=0.01']) -
                       np.array(std_dev_replay['tau=5_eta=0.01']),
                       np.array(averages_replay['tau=5_eta=0.01']) + np.array(std_dev_replay['tau=5_eta=0.01']),
                       alpha=0.4)
ax1[0][0].fill_between(np.arange(20), np.array(averages_nonreplay['tau=5_eta=0.01']) -
                       np.array(std_dev_nonreplay['tau=5_eta=0.01']),
                       np.array(averages_nonreplay['tau=5_eta=0.01']) + np.array(std_dev_nonreplay[
	                                                                                   'tau=5_eta=0.01']),
                       alpha=0.2)


plt.show()