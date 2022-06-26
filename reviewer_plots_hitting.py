# Plotting the evolution of wall bumping and percentage of random 
# walks as a function of trials

#!/usr/bin/python3

'''
Plotting the best cases. For replay, this is tau_e = 0.04s, eta = 1; non-replay is tau_e = 1s, eta = 0.01
'''

import numpy as np
import matplotlib.pyplot as plt
import csv
import copy

N_EXP = 30
results_non_replay = []
results_replay = []
means_non_replay = []
means_replay = []
medians_non_replay = []
medians_replay = []
std_devs_non_replay = []
std_devs_replay = []
percentiles_non_replay = []
percentiles_replay = []

# Loading the random walk data
N_EXP = 15
with open('data/trial_times/hitting_NON_REPLAY_final.csv', newline='') as file:
	data = csv.reader(file, delimiter=',')
	for i, row in enumerate(data):
		if (1 <= i < N_EXP):
			results_non_replay.append([0 if j == "" else float(j) for j in row[1:]])

N_EXP = 30
with open('data/trial_times/hitting_WITH_REPLAY_final.csv', newline='') as file:
	data = csv.reader(file, delimiter=',')
	for i, row in enumerate(data):
		if (1 <= i < N_EXP):
			results_replay.append([0 if j == "" else float(j) for j in row[1:]])

for i in range(len(results_non_replay)):
    plt.plot(results_non_replay[i], 'b')

for i in range(len(results_replay)):
    plt.plot(results_replay[i], 'r')

plt.ylim(0,10)
	
plt.legend()
plt.savefig('wall_hitting_comparison.png')
plt.show()
