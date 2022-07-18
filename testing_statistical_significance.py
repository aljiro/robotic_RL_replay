from scipy.stats import wilcoxon
import numpy as np
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
	step = 42
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


from scipy import stats



tau_replay = '0.04'
eta_replay = '1'
tau_nonreplay = '1'
eta_nonreplay = '0.01'
for j in range(20):
	data1 = np.zeros(40)
	data2 = np.zeros(40)
	for i in range(40):
		data1[i] = results_full_replay['tau=' + tau_replay + '_eta=' + eta_replay][i][j]
		data2[i] = results_full_nonreplay['tau=' + tau_nonreplay + '_eta=' + eta_nonreplay][i][j]
	_, p_value = wilcoxon(data1, data2)
	#perform Friedman Test
	_, p_value_f = stats.friedmanchisquare(data1, data2)
	print(data1)
	print(data2)
	print('p-value for trial ' + str(j+1) + ' is: ' + str(p_value))