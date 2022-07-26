import csv

class RLLogger:
    def __init__(self, replay):
        self.data_dir = "data/trial_times"
        self.pnames =  ["trial_times", "random_times", "hitting", "activations", "total_counts"]
        self.replay = replay

    def initialize(self, tau_elig, eta):
        
        for i in range(len(self.pnames)):
            filename = self.data_dir + "/" + self.pnames[i]

            if self.replay:
                filename += "_WITH_REPLAY_FULL.csv"
            else:
                filename += "_NON_REPLAY_FULL.csv"

            with open(filename, 'a') as f:
                wr = csv.writer(f, quoting=csv.QUOTE_ALL)
                wr.writerow("")
                wr.writerow(["tau_elig=" + str(tau_elig), "eta=" + str(eta)])

    def saveAll( self, experiment_number, data ):
         for i in range(len(self.pnames)):
            filename = self.data_dir + "/" + self.pnames[i]

            if self.replay:
                filename += "_WITH_REPLAY_FULL.csv"
            else:
                filename += "_NON_REPLAY_FULL.csv"

            with open(filename, 'a') as f:
                wr = csv.writer(f, quoting=csv.QUOTE_ALL)
                wr.writerow([experiment_number] + data[self.pnames[i]])
           