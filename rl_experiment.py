import rospy

class Experiment:

    def run( self, exp_no ):
        current_trial = 0

        while not rospy.core.is_shutdown() or current_trial > self.max_trials:
            rate = rospy.Rate(int(1 / self.delta_t))
            self.t += self.delta_t

            action, trial_end = self.model.step()

            if trial_end:
                self.model.restart()
                current_trial += 1
            else:
                self.miro.execute(action)


            rate.sleep()