import rospy
from enum import Enum
from miro_agent import *
from rl_brain import *

class TimerState(Enum):
    GOAL_FOUND = 0
    WALL = 1
    MOVING = 2

class RoboTimer():
    def __init__(self):
        self.h = 0.01
        self.t = 0
        self.state = TimerState.MOVING

    def tick( self, state ):
        if state != self.state:
            self.state = state
            self.t = 0

        self.t += self.h

class RL_Algorithm(object):
    def __init__(self, replay = False):
        self.t = 0
        self.replay = replay
        self.R = 0
        self.vlambda = 0 
        self.brain = RLBrain()

    def initialize( self ):
        return False
        
    def goalFound( self ):
        return self.R == 1

    def updateNetwork( self, replay ):
        # Equations 4 and 6
        # If replay, adds STP
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

    def step( self ):
        # Is the yreplay a moving target?
        self.currentAction = Action.STOP

        if self.goalFound():
            self.timer.tick(TimerState.GOAL_FOUND)
            self.R = self.vlambda = 1.0
            self.updateWeights()
            
            if self.replay:
                if 1.0 < self.timer.elapsed() < 1.1:
                    self.runReplay()
                self.updateWeights()
            
            # Update elegibility trace for 2 seconds
            if self.timer.elapsed() > 2:
                self.R = self.vlambda = 0.0
                self.currentAction = Action.MOVE_RANDOM_LOCATION
        elif self.wall:
            self.timer.tick(TimerState.WALL)

            if self.timer.elapsed() < 0.5:
                self.R = -1
                self.currentAction = Action.WALL_AVOIDANCE
                self.udpdateWeights()
            else:
                self.timer.stop()
                self.wall = False
        else:
            self.timer.tick(TimerState.MOVING)
            self.R = 0

            if self.timer.tick() > 0.5:
                if self.acDrive():
                    theta = self.getActionCellDirection()
                else:
                    theta = self.getRandomWalkDirection()
                
                self.currentAction = Action.MiRoState.MOVE_TO_TARGET
                self.currentAction.setTheta(theta)
                self.timer.stop()
        
        self.updateNetworkVariables()
        return self.currentAction





