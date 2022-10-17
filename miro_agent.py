
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum

class MiRoState(Enum):
    STALLED = 0
    MOVE_RANDOM_LOCATION = 1
    WALL_AVOIDANCE = 2
    MOVE_TO_TARGET = 3

class MiRo(object):
    def state( self, state ):
        self.state = state

    def move( self ):
        self.state.move()



class ShamMiro(object):
    def __init__(self, x0, y0, theta0):
        self.pstep = 0.005
        self.ostep = 4.0
        self.x, self.y, self.theta = x0, y0, theta0
        self.scale = 0.05
        self.r = 1.0*self.scale # Wheel radius
        self.D = 2.0*self.scale # Diameter of the robot
        # Position of the sensors in the body frame of reference
        self.p_sensors = np.array([[self.r*np.cos(np.pi/4.0), self.r*np.sin(np.pi/4.0), 1],
                              [self.r*np.cos(-np.pi/4.0), self.r*np.sin(-np.pi/4.0), 1]]).T
        
    def getSensorTransform( self ):
        # This is the matrix defining the sensor to motor maping
        return np.array([[1, 0], [0, 1]])

    def getBodyTransform( self ):
        # Affine transform from the body to the work FoR
        return np.array([[np.cos(self.theta), -np.sin(self.theta), self.x],
                         [np.sin(self.theta), np.cos(self.theta), self.y],
                         [0.0, 0.0, 1.0]])
        
    def sense( self, G ): 
        k1 = 10.0
        k2 = 10.0
        P = self.getSensorTransform()
        T = self.getBodyTransform()
        # The position of the sensor is obtained in the world space
        # and the sensor permutation is also applied
        n_sensors = np.dot(T, np.dot(self.p_sensors, P))
        # Then we get the gradient value at those positions
        r1 = G(n_sensors[0,0], n_sensors[1,0])
        r2 = G(n_sensors[0,1], n_sensors[1,1])
        
        return k1*r1, k2*r2
        
    def wrap( self, x, y ):
        if x < 0:
            x = 2.0
        elif x > 2.0:
            x = 0.0
        
        if y < 0:
            y = 2.0
        elif y > 2.0:
            y = 0

        return x,y 

    def updatePosition(self, G):
        # Main step function
        # First sense
        phi_L, phi_R = self.sense(G)
        # Then compute forward kinematics
        vl = (self.r/self.D)*(phi_R + phi_L)
        omega = (self.r/self.D)*(phi_R - phi_L)

        # Update the next statefrom the previous one
        self.theta += self.ostep*omega
        self.x += self.pstep*vl*np.cos(self.theta)
        self.y += self.pstep*vl*np.sin(self.theta)
        self.x, self.y = self.wrap(self.x, self.y)

        return self.x, self.y, self.theta

    def draw(self, ax):
        T = self.getBodyTransform()
        n_sensors = np.dot(T, self.p_sensors)
        left_wheel = np.dot(T, np.array([0, self.D/2.0, 1]).T)
        right_wheel = np.dot(T, np.array([0, -self.D/2.0, 1]).T)

        # drawing body
        body = Circle((self.x, self.y), self.D/2.0, fill = False, color = [0, 0, 0] )
        # Drawing sensors
        s1 = Circle((n_sensors[0, 0], n_sensors[1, 0]), self.scale*0.1, color = 'red' )
        s2 = Circle((n_sensors[0, 1], n_sensors[1, 1]), self.scale*0.1, color = 'red' )
        w1 = Circle((left_wheel[0], left_wheel[1]), self.scale*0.2, color = 'black' )
        w2 = Circle((right_wheel[0], right_wheel[1]), self.scale*0.2, color = 'black' )

        ax.add_patch(body)
        ax.add_patch(s1)
        ax.add_patch(s2)
        ax.add_patch(w1)
        ax.add_patch(w2)