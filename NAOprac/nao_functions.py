#import almath
#import argparse
from naoqi import ALProxy
import motion
import numpy as np
import time
import random as rnd
import copy
import images2
import cv2
import ConfigParser
import threading
from pybrain.tools.customxml import NetworkReader


def flip_angles(angs, js):
    """
    Given some andgles and joints, it returnes the angles for the joints on the other side of the body
    """
    angs_flipped = copy.copy(angs)
    for j1 in range(0, len(js)):
        joint1_name = js[j1]
        if joint1_name[0] == 'L':
            rest_of_joint1_name = joint1_name[1:]
            for j2 in range(0, len(js)):
                joint2_name = js[j2]
                rest_of_joint2_name = joint2_name[1:]
                if joint2_name[0] == 'R' and rest_of_joint1_name == rest_of_joint2_name and not ("Pitch" in rest_of_joint1_name):
                    angs_flipped[j1] = -angs[j2]
                    angs_flipped[j2] = -angs[j1]
                elif joint2_name[0] == 'R' and rest_of_joint1_name == rest_of_joint2_name:
                    angs_flipped[j1] = angs[j2]
                    angs_flipped[j2] = angs[j1]
    return angs_flipped


def load_network():
    net = NetworkReader.readFrom('trained_nn.xml')
    return net


#  net - the neural network, can load with load_network()
#  inp - the [position, velocity, angle] bin values
def get_nn_output(net, inp):
    action = net.activate(inp)
    return action


class Nao:
    def __init__(self, ip, port=9559):
        #SET UP THE NAO
        self.NAO_PORT = port  # 9559
        self.NAO_IP = ip  # "169.254.254.250"
        self.motionProxy = ALProxy("ALMotion", ip, port)
        self.autonomousLifeProxy = ALProxy("ALAutonomousLife", ip, port)
        self.postureProxy = ALProxy("ALRobotPosture", ip, port)
        self.tracker = ALProxy("ALTracker", ip, port)
        self.awareness = ALProxy('ALBasicAwareness', ip, port)
        self.camProxy = ALProxy("ALVideoDevice", ip, port)

        client = "python_GVM"
        resolution = 0    # 0: 160x120, 1: 320x240, 2: 640x480, 3: 1280x960
        color_space = 13   # BGR (because OpenCV)
        fps = 30
        self.videoClient = self.camProxy.subscribeCamera(client, 0, resolution, color_space, fps)

        config = ConfigParser.ConfigParser()
        config.read('config.ini')
        self.min_angle = float(config.get("nao_params", "right_angle_max"))
        self.max_angle = float(config.get("nao_params", "left_angle_max"))
        self.num_angles = float(config.get("nao_params", "num_angles"))

        # JOINTS AND ANGLES
        self.joints = ["LShoulderPitch", "LShoulderRoll",
                       "RShoulderPitch", "RShoulderRoll",
                       "LElbowYaw", "LElbowRoll", "LWristYaw",
                       "RElbowYaw", "RElbowRoll", "RWristYaw"]
        self.tilt_left = [0.18250393867492676, -0.08748006820678711, 0.6727747321128845, 0.14530789852142334, -1.9174580574035645, -0.5, 0.15029001235961914, 1.735066294670105, 0.9, -0.07674193382263184]#3313021659851074, 7286896109580994#[0.846985399723053, -0.09535059332847595, 0.14883995056152344, 0.07359004020690918, -1.5987539291381836, -0.8236891627311707, -0.25315189361572266, 1.498676061630249, 0.08287787437438965, -0.3237159252166748]
        self.tilt_right = flip_angles(self.tiltLeft, self.joints)
        self.angle_lr = self.tilt_left  # The current tilt in the left-right axis
        self.angle_lr_interpolation = int(self.num_angles / 2)  # The current tilt position

        self.tilt_back = []
        self.tilt_forward = []
        self.tilt_fb = []
        self.tilt_fb_interpolation = 0

        self.shoulder_roll_joints = ["LShoulderRoll", "RShoulderRoll"]
        self.shoulder_roll_angles = [-0.1, 0.1]

        self.hip_joints = ["LHipYawPitch"]
        self.hip_angles = [0.25]

    def initial_setup(self):
        """
        Initial set up of nao that must be done first thing
        """
        self.autonomousLifeProxy.setState("disabled")
        self.motionProxy.setBreathEnabled("Body", False)
        self.motionProxy.setIdlePostureEnabled("Body", False)
        self.motionProxy.setStiffnesses("Body", 0.1)
        self.awareness.stopAwareness()

        target_name = "RedBall"
        diameter_of_ball = 0.04
        self.tracker.registerTarget(target_name, diameter_of_ball)
        self.tracker.track(target_name)
        self.tracker.setTimeOut(100)

    def set_up_to_hold_tray(self):
        """
        Reset the nao so that it is in a position to hold the tray
        """
        speed = 0.5
        self.postureProxy.goToPosture("StandInit", speed)
        self.motionProxy.setAngles(self.hip_joints, self.hip_angles, speed)  # Make nao lean backwards a bit, helps with keeping tray flat
        self.motionProxy.setAngles(self.joints, self.starting_angles, speed)
        self.motionProxy.setAngles(self.shoulder_roll_joints, self.shoulder_roll_angles, speed)

    def get_q_matrix_from_file(self, q_mat_name):
        """
        Loads a Q matrix learnt by one of the methods

        q_mat_name is either q_learn or q_wal. Do not need to include .npy
        """
        self.learnt_q_matrix = np.load(q_mat_name)

    def interpolate_angles_fixed_lr(self, interpolation_value, num_interpolations=0):
        """
        Given the two sets of angles that describe the full left tilt anf full right tilt, 
        update the angles that describe the position in between, determined by interpolation_value
        """
        new_angs = [0 for i in self.tiltLeft]  # Construct empty array
        if num_interpolations == 0:
            num_interpolations = self.num_angles  # Use the value loaded from file by default
        proportion1 = float(interpolation_value) / float(num_interpolations)
        proportion2 = 1 - proportion1
        for i in range(0, len(new_angs)):
            new_angs[i] = proportion1 * self.tilt_left[i] + proportion2 * self.tilt_right[i]
        self.angle_lr = new_angs
        self.angle_lr_interpolation = interpolation_value

    def interpolate_angles_relative_lr(self, interpolation_value_change, num_interpolations=0):
        """
        Given the two sets of angles that describe the full left tilt anf full right tilt, 
        go the angles that describe the position in between, determined by change in interpolation value
        """
        self.angle_lr_interpolation = self.angle_lr_interpolation + interpolation_value_change  # Update the value based on the relaive change of interpolated angle 
        if num_interpolations == 0:
            num_interpolations = self.num_angles  # Use the value loaded from file by default

        if self.angle_lr_interpolation < 0:  # Make sure new interpolation value is bounded
            self.angle_lr_interpolation = 0
        elif self.angle_lr_interpolation > num_interpolations:
            self.angle_lr_interpolation = num_interpolations

        new_angs = [0 for i in self.tiltLeft]  # Construct empty array
        proportion1 = float(self.angle_lr_interpolation) / float(num_interpolations)
        proportion2 = 1 - proportion1
        for i in range(0, len(new_angs)):
            new_angs[i] = proportion1 * self.tilt_left[i] + proportion2 * self.tilt_right[i]
        self.angle_lr = new_angs

    def hands_open(self):
        hands = ["RHand", "LHand"]
        angs = [1, 1]
        self.motionProxy.setAngles(hands, angs, 1)

    def hands_grab(self):
        hands = ["RHand", "LHand"]
        angs = [0, 0]
        self.motionProxy.setAngles(hands, angs, 1)

    def go_to_angles(self, angles, joints, speed=0.5):
        self.motionProxy.setAngles(joints, angles, speed)

    def go_to_interpolated_angles_lr(self, speed=1):
        self.motionProxy.setAngles(self.joints, self.angle_lr, speed)
    
    def go_to_interpolated_angles_fb(self, speed=1):
        self.motionProxy.setAngles(self.joints, self.angle_fb, speed)

    def show_image(self):
        """
        Displays an image of what the NAO camera currently sees. 
        Uses functions from the images2.py file
        """
        images = self.camProxy.getImageRemote(self.videoClient)
        imgs = images2.toCVImg(images)
        cv2.imshow("image", imgs)
        cv2.waitKey(1)

    def continually_update_ball_information(self):
        """
        Spawns a new thread to keep the position and velocity of the ball updated at all times
        """
        thread = threading.Thread(target=self.__update_ball_information, args=())
        thread.daemon = True                            # Daemonize thread
        thread.start()

    def __update_ball_information(self):
        """
        The function called by the new thread to keep the ball information updated
        """
        self.time = time.time()
        self.ball_pos_lr = self.tracker.getTargetPosition(0)[1]
        self.ball_pos_fb = self.tracker.getTargetPosition(0)[0]
        # INCLUDE SOME KIND OF WEIGHT TIME
        time.sleep(0.1)
        while True:
            # Position
            self.ball_pos_lr_prev = self.ball_pos_lr  # Store the old ball positions
            self.ball_pos_fb_prev = self.ball_pos_fb

            pos = self.tracker.getTargetPosition(0)  # Gets the x, y, z coordinates of the ball. LR is y, FB is x
            self.ball_pos_lr = pos[1]
            self.ball_pos_fb = pos[0]

            pos_lr_diff = self.ball_pos_lr - self.ball_pos_lr_prev  # Calculate difference
            pos_fb_diff = self.ball_pos_fb - self.ball_pos_fb_prev

            # Time
            self.time_prev = self.time  # Store old time
            self.time = time.time()  # Set new time

            time_diff = self.time - self.time_prev  # Calculate difference

            # Velocity
            self.ball_vel_lr_prev = self.ball_vel_lr  # Store the old ball velocities
            self.ball_vel_fb_prev = self.ball_vel_fb

            self.ball_vel_lr = pos_lr_diff / time_diff  # Calculate new velocities
            self.ball_vel_fb = pos_fb_diff / time_diff

            # Angle
            self.ball_ang_lr = self.angle_lr_interpolation
            self.ball_ang_fb = self.angle_fb_interpolation


# Functions to add:
# - Rest function
# - Update angles of joints upon key press
# - Get angles and other data saved in config file
# - Check if lost ball - maybe also with threading