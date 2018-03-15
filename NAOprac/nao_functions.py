from naoqi import ALProxy
import numpy as np
import time
import copy
import images2
import cv2
import ConfigParser
import threading
import math
from pybrain.tools.customxml import NetworkReader
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from msvcrt import getch


def flip_angles(angs, js):
    """
    Given some angles and joints, it returns the angles for the joints on the other side of the body
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


def __get_bin(val, max_val, num_val_bins):
        val_percent = (val+max_val) / (2 * max_val)
        val_bin = math.floor(val_percent * num_val_bins)
        if val_bin < 0:
            val_bin = 0
        elif val_bin >= num_val_bins:
            val_bin = num_val_bins-1
        return val_bin


def inputs_to_bins(inputs, input_max, num_bins):
    """
    Takes actual values of position, velocity and angle of ball and
    converts them to a discrete bin vale based on the possible ranges
    those values can take and the number of discrete bins for each value
    """
    pos = __get_bin(inputs[0], input_max[0], num_bins[0])
    vel = __get_bin(inputs[1], input_max[1], num_bins[1])
    ang = __get_bin(inputs[2], input_max[2], num_bins[2])
    return pos, vel, ang


def get_reward(p, v, np, nv):
    """
    Gets the reward based on the given state
    """
    reward = 0.0

    #pos_gap = 0
    # If near center, give big reward
    #if abs(p) >= int((np)/2) - pos_gap - 1 and abs(p) <= int((np)/2) + pos_gap:
    #    reward = 1

    #vel_gap = 2
    # If low velocity, give medium reward
    #if abs(v) >= int((nv)/2) - vel_gap and abs(v) <= int((nv)/2) + vel_gap:
    #    reward += 0.5

    # If it is by the edge areas, give a big punishment
    edge_gap = 1
    #if p >= np-1 - edge_gap or p <= 0 + edge_gap:
    #    reward = -1
    #else:
    #    reward = 1

    reward = -1
    if p >= int(np/2)-1-edge_gap and p <= int(np/2) + edge_gap:
        if v >= int(nv/2)-1-edge_gap and v <= int(nv/2)+edge_gap:
            reward = 1

    return reward


def get_reward_nn(inputs, max_values):
    reward = 0

    p = inputs[0]
    p_max = max_values[0]

    p_upper = int(p_max/2)
    p_lower = int(p_max/2)
    if p_max % 2 == 0:
        p_lower -= 1

    if p >= p_lower and p <= p_upper:
        reward = 0.5
    return reward


def update_nn(net, inputs, action, outputs, reward, future_action_value, learn_rate=0.001, discount_factor=0.99, momentum=0.99):
    outputs[action] = ((1 - learn_rate) * outputs[action]) + (learn_rate * (reward + discount_factor * max(future_action_value))) # Have to pass through twice

    ds = SupervisedDataSet(3, 3)
    ds.addSample(list(inputs), outputs)

    train = BackpropTrainer(net, ds, learningrate=learn_rate, momentum=momentum)
    train.trainEpochs(1)

    print inputs, action, outputs, reward
    return net


def update_q(q, prev_state, curr_state, action, reward, learn_rate=0.5, discount_factor=0.99, prnt=False):
    """
    Update the Q matrix
    """
    # The old state
    p = int(prev_state[0])
    v = int(prev_state[1])
    a = int(prev_state[2])
    old_val = copy.copy(q[p][v][a])

    # The new state
    p2 = int(curr_state[0])
    v2 = int(curr_state[1])
    a2 = int(curr_state[2])

    # Update and normalise
    q[p][v][a][action] = ((1-learn_rate) * old_val[action]) + learn_rate * (reward + discount_factor * max(q[p2][v2][a2]))
    #q[p][v][a] = __normalise(q[p][v][a], p, v, a)
    if prnt:
        print(old_val, p, v, a, q[p][v][a], p2, v2, a2, q[p2][v2][a2], (reward + discount_factor * max(q[p2][v2][a2])))
    return q


def __normalise(q, p_var, v_var, a_var):
    """
    Maps all values to between -1 and 1
    """
    q_var = copy.copy(q)
    sumup = 0
    for i in q_var:
        sumup += abs(i)
    if sumup > 0:
        for i in range(0, len(q)):
            q_var[i] = (q[i]/sumup)
    return q_var


def normalise_whole_q(q):
    """
    Normalises the whole q matrix
    """
    q_copy = copy.copy(q)
    for p in range(0, len(q)):
        for v in range(0, len(q[p])):
            for a in range(0, len(q[p][v])):
                sumup = 0
                for action in range(0, len(q[p][v][a])):
                    sumup += abs(q[p][v][a][action])
                if sumup > 0:
                    for action in range(0, len(q[p][v][a])):
                        q_copy[p][v][a][action] = q[p][v][a][action] / sumup
    return q_copy


def load_network():
    net = NetworkReader.readFrom('nn_trained_networks/trained_nn.xml')
    return net


def load_q_matrix(p, v, a, act=2, is_nao=False, is_delay=False):
    extra = ""
    if is_nao:
        extra += "nao_"
    if is_delay:
        extra += "DELAY_"
    fname = "q_mats/q_" + extra + str(p) + "_" + str(v) + "_" + str(a) + "_" + str(act) + ".npz"
    data_from_file = np.load(fname)
    print "Loading: ", fname
    q_mat = data_from_file["q"]
    #print(q_mat)
    return q_mat


def save_q(q, p, v, a, act=2, delay=False, er=False):
    """
    Saves the Q matrix as a zipped NumPy binary file
    Also includes an array called "metadata" with information on the Q
    matrix
    """
    # metadata = {
    #                 "num_pos": self.num_bins_pos,
    #                 "num_vel": self.num_bins_vel,
    #                 "num_ang": self.num_bins_ang,
    #                 "num_actions": self.num_actions,
    #                 "max_pos": self.max_pos,
    #                 "max_vel": self.max_vel,
    #                 "max_ang": self.max_ang
    #             }
    extra = ""
    if delay:
        extra = "DELAY_"
    elif er:
        extra = "er_"
    np.savez("q_mats/q_nao_" + extra + str(p) + "_" + str(v) + "_" + str(a) + "_" + str(act) + ".npz", q=q)


def load_exp(p, v, a, act):
    data_from_file = np.load("nao_experiences/nao_exp_" + str(p) + "_" + str(v) + "_" + str(a) + "_" + str(act) + ".npz")
    return data_from_file["exp"]


def save_exp(exp, p, v, a, max_p, max_v, max_a, act):
    metadata = {
                "num_pos": p,
                "num_vel": v,
                "num_ang": a,
                "num_actions": act,
                "max_pos": max_p,
                "max_vel": max_v,
                "max_ang": max_a
            }
    np.savez("nao_experiences/nao_exp_"+str(p)+"_"+str(v)+"_"+str(a)+"_"+str(act)+".npz", exp=exp, metadata=metadata)


#  net - the neural network, can load with load_network()
#  inp - the [position, velocity, angle] bin values
def get_nn_output(net, inp):
    action = net.activate(inp)
    return action


def get_action(q_mat, p, v, a, max_p, max_v):
    """
    Gets the action, using a consensus of surrounding cells
    """
    votes = []
    votes.append(np.argmax(q_mat[p][v][a]))
    if p+1 < max_p and abs(sum(q_mat[p+1][v][a])) > 0:
        votes.append(np.argmax(q_mat[p+1][v][a]))
    if p-1 >= 0 and abs(sum(q_mat[p-1][v][a])) > 0:
        votes.append(np.argmax(q_mat[p-1][v][a]))
    if v+1 < max_v and abs(sum(q_mat[p][v+1][a])) > 0:
        votes.append(np.argmax(q_mat[p][v+1][a]))
    if v-1 >= 0 and abs(sum(q_mat[p][v-1][a])) > 0:
        votes.append(np.argmax(q_mat[p][v-1][a]))

    action = np.argmax(q_mat[p][v][a])  # By dafault, and in case of tie, have it be the action of current state
    if sum(votes) > len(votes)/2.0:  # True if majority of votes are 1
        action = 1
    elif sum(votes) < len(votes)/2.0:
        action = 0
    print "Original:", np.argmax(q_mat[p][v][a]), "decided:", action, "votes:", votes, sum(votes), len(votes)/2.0

    return action
    


class Nao:
    def __init__(self, ip, angs=21, port=9559):
        # SET UP THE NAO
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

        # JOINTS AND ANGLES
        self.joints = ["LShoulderPitch", "LShoulderRoll",
                       "RShoulderPitch", "RShoulderRoll",
                       "LElbowYaw", "LElbowRoll", "LWristYaw",
                       "RElbowYaw", "RElbowRoll", "RWristYaw"]


        # self.tilt_left = [0.18250393867492676, -0.15,
        #                  0.6727747321128845, 0.17,
        #                  -1.7, -0.8, 0.15029001235961914,
        #                  1.6, 0.5, -0.1]

        ##self.tilt_left = [0.5194180607795715, -0.15755724906921387, 0.33260154724121094, 0.15440550446510315, -1.6312140226364136, -0.592301607131958, 0.11547386646270752, 1.6693739891052246, 0.7076734900474548, -0.13481613993644714]
        self.tilt_left = [0.29525595903396606, -0.15372799336910248, 0.5568179488182068, 0.15815134346485138, -1.6769975423812866, -0.7307573556900024, 0.13868460059165955, 1.6235711574554443, 0.5692337155342102, -0.11160540580749512]
        # self.min_angle = float(config.get("nao_params", "right_angle_max"))
        # self.max_angle = float(config.get("nao_params", "left_angle_max"))
        self.num_angles = angs  # float(config.get("nao_params", "num_angles"))

        self.tilt_right = flip_angles(self.tilt_left, self.joints)
        self.angle_lr = self.tilt_left  # The current tilt in the left-right axis
        self.angle_lr_interpolation = int(self.num_angles / 2)  # The current tilt position

        self.tilt_back = []
        self.tilt_forward = []
        self.angle_fb = []
        self.angle_fb_interpolation = 0

        self.shoulder_roll_joints = ["LShoulderRoll", "RShoulderRoll"]
        self.shoulder_roll_angles = [-0.8, 0.8]

        self.hip_joints = ["LHipYawPitch"]
        self.hip_angles = [0.25]

        # BALL INFORMATION
        self.ball_pos_lr = 0
        self.ball_pos_fb = 0
        self.ball_vel_lr = 0
        self.ball_vel_fb = 0
        self.ball_ang_lr = self.angle_lr_interpolation
        self.ball_ang_fb = 0

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
        speed = 0.2
        self.postureProxy.goToPosture("StandInit", speed)
        time.sleep(0.5)

        self.motionProxy.setAngles(self.hip_joints, self.hip_angles, speed)  # Make nao lean backwards a bit, helps with keeping tray flat
        time.sleep(0.5)

        self.motionProxy.setAngles(self.shoulder_roll_joints, self.shoulder_roll_angles, speed)
        time.sleep(0.5)

        self.interpolate_angles_fixed_lr(int(self.num_angles/2), self.num_angles)  # Set angles to be middle value
        self.go_to_interpolated_angles_lr(speed=speed)
        time.sleep(0.5)

    def get_q_matrix_from_file(self, q_mat_name):
        """
        Loads a Q matrix learnt by one of the methods

        q_mat_name is either q_learn or q_wal. Do not need to include .npy
        """
        self.learnt_q_matrix = np.load(q_mat_name)

    def interpolate_angles_fixed_lr(self, interpolation_value, num_interpolations=0):
        """
        Given the two sets of angles that describe the full left tilt and full
        right tilt, update the angles that describe the position in between,
        determined by interpolation_value
        """
        new_angs = [0 for i in self.tilt_left]  # Construct empty array
        if num_interpolations == 0:
            num_interpolations = self.num_angles -1 # Use the value loaded from file by default
        proportion1 = float(interpolation_value) / float(num_interpolations)
        proportion2 = 1 - proportion1
        for i in range(0, len(new_angs)):
            new_angs[i] = proportion1 * self.tilt_left[i] + proportion2 * self.tilt_right[i]
        #print proportion1, proportion2, interpolation_value, num_interpolations
        self.angle_lr = new_angs
        self.angle_lr_interpolation = interpolation_value

    def interpolate_angles_relative_lr(self, interpolation_value_change, num_interpolations=0, lower_bound=0, upper_bound=0):
        """
        Given the two sets of angles that describe the full left tilt and full right tilt,
        go the angles that describe the position in between, determined by change in interpolation value
        """
        self.angle_lr_interpolation = self.angle_lr_interpolation + interpolation_value_change  # Update the value based on the relative change of interpolated angle
        if num_interpolations == 0:
            num_interpolations = self.num_angles-1  # Use the value loaded from file by default
        if upper_bound == 0:
            upper_bound = num_interpolations

        if self.angle_lr_interpolation < lower_bound:  # Make sure new interpolation value is bounded
            self.angle_lr_interpolation = lower_bound
        elif self.angle_lr_interpolation > upper_bound:
            self.angle_lr_interpolation = upper_bound

        new_angs = [0 for i in self.tilt_left]  # Construct empty array
        proportion1 = float(self.angle_lr_interpolation) / float(num_interpolations)
        proportion2 = 1.0 - proportion1
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

    def get_angles(self, joints):
        return self.motionProxy.getAngles(joints, False)  # False means absolute not relative

    def go_to_interpolated_angles_lr(self, speed=1):
        #id = self.motionProxy.post.setAngles(self.joints, self.angle_lr, speed)
        id = self.motionProxy.post.angleInterpolationWithSpeed(self.joints, self.angle_lr, speed)
        #print self.motionProxy.isRunning(id)
        self.motionProxy.wait(id, 0)
        #print self.motionProxy.isRunning(id)

    def go_to_interpolated_angles_fb(self, speed=1):
        self.motionProxy.setAngles(self.joints, self.angle_fb, speed)

    def rest(self):
        self.motionProxy.rest()

    def stand(self, speed=0.2):
        self.postureProxy.goToPosture("StandInit", speed)
        self.motionProxy.setAngles(self.hip_joints, self.hip_angles, speed)
        self.interpolate_angles_fixed_lr(5, 10)  # Set angles to be middle value
        self.go_to_interpolated_angles_lr(speed=speed)

    def show_image(self):
        """
        Displays an image of what the NAO camera currently sees.
        Uses functions from the images2.py file
        """
        images = self.camProxy.getImageRemote(self.videoClient)
        imgs = images2.toCVImg(images)
        cv2.imshow("image", imgs)
        cv2.waitKey(1)

    def record_angles(self, countdown=0):
        """
        Prints out the angles of all the upper body joints.
        If the countdown parmeter is 0, then it will wait until enter is
        pressed before recording otherwise, will count down the specified time
        """
        if countdown <= 0:
            raw_input("Press enter when ready ")
        else:
            time.sleep(countdown)
        angs = self.motionProxy.getAngles(self.joints, False)
        print(angs)

    def get_tray_angle(self):
        """
        Returns the angles between the hands, based on their position
        in 3D space
        """
        xl, yl, _, _, _, _ = self.motionProxy.getPosition("LHand", 2, True)
        xr, yr, _, _, _, _ = self.motionProxy.getPosition("RHand", 2, True)
        angle = np.arctan((xl-xr)/(yl-yr))
        return angle

    def manipulate_limbs(self, move_size=0.1):
        """
        Allows for control of each limb via keyboard input
        """
        for j in range(0, len(self.joints)):
            joint = self.joints[j]
            print joint
            move_this_joint = True
            while move_this_joint:
                move = getch()
                # print(ord(move))
                ang = self.get_angles(joint)[0]
                if move == 'j':
                    self.go_to_angles(ang + move_size, joint)
                elif move == 'k':
                    self.go_to_angles(ang - move_size, joint)
                elif move == 'l':
                    move_this_joint = False
                elif move == ' ':
                    print self.get_angles(joint)[0]

    def continually_update_ball_information(self, wait_time=0.1):
        """
        Spawns a new thread to keep the position and velocity of the ball updated at all times
        """
        thread = threading.Thread(target=self.__update_ball_information, args=([wait_time]))
        thread.daemon = True  # Daemonize thread
        thread.start()

    def __update_ball_information(self, wait_time=0.1):
        """
        The function called by the new thread to keep the ball information updated
        """
        self.time = time.time()
        self.ball_pos_lr = self.tracker.getTargetPosition(0)[1]
        self.ball_pos_fb = self.tracker.getTargetPosition(0)[0]
        # INCLUDE SOME KIND OF WEIGHT TIME
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
            time.sleep(wait_time)
            self.time = time.time()  # Set new time

            time_diff = self.time - self.time_prev  # Calculate difference

            # Velocity
            self.ball_vel_lr_prev = self.ball_vel_lr  # Store the old ball velocities
            self.ball_vel_fb_prev = self.ball_vel_fb

            self.ball_vel_lr = pos_lr_diff / time_diff  # Calculate new velocities
            self.ball_vel_fb = pos_fb_diff / time_diff

            # Angle
            self.ball_ang_lr = self.angle_lr_interpolation  # self.get_tray_angle()
            self.ball_ang_fb = self.angle_fb_interpolation  # UPDATE WHEN FB WORKING


# Functions to add:
# - Get angles and other data saved in config file
# - Check if lost ball - maybe also with threading
# - Change loading q matrix so that it applies the metadata to nao

def prnt():
    experiences = load_exp(12, 12, 10, 2)
    for i in range(0, len(experiences)):
            for j in range(0, len(experiences[i])):
                for k in range(0, len(experiences[i][j])):
                    print i, j, k, experiences[i][j][k]

def clean_experiences(experiences):
    for i in range(0, len(experiences)):
        for j in range(0, len(experiences[i])):
            for k in range(0, len(experiences[i][j])):
                to_remove = []
                for e in range(0, len(experiences[i][j][k][0])):
                    if experiences[i][j][k][0][e]["new_state"][2] == k:  # Angel hasn't changed
                        to_remove.append((i, j, k, e))
                offset = 0
                for r in to_remove:
                    arr = experiences[r[0]][r[1]][r[2]][0]
                    print(r)
                    experiences[r[0]][r[1]][r[2]][0] = np.delete(arr, r[3]-offset)
                    offset+=1
    return experiences
#prnt()