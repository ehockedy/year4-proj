
import random

import pygame
from pygame.locals import *
from pygame.color import *

import pymunk
from pymunk import Vec2d
import pymunk.pygame_util

import numpy as np
import math
import copy
import time

import configparser
import json 

config = configparser.ConfigParser()
config.read('config.ini')

pygame.init()
screen = pygame.display.set_mode((600, 600))
clock = pygame.time.Clock()
running = True

### Physics stuff
space = pymunk.Space()
#space.gravity = (0.0, -900.0)
space._set_gravity(Vec2d(0, -981))
#space.gravity = (0.0, 0.0)
draw_options = pymunk.pygame_util.DrawOptions(screen)

w = 400
h = 20
fp = [(w/2,-h/2), (-w/2, h/2), (w/2, h/2), (-w/2, -h/2)]
mass = 100
moment = pymunk.moment_for_poly(mass, fp[0:2])

x_pos = 300
y_pos = 100
angle = np.pi/24


trayBody = pymunk.Body(mass, moment)
trayBody.position = x_pos, y_pos
trayBody.angle = angle
trayShape = pymunk.Poly(trayBody, fp)
space.add(trayBody, trayShape)

trayJointBody = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
trayJointBody.position = trayBody.position
j = pymunk.PinJoint(trayBody, trayJointBody, (0, 0), (0, 0))
space.add(j)

reset = False
addBall = False

class BallBalancer:
    def __init__(self):
        
        self.tray_width = 400
        self.tray_height = 0.001
        self.tray_x_pos = 300
        self.tray_y_pos = 100
        self.tray_angle = -0.05#np.pi / 24
        self.rotation = 600000
        
        self.ball_radius = 25

        self.NUM_ACTIONS = 2  # Number of actions that can be taken. Normally 2, rotate clockwise or anticlockwise
        self.NUM_X_DIVS = 2  # Number of divisions x plane is split into, for whole tray
        self.NUM_VELOCITIES = 49  # Number of velocity buckets
        self.NUM_ANGLES = int(config["nao_params"]["num_angles"]) #2 * np.pi
        self.MAX_VELOCITY = 400#math.sqrt(2 * mass/100 * 981/100 * math.sin(MAX_ANGLE) * w)  # Using SUVAT and horizontal component of gravity, /100 because of earlier values seem to be *100

        self.MAX_ANGLE = float(config["nao_params"]["left_angle_max"])
        self.MIN_ANGLE = float(config["nao_params"]["right_angle_max"])

        self.NUM_NAO_ANGLES = (self.__get_state(0, 0, self.MAX_ANGLE)[2]-1) - (self.__get_state(0, 0, self.MIN_ANGLE)[2]+1)
        self.NUM_ANGLE_BINS = self.NUM_NAO_ANGLES
        self.first_ang = self.__get_state(0, 0, self.MIN_ANGLE)[2]+1
        self.last_ang = self.__get_state(0, 0, self.MAX_ANGLE)[2]-1
        #int((abs(self.MAX_ANGLE) + abs(self.MIN_ANGLE)) / (np.pi*2.0) * float(self.NUM_ANGLES))

        self.iterations = 0

        self.Q = np.zeros((self.NUM_X_DIVS, self.NUM_VELOCITIES, self.NUM_ANGLES, self.NUM_ACTIONS))
        self.Q.fill(0.5)

        self.best_time = 0
        self.time_threshold = 0.7
        self.real_time_threshold = 30

        self.explore_rate = 0.2
        self.learn_rate = 0.8
        self.discount_factor = 0.9
        self.scale_reduction = 2#1.7

        self.max_time = 100
        self.threshold_counter = 0
        self.num_thresholds = 15
        self.max_num_iterations = 1000
        self.max_draw_iterations = 500
        self.explore_reduction_freq = 50

        self.train = True  # If train and update Q
        self.start_time = time.time()

        self.Qfreq = copy.copy(self.Q)

        self.a_high = 0
        self.a_low = self.NUM_ANGLES
        self.a = 0.0
        self.p = 0.0
        self.v = 0.0

        self.num_dists = 0
        self.tot_dist = 0
        self.num_frames_off_tray = 0  # The number of frames in which the ball has not been in contact with the tray
        self.length_on_tray = 50

        self.trained = False
        self.draw = True

        self.prev_ball_pos = 0

        self.speed_past_origin = None

        self.ball_changed_direction = False
        self.was_negative = False
        self.was_positive = False

    def set_up_pygame(self):
        pygame.init()
        self.screen = pygame.display.set_mode((600, 600))
        self.clock = pygame.time.Clock()

        # Physics stuff
        self.space = pymunk.Space()
        self.space._set_gravity(Vec2d(0, -981))
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)

        #print(self.NUM_NAO_ANGLES, self.__get_state(0, 0, 0)[2], self.__get_state(0, 0, self.MIN_ANGLE)[2]+1, self.__get_state(0, 0, self.MAX_ANGLE)[2]-1)

    def create_world(self):
        fp = [(self.tray_width/2, -self.tray_height/2), (-self.tray_width/2, self.tray_height/2-10), (self.tray_width/2, self.tray_height/2-10), (-self.tray_width/2, -self.tray_height/2)]
        mass = 100
        moment = pymunk.moment_for_poly(mass, fp[0:2])

        self.trayBody = pymunk.Body(mass, moment)
        self.trayBody.position = self.tray_x_pos, self.tray_y_pos
        self.trayBody.angle = self.tray_angle
        trayShape = pymunk.Poly(self.trayBody, fp)
        self.space.add(self.trayBody, trayShape)

        trayJointBody = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        trayJointBody.position = trayBody.position
        j = pymunk.PinJoint(self.trayBody, trayJointBody, (0, 0), (0, 0))
        self.space.add(j)

        #self.__add_ball(400, 200, 1, self.ball_radius, self.tray_width)
    
    def update_tray_angle(self, angle):
        self.trayBody.angle = angle
        self.tray_angle = angle

    def action_to_num_and_dir(self, action, num_actions):
        num_actions_in_each_dir = math.ceil(num_actions/2.0)
        num = (action % num_actions_in_each_dir) + 1
        dire = -1
        if action >= num_actions_in_each_dir:
            dire = 1
            num = (num_actions_in_each_dir+1) - num
        #print("act:", action, "na:", num_actions, "ned:", num_actions_in_each_dir, "num:", num, "dir:", dire)
        return num, dire

    # direction=-1 is clockwise
    # direction=1 is anticlockwise
    def do_action(self, num_of_angs_to_move, direction, slow=False, record=True, draw=False, speed=60):
        turn = True
        if not abs(direction) == 1:
            direction = 0
            turn = False
            print("DIRECTION VALUE IS WRONG IN do_action")
        target_bin = self.a + direction*num_of_angs_to_move  # The bin of the angle we are aiming for
        target_angle = self.MIN_ANGLE + target_bin * (self.MAX_ANGLE - self.MIN_ANGLE)/self.NUM_ANGLE_BINS  # The actual angle we are aiming for NUM_ANGLE_BINS should be changes, probably to a parameter
        # print(self.a, num_of_angs_to_move, direction, target_angle, self.MIN_ANGLE, self.MAX_ANGLE, self.NUM_ANGLE_BINS)
        if target_bin > self.NUM_ANGLE_BINS:
            target_bin = self.NUM_ANGLE_BINS
        elif target_bin < 0:
            target_bin = 0
        if target_angle > self.MAX_ANGLE:
            target_angle = self.MAX_ANGLE
        elif target_angle < self.MIN_ANGLE:
            target_angle = self.MIN_ANGLE
        # print(self.a, num_of_angs_to_move, direction, target_angle, self.MIN_ANGLE, self.MAX_ANGLE, self.NUM_ANGLE_BINS)
        while turn and self.is_ball_on_tray(self.length_on_tray):
            # print(self.trayBody.angle, target_angle, self.MIN_ANGLE, target_bin)
            if direction == -1:
                if self.trayBody.angle > target_angle and self.trayBody.angle > self.MIN_ANGLE:  # Keep rotating util we are past the angle we are aiming for
                    self.trayBody.apply_force_at_local_point(Vec2d.unit() * self.rotation, (-self.tray_width/2, 0))  # rotate flipper clockwise
                else:
                    turn = False
                    self.ball.body.velocity[1] = 0
            elif direction == 1:
                # print(self.NUM_ANGLE_BINS, self.trayBody.angle, target_angle, self.MIN_ANGLE, target_bin)
                if self.trayBody.angle < target_angle and self.trayBody.angle < self.MAX_ANGLE:
                    self.trayBody.apply_force_at_local_point(Vec2d.unit() * self.rotation, (self.tray_width/2, 0))  # rotate flipper anticlockwise
                else:
                    turn = False
                    self.ball.body.velocity[1] = 0
            self.step_simulation(slow, record, draw, speed)
        self.a = target_bin
        self.trayBody.angular_velocity = 0
        self.prev_ball_pos = self.get_pos_ball_along_tray()
        
    def step_simulation(self, slow_tray=True, record_speed=True, draw=False, draw_speed=60):       
        dt = 1.0/60.0/5.
        for x in range(5):
            self.space.step(dt)

        if slow_tray:
            self.trayBody.angular_velocity = 0

        if record_speed:
            c_speed = self.speed_of_ball_at_centre()
            if c_speed is not None:
                self.speed_past_origin = c_speed
            else:
                self.speed_past_origin = None

            self.ball_changed_direction = self.has_ball_changed_direction()  # Used because an action is only good if it changes the direction of the ball

        if draw:
            self.draw_scene(draw_speed)

    def __get_state(self, pos, vel, ang):
        p_bin = math.floor(((pos + (self.tray_width/2)) / (self.tray_width) * self.NUM_X_DIVS))
        v_bin = math.floor(((vel + self.MAX_VELOCITY) / (self.MAX_VELOCITY*2)) * self.NUM_VELOCITIES)
        a_bin = (math.floor(((ang + np.pi) / (2*np.pi)) * self.NUM_ANGLES + self.NUM_ANGLES//4)) % self.NUM_ANGLES  # starting from right is 0, then increases going anticlockwise
        # In terms of normal angles, when horizontal, angle is 0
        # When going from horizontal and turning anticlockwise angle goes from 0 and increases, with pi at other horizontal
        # When going clockwise, angle goes below 0 and then decreases
        # if abs(pos) >= self.tray_width/2:
            # p_bin = -1
        if p_bin > 1:
            p_bin = 1
        elif p_bin < 0:
            p_bin = 0
        return p_bin, v_bin, a_bin

    def normalised(self, q, actions):
        sumup = 0
        for i in actions:
            sumup += abs(i)
        actions_copy = copy.copy(actions)
        if sumup > 0:
            for i in range(0, len(actions)):
                actions_copy[i] = actions[i]/sumup
        return actions_copy

    def add_ball(self, xdist, vel, angle=0, mass=1, radius=25):
        if angle == 0:
            angle = self.tray_angle
        inertia = pymunk.moment_for_circle(mass, 0, radius, (0, 0))
        body = pymunk.Body(1, inertia)
        ang = np.pi/2 - angle - np.arctan2(radius, xdist)  # Right angle - angle of tray - angle from the tray to the line that goes through the centre of the ball
        hyp = np.sqrt(xdist * xdist + radius * radius)  # Dist from centre of tray to centre of ball
        xpos = hyp * np.sin(ang)  # Distance in x direction of ball, from centre of tray
        ypos = hyp * np.cos(ang)
        body.position = self.tray_x_pos + xpos, self.tray_y_pos + ypos + 0.1  # extra 0.1 makes sure does not lie in the tray
        shape = pymunk.Circle(body, radius, (0, 0))
        shape.elasticity = 0  # 0.95
        body.velocity = Vec2d(vel, 0)
        self.space.add(body, shape)
        shape.current_velocity = vel
        shape.current_position = self.__get_ball_pos_x(shape)
        shape.current_action = 0
        shape.current_angle = 0
        shape.previous_action = 0
        shape.previous_velocity = 0
        shape.previous_position = 0
        shape.previous_angle = 0
        shape.angle = angle
        shape.rad = radius
        shape.xdist = xdist
        shape.mass = 1
        self.prev_ball_pos = xdist
        self.ball = shape
        self.ball_body = body
        self.num_frames_off_tray = 0  # Reset the numebr of frams the ball has been off the tray for
        self.ball_changed_direction = False  # THIS AND THE NEXT TWO SHOULD BE MADE A PROPERTY OF THE BALL
        self.was_negative = False
        self.was_positive = False

    def get_pos_ball_along_tray(self):
        # ang = np.pi/2 - self.ball.angle - np.arctan2(self.ball.rad, self.ball.xdist)  # Right angle - angle of tray - angle from the tray to the line that goes through the centre of the ball
        # hyp = np.sqrt(self.ball.xdist * self.ball.xdist + self.ball.rad * self.ball.rad)  # Dist from centre of tray to centre of ball
        # xpos = hyp * np.sin(ang) 
        x = self.ball.body.position[0] - self.tray_x_pos
        return x / np.cos(self.trayBody.angle)

    def remove_ball(self):
        self.space.remove(self.ball, self.ball.body)

    def __get_ball_pos_x(self, ball):
        pos = -1
        dist_squared = (ball.body.position.x - self.tray_x_pos)**2 + (ball.body.position.y - self.tray_y_pos)**2 - (self.ball_radius+self.tray_height/2)**2
        if(dist_squared < 0):
            pos = 0
        else:
            pos = math.sqrt(dist_squared)
        if ball.body.position.x < x_pos:
            pos = pos * -1
        return pos

    def __get_centre_of_ang_bin(self, b):
        jump = 2*np.pi / self.NUM_ANGLES  # Size of a bin
        bin_pos = (b + 0.5) % self.NUM_ANGLES  # The bth bin, and then half way through that
        centre_angle = (jump * bin_pos - np.pi - np.pi/2)  # -pi since tray angles go from -pi to +pi, and want 0 to 2pi
        return centre_angle

    def reset_draw(self):
        self.screen = pygame.display.set_mode((600, 600))
        self.clock = pygame.time.Clock()
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
    
    def draw_scene(self, draw_speed=60):
        self.screen.fill(THECOLORS["white"])
        self.space.debug_draw(self.draw_options)

        self.clock.tick(draw_speed)
        pygame.display.flip()
    
    def continue_running(self):
        running = True
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == KEYDOWN and event.key == K_ESCAPE:
                running = False
        return running

    def get_reduced_q(self):
        return self.Q[:, :, self.first_ang: self.last_ang]

    def get_ball_info_bins(self):
        return self.__get_state(self.ball.current_position, self.ball.current_velocity, self.ball.current_angle)

    def get_ball_info(self):
        pos = self.get_pos_ball_along_tray()
        vel = self.get_x_velocity(self.ball.body.velocity)
        ang = self.trayBody.angle
        return pos, vel, ang

    def avg_dist(self):
        self.tot_dist += abs(self.ball.current_position)
        self.num_dists += 1
        return self.tot_dist / self.num_dists

    def reset_avg_dist(self):
        self.tot_dist = 0
        self.num_dists = 0

    def is_ball_on_tray(self, limit=50):
        val = False
        if len(self.ball.shapes_collide(self.space.shapes[0]).points) > 0:  # If there is contact, return true
            val = True

            self.num_frames_off_tray = 0  # Set the counter for number of frames without contact to zero
        elif self.num_frames_off_tray < limit:  # If the number of frames without contact is below the threshold. If false, false will be returned i.e. no longer in contact.
            self.num_frames_off_tray += 1  # increase the counter
            val = True  # Return true
            #print(self.num_frames_off_tray)
        #print(len(self.ball.shapes_collide(self.space.shapes[0]).points), self.num_frames_off_tray)
        return val

    def speed_of_ball_at_centre(self):
        vel = None
        if self.get_pos_ball_along_tray() < 0 and self.prev_ball_pos >= 0:  # If the ball goes past the centre
            vel = self.get_x_velocity(self.ball.body.velocity)
            #print( vel, self.ball.body.velocity)
        elif self.get_pos_ball_along_tray() > 0 and self.prev_ball_pos <= 0:
            vel = self.get_x_velocity(self.ball.body.velocity)
            #print( vel, self.ball.body.velocity)
            #print( self.get_pos_ball_along_tray(),  self.prev_ball_pos)
        self.prev_ball_pos = self.get_pos_ball_along_tray()
        #print(self.get_pos_ball_along_tray(), self.prev_ball_pos)
        return vel

    def has_ball_changed_direction(self):
        has_changed = False
        vel = self.get_x_velocity(self.ball.body.velocity)
        if self.was_positive and vel < 0:  # Check if is negative, and has been positive at any point
            has_changed = True
            self.was_negative = True 
        elif vel < 0:  # Register that has been negative
            self.was_negative = True 
        elif self.was_negative and vel > 0:  # Check if is positive, and has been negative at any point
            has_changed = True 
            self.was_positive = True
        elif vel > 0:
            self.was_positive = True
        return has_changed

    def get_x_velocity(self, vel):
        z = np.sqrt(vel[0]*vel[0] + vel[1] * vel[1])
        if vel[0] < 0:
            z = z*-1
        return z

    def get_bin(self, val, max_val, num_val_bins): #CHECK THIS WORKS WITH ALL VALUES
        val_percent = (val+max_val) / (2 * max_val)
        val_bin = math.floor(val_percent * num_val_bins)
        if val_bin < 0:
            val_bin = 0
        elif val_bin >= num_val_bins:
            val_bin = num_val_bins-1
        return val_bin

    def get_bin_info(self, pos_val, pos_max, pos_num, vel_val, vel_max, vel_num, ang_val, ang_max, ang_num):
        pos = self.get_bin(pos_val, pos_max, pos_num)
        vel = self.get_bin(vel_val, vel_max, vel_num)
        ang = self.get_bin(ang_val, ang_max, ang_num)
        return pos, vel, ang


num_bins_ang = 4  # number of angles above and below
num_bins_pos = 10
num_bins_vel = 8
num_actions = num_bins_ang * 2  # First half is clockwise (-1 dir), second half is anticlockwise (1 dir)

max_ang = 0.1
max_pos = 200
max_vel = 600

q_mat = np.zeros((num_bins_pos, num_bins_vel, num_bins_ang, num_actions))


def setup_nn(num_pos=10, num_vel=8, num_ang=4):
    trainer = BallBalancer()
    trainer.set_up_pygame()
    trainer.create_world()

    num_bins_ang = num_ang
    num_bins_pos = num_pos
    num_bins_vel = num_vel
    num_actions = num_bins_ang * 2  # First half is clockwise (-1 dir), second half is anticlockwise (1 dir)

    trainer.reduction_freq = 2000
    trainer.random_action_chance = 1
    trainer.reduction_amount = 1.5

    trainer.length_on_tray = 30
    trainer.discount_factor = 0

    trainer.NUM_ANGLE_BINS = num_bins_ang  # Make it so dont have to do this
    print(num_bins_pos)
    return trainer


training_data = []
training_data_json = []

train_speed = 60


def transfer_data():
    old_data = list(np.load("training_data_nn.npy"))

    # load_json = open("training_data.json", "r")
    # loaded_json = json.load(load_json)
    new_data = []  # loaded_json["data"]

    for i in old_data:
        new_datum = {"pos": int(i[0]), "vel": int(i[1]), "ang": int(i[2]), "action": int(i[3])}
        new_data.append(new_datum)

    json_output = open("training_data.json", 'w')
    json_data = {
                    "data": new_data,
                    "matadata": [
                        {
                            "num_pos": num_bins_pos, 
                            "num_vel": num_bins_vel, 
                            "num_ang": num_bins_ang, 
                            "max_pos": max_pos, 
                            "max_vel": max_vel, 
                            "max_ang": max_ang
                        }
                    ]
    }
    # loaded_json["data"] = new_data
    json.dump(json_data, json_output)


def generate_data_nn(trainer, append_q=True, save_q=False, pos_range=(-max_pos, max_pos), vel_range=(-max_vel, max_vel)):
    i = 0
    running = True
    print(num_bins_pos)

    if append_q:
        # training_data = list(np.load("training_data_nn.npy"))
        load_json = open("training_data.json", "r")
        # print(load_json)
        training_data_json = json.load(load_json)["data"]
        #print(training_data_json)
    else:
        training_data_json = []

    while i < 100 and trainer.continue_running() and running:
        new_ang = (random.randint(0, max_ang*100)/100) * ((-1)**(random.randint(1,2)))  # Random andgle the tray will be for this test
        new_pos = int(random.uniform(*pos_range))  # random.randint(0, max_pos) * ((-1)**(random.randint(1,2)))  # Random position of the ball on the tray for this test
        new_vel = int(random.uniform(*vel_range))  # random.randint(0, max_vel) * ((-1)**(random.randint(1,2)))  # Random velocity of the ball for this test
        # print(new_pos, new_vel)
        if random.random() < 0.5:  # Get it into a smaller range more often, as these values are seen much more
            new_pos = new_pos // 2.0
            new_vel = new_vel // 2.0 

        trainer.update_tray_angle(new_ang)  # Move the tray to the chosen position
        trainer.add_ball(new_pos, new_vel)  # Add the new ball
        trainer.a = trainer.get_bin(new_ang, max_ang, num_bins_ang)
        # print("\n\n\n")
        continue_this_ball = True
        while trainer.is_ball_on_tray() and continue_this_ball:  # See what happens after action has taken place

            bin_ang = trainer.get_bin(new_ang, max_ang, num_bins_ang)
            bin_pos = trainer.get_bin(new_pos, max_pos, num_bins_pos)
            bin_vel = trainer.get_bin(new_vel, max_vel, num_bins_vel)

            not_done_yet = True
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                elif event.type == KEYDOWN and event.key == K_ESCAPE:
                    running = False
                elif event.type == KEYDOWN and event.key == K_DOWN:
                    not_done_yet = False
                    continue_this_ball = False
                elif event.type == KEYDOWN and not_done_yet:
                    p, v, a = trainer.get_ball_info()
                    inputs = trainer.get_bin_info(p, max_pos, num_bins_pos, v, max_vel, num_bins_vel, a, max_ang, num_bins_ang)
                    output = 0
                    if event.key == K_LEFT:
                        trainer.do_action(1, 1, slow=True, draw=True, speed=train_speed)  # Carry out that action
                        output = 1
                    elif event.key == K_RIGHT:
                        trainer.do_action(1, -1, slow=True, draw=True, speed=train_speed)
                        output = -1
                    elif event.key == K_UP:
                        output = 0
                    new_data = {"pos": inputs[0], "vel": inputs[1], "ang": inputs[2], "action": output}  # Format the NN inputs and output into json format
                    training_data_json.append(new_data)  # Add to the data
                    not_done_yet = False
            trainer.step_simulation(True, True, True, train_speed)
        trainer.remove_ball()

        i+=1

        if i % trainer.reduction_freq == 0:
            trainer.random_action_chance /= trainer.reduction_amount
            print(i)
    if save_q:  # Saves the training data in a json format, with the metadata about the parameters used for training
        np.save("training_data_nn", training_data)
        json_output = open("training_data.json", 'w')
        json_data = {
                        "matadata": [
                            {
                                "num_pos": num_bins_pos, 
                                "num_vel": num_bins_vel, 
                                "num_ang": num_bins_ang, 
                                "max_pos": max_pos, 
                                "max_vel": max_vel, 
                                "max_ang": max_ang
                            }
                        ]
                        "data": training_data_json,
        }
        json.dump(json_data, json_output)



#print(training_data)
#NEURAL NETWORK
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.customxml import NetworkWriter
from pybrain.tools.customxml import NetworkReader

TRAIN_NN = False

def train_nn(trainer, structure=(2), momentum=0.99, learningrate=0.001, train_time=200):
    load_json = open("training_data.json", "r")
    training_data = json.load(load_json)["data"]

    struct = (3,) + structure + (1,)

    print(struct)

    net = buildNetwork(*struct, bias=True)  # * converts list tuple into separate parameters

    ds = SupervisedDataSet(3, 1)
    for i in training_data:
        inp = [i["pos"], i["vel"], i["ang"]]
        out = [i["action"]]
        #print(inp, out)
        ds.addSample(inp, out)
    #print(ds)
    train = BackpropTrainer(net, ds, learningrate=learningrate, momentum=momentum)
    train.trainEpochs(train_time)

    NetworkWriter.writeToFile(net, 'trained_nn.xml')


def load_network():
    net = NetworkReader.readFrom('trained_nn.xml')
    return net


#  net - the neural network, can load with load_network()
#  inp - the [position, velocity, angle] bin values
def get_nn_output(net, inp):
    action = net.activate(inp)
    return action


def evaluate_nn(trainer, number_of_trials=10, iteration_limit=200, action_threshold=0.2, pos_range=(-max_pos, max_pos), vel_range=(-max_vel, max_vel), draw_output=True, draw_speed=60):
    net = NetworkReader.readFrom('trained_nn.xml')

    number_completed = 0  # The number of trials in which the ball stays on for the max number of iterations

    i = 0
    running = True
    while i < number_of_trials and trainer.continue_running() and running:
        if i%100 == 0:
            print(i)
        new_ang = (random.randint(0, max_ang*100)/100) * ((-1)**(random.randint(1,2)))  # Random andgle the tray will be for this test
        new_pos = int(random.uniform(*pos_range))  # random.randint(0, max_pos) * ((-1)**(random.randint(1,2)))  # Random position of the ball on the tray for this test
        new_vel = int(random.uniform(*vel_range))  # random.randint(0, max_vel) * ((-1)**(random.randint(1,2)))  # Random velocity of the ball for this test

        if random.random() < 1:  # Get it into a smaller range more often, as these values are seen much more
            new_vel = new_vel // 2.0 

        trainer.update_tray_angle(new_ang)  # Move the tray to the chosen position
        trainer.add_ball(new_pos, new_vel)  # Add the new ball
        trainer.a = trainer.get_bin(new_ang, max_ang, num_bins_ang)
        #print("\n\n\n")
        continue_this_ball = 0
        limit = iteration_limit
        while trainer.is_ball_on_tray() and continue_this_ball < limit and running:  # See what happens after action has taken place
            #bin_ang = trainer.get_bin(new_ang, max_ang, num_bins_ang)
            #bin_pos = trainer.get_bin(new_pos, max_pos, num_bins_pos)
            #bin_vel = trainer.get_bin(new_vel, max_vel, num_bins_vel)
            p, v, a = trainer.get_ball_info()
            p, v, a = trainer.get_bin_info(p, max_pos, num_bins_pos, v, max_vel, num_bins_vel, a, max_ang, num_bins_ang)

            action = net.activate([p, v, a])
            #print(action)
            #print(continue_this_ball, action, p, v, a)
            if abs(action) > action_threshold:
                if action < 0:
                    trainer.do_action(1, -1, slow=True, draw=draw_output, speed=draw_speed)
                elif action > 0:
                    trainer.do_action(1, 1, slow=True, draw=draw_output, speed=draw_speed)
                else:
                    trainer.do_action(1, 0, slow=True, draw=draw_output, speed=draw_speed)
                continue_this_ball += 1
            else:
                trainer.step_simulation(True, True, draw_output, draw_speed)
            
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                elif event.type == KEYDOWN and event.key == K_ESCAPE:
                    running = False
            
            if continue_this_ball == limit:
                number_completed += 1
            #for abc in range(0, 5):
            #    
        trainer.remove_ball()

        i+=1

    print(number_completed/number_of_trials)

        
# if SAVING:
#     np.save("q", q_mat)
# if LOADING:
#     q_mat = np.load("q.npy")

# for i in range(0, 0):
    
#     # Generate random starting position, velocity and angle for ball
#     new_pos = 150 #random.randint(0, max_pos) * ((-1)**(random.randint(1,2)))  # Random position of the ball on the tray for this test
#     new_vel = 0# random.randint(0, max_vel) * ((-1)**(random.randint(1,2)))  # Random velocity of the ball for this test
#     new_ang = (random.randint(0, max_ang*100)/100) * ((-1)**(random.randint(1,2)))  # Random andgle the tray will be for this test

#     trainer.update_tray_angle(new_ang)  # Move the tray to the chosen position
#     trainer.add_ball(new_pos, new_vel)  # Add the new ball
#     print("START", i, new_pos, new_vel, new_ang)
#     while trainer.is_ball_on_tray(trainer.length_on_tray) and trainer.continue_running():
#         trainer.reset_draw()
#         bin_pos = trainer.get_bin(new_pos, max_pos, num_bins_pos)
#         bin_vel = trainer.get_bin(new_vel, max_vel, num_bins_vel)
#         bin_ang = trainer.get_bin(new_ang, max_ang, num_bins_ang)
#         action = np.argmax(q_mat[bin_pos][bin_vel][bin_ang])
        
#         num_bins, direction = trainer.action_to_num_and_dir(action, num_actions)
#         print("\n\npos:", new_pos, bin_pos, "\nvel:", new_vel, bin_vel, "\nang:", new_ang, bin_ang, "\nact:", action, "\nbins:", num_bins, " dir:", direction, "\nq:", q_mat[bin_pos][bin_vel][bin_ang])
#         trainer.do_action(num_bins, direction, draw=True, speed=60)
#         #print(q_mat[bin_pos][bin_vel][bin_ang], action)
#         #for j in range(0, 1):
#         #    trainer.step_simulation(True, True, True, 60)
#         new_pos, new_vel, new_ang = trainer.get_ball_info()
#         #print(i, new_pos, new_vel, new_ang)
#     print("FOT:", trainer.num_frames_off_tray)
#     trainer.remove_ball()





    #TODO:
    # Read why update q formula works like it does
    # Implment training with the Q matrix 

###TRAINING
#Make simulation world
#While training:
#   Randomly generate the angle, speed and position of the ball for this current episode of the simulation
#   Let the simulation happen
#   Generate a reward for how well it did
#   Update Q

###ONCE TRAINED
#Given state of ball, let it do the action
#Once that action is complete, choose the next one given the new state - may have to wait between doing actions