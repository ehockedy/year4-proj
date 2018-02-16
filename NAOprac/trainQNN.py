import random

import pygame
from pygame.locals import *
from pygame.color import *

import pymunk
from pymunk import Vec2d
import pymunk.pygame_util

import numpy as np
import math
import time

import configparser
import json
import os

from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.customxml import NetworkWriter
from pybrain.tools.customxml import NetworkReader

config = configparser.ConfigParser()
config.read('config.ini')


class BallBalancer:
    def __init__(self):

        self.tray_width = 250
        self.tray_height = 0.001
        self.tray_x_pos = 300
        self.tray_y_pos = 100
        self.tray_angle = -0.05  # np.pi / 24
        self.rotation = 50000

        self.ball_radius = 25

        self.MAX_ANGLE = float(config["nao_params"]["left_angle_max"])
        self.MIN_ANGLE = float(config["nao_params"]["right_angle_max"])

        self.num_bins_pos = 10
        self.num_bins_vel = 8
        self.num_bins_ang = 4

        self.iterations = 0

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

        self.file_location = "nn_training_data/training_data_0_0_0.json"

    def set_up_pygame(self):
        pygame.init()
        self.screen = pygame.display.set_mode((600, 600))
        self.clock = pygame.time.Clock()

        # Physics stuff
        self.space = pymunk.Space()
        self.space._set_gravity(Vec2d(0, -981))
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)

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
        trayJointBody.position = self.trayBody.position
        j = pymunk.PinJoint(self.trayBody, trayJointBody, (0, 0), (0, 0))
        self.space.add(j)

    def update_tray_angle(self, angle):
        self.trayBody.angle = angle
        self.tray_angle = angle

    # direction=-1 is clockwise
    # direction=1 is anticlockwise
    def do_action(self, num_of_angs_to_move, direction, slow=False, record=True, draw=False, speed=60):
        turn = True
        if not abs(direction) == 1:
            direction = 0
            turn = False
            print("DIRECTION VALUE IS WRONG IN do_action")
        target_bin = self.a + direction*num_of_angs_to_move  # The bin of the angle we are aiming for
        target_angle = self.MIN_ANGLE + target_bin * (self.MAX_ANGLE - self.MIN_ANGLE)/self.num_bins_ang  # The actual angle we are aiming for num_bins_ang should be changes, probably to a parameter

        if target_bin > self.num_bins_ang:
            target_bin = self.num_bins_ang
        elif target_bin < 0:
            target_bin = 0
        if target_angle > self.MAX_ANGLE:
            target_angle = self.MAX_ANGLE
        elif target_angle < self.MIN_ANGLE:
            target_angle = self.MIN_ANGLE

        while turn and self.is_ball_on_tray(self.length_on_tray):
            if direction == -1:
                if self.trayBody.angle > target_angle and self.trayBody.angle > self.MIN_ANGLE:  # Keep rotating util we are past the angle we are aiming for
                    self.trayBody.apply_force_at_local_point(Vec2d.unit() * self.rotation, (-self.tray_width/2, 0))  # rotate flipper clockwise
                else:
                    turn = False
                    self.ball.body.velocity[1] = 0
            elif direction == 1:
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
        if ball.body.position.x < self.tray_x_pos:
            pos = pos * -1
        return pos

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
        return val

    def speed_of_ball_at_centre(self):
        vel = None
        if self.get_pos_ball_along_tray() < 0 and self.prev_ball_pos >= 0:  # If the ball goes past the centre
            vel = self.get_x_velocity(self.ball.body.velocity)
        elif self.get_pos_ball_along_tray() > 0 and self.prev_ball_pos <= 0:
            vel = self.get_x_velocity(self.ball.body.velocity)
        self.prev_ball_pos = self.get_pos_ball_along_tray()
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

    def get_bin(self, val, max_val, num_val_bins):
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

    def calculate_reward(self, p, p_max, v, v_max):
        reward = 0

        p_gap = 0
        v_gap = 0
        if p_max % 2 == 0:
            p_gap = 1
        if v_max % 2 == 0:
            v_gap = 1

        if p >= int(p_max/2) - p_gap and p <= int(p_max/2):
            if v >= int(v_max/2) - v_gap and v <= int(v_max/2):
                reward = 1
        else:
            reward = -1

        return reward


max_ang = 0.1
max_pos = 125
max_vel = 600


def setup_nn(num_pos=10, num_vel=8, num_ang=10, max_pos=125, max_vel=600, max_ang=0.1):
    trainer = BallBalancer()
    trainer.set_up_pygame()
    trainer.create_world()

    trainer.num_bins_pos = num_pos
    trainer.num_bins_vel = num_vel
    trainer.num_bins_ang = num_ang

    trainer.max_pos = max_pos
    trainer.max_vel = max_vel
    trainer.max_ang = max_ang

    trainer.length_on_tray = 30
    trainer.discount_factor = 0

    bin_values = str(trainer.num_bins_pos) + "_" + str(trainer.num_bins_vel) + "_" + str(trainer.num_bins_ang)
    trainer.file_location = "nn_training_data/training_data_" + bin_values + ".json"

    return trainer


training_data = []
training_data_json = []

train_speed = 60


def generate_data_nn(trainer, append_q=True, save_q=False, pos_range=(-max_pos, max_pos), vel_range=(-max_vel, max_vel)):
    i = 0
    running = True

    if append_q:  # If data is already existing for the current parameters, update that.
        wd = os.getcwd()
        wd = wd.replace("\\", "/")  # Since slash directions are different
        if os.path.exists(trainer.file_location):
            print("UPDATE EXISTING DATA")
            load_json = open(trainer.file_location, "r")
            training_data_json = json.load(load_json)["data"]
        else:  # Data set does not exist for current parameters
            print("NEW DATA SET")
            training_data_json = []
    else:
        training_data_json = []

    while i < 100 and trainer.continue_running() and running:
        new_ang = (random.randint(0, trainer.max_ang*100)/100) * ((-1)**(random.randint(1,2)))  # Random andgle the tray will be for this test
        new_pos = int(random.uniform(*pos_range))  # random.randint(0, max_pos) * ((-1)**(random.randint(1,2)))  # Random position of the ball on the tray for this test
        new_vel = int(random.uniform(*vel_range))  # random.randint(0, max_vel) * ((-1)**(random.randint(1,2)))  # Random velocity of the ball for this test

        if random.random() < 0.5:  # Get it into a smaller range more often, as these values are seen much more
            new_pos = new_pos // 2.0
            new_vel = new_vel // 2.0

        trainer.update_tray_angle(new_ang)  # Move the tray to the chosen position
        trainer.add_ball(new_pos, new_vel)  # Add the new ball
        trainer.a = trainer.get_bin(new_ang, trainer.max_ang, trainer.num_bins_ang)

        continue_this_ball = True
        while trainer.is_ball_on_tray() and continue_this_ball:  # See what happens after action has taken place
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
                    inputs = trainer.get_bin_info(p, trainer.max_pos, trainer.num_bins_pos, v, trainer.max_vel, trainer.num_bins_vel, a, trainer.max_ang, trainer.num_bins_ang)
                    output = [0, 0, 0]
                    if event.key == K_LEFT:
                        trainer.do_action(1, 1, slow=True, draw=True, speed=train_speed)  # Carry out that action
                        output = [1, 0, 0]
                    elif event.key == K_RIGHT:
                        trainer.do_action(1, -1, slow=True, draw=True, speed=train_speed)
                        output = [0, 0, 1]
                    elif event.key == K_UP:
                        output = [0, 1, 0]
                    print(inputs, output)
                    new_data = {"pos": p, "vel": v, "ang": a, "out1": output[0], "out2": output[1], "out3": output[2]}  # Format the NN inputs and output into json format
                    training_data_json.append(new_data)  # Add to the data
                    not_done_yet = False
            trainer.step_simulation(True, True, True, train_speed)
        trainer.remove_ball()

        i += 1

    if save_q:  # Saves the training data in a json format, with the metadata about the parameters used for training
        json_output = open(trainer.file_location, 'w')
        json_data = {
                        "metadata": [
                            {
                                "num_pos": trainer.num_bins_pos,
                                "num_vel": trainer.num_bins_vel,
                                "num_ang": trainer.num_bins_ang,
                                "max_pos": trainer.max_pos,
                                "max_vel": trainer.max_vel,
                                "max_ang": trainer.max_ang
                            }
                        ],
                        "data": training_data_json,
        }
        json.dump(json_data, json_output)


# NEURAL NETWORK
# trainer is an instance of ball balancer
def train_nn(trainer, structure=(2,), momentum=0.99, learningrate=0.001, train_time=200):
    load_json = json.load(open(trainer.file_location, "r"))
    training_data = load_json["data"]
    metadata = load_json["metadata"][0]  # 0 because is an array

    struct = (3,) + structure + (3,)

    print("TRAINING")

    net = buildNetwork(*struct, bias=True)  # * converts list tuple into separate parameters

    ds = SupervisedDataSet(3, 3)
    for i in training_data:
        in_pos = trainer.get_bin(i["pos"], metadata["max_pos"], metadata["num_pos"])
        in_vel = trainer.get_bin(i["vel"], metadata["max_vel"], metadata["num_vel"])
        in_ang = trainer.get_bin(i["ang"], metadata["max_ang"], metadata["num_ang"])
        inp = [in_pos, in_vel, in_ang]
        out = [i["out1"], i["out2"], i["out3"]]
        ds.addSample(inp, out)

    train = BackpropTrainer(net, ds, learningrate=learningrate, momentum=momentum)
    train.trainEpochs(train_time)

    NetworkWriter.writeToFile(net, 'nn_trained_networks/trained_nn.xml')


def load_network():
    net = NetworkReader.readFrom('nn_trained_networks/trained_nn.xml')
    return net


def load_q_network():
    net = NetworkReader.readFrom('nn_trained_networks/q_learnt_nn.xml')
    return net


#  net - the neural network, can load with load_network()
#  inp - the [position, velocity, angle] bin values
def get_nn_output(net, inp):
    action = net.activate(inp)
    return action


def update_nn(net, inputs, outputs, learningrate=0.001, momentum=0.99):
    ds = SupervisedDataSet(3, 3)
    ds.addSample(list(inputs), outputs)

    train = BackpropTrainer(net, ds, learningrate=learningrate, momentum=momentum)
    train.trainOnDataset(ds)
    return net


def evaluate_nn(trainer, number_of_trials=10, iteration_limit=200, action_threshold=0.2,
                pos_range=(-max_pos, max_pos), vel_range=(-max_vel, max_vel),
                learn_rate=0.4, discount_factor=0.99,
                explore_rate=0.5, explore_rate_reduction=2, explore_rate_freq=50,
                draw_output=True, draw_speed=60, train=True, prnt=False, net=None):

    if net is None:
        net = NetworkReader.readFrom('nn_trained_networks/trained_nn.xml')

    number_completed = 0  # The number of trials in which the ball stays on for the max number of iterations

    print("EVALUATING")

    i = 0
    running = True
    while i < number_of_trials and trainer.continue_running() and running:
        if i % explore_rate_freq == 0:
            print(i)
            explore_rate /= explore_rate_reduction
        new_ang = (random.randint(0, trainer.max_ang*100)/100) * ((-1)**(random.randint(1,2)))  # Random andgle the tray will be for this test
        new_pos = int(random.uniform(*pos_range))  # random.randint(0, max_pos) * ((-1)**(random.randint(1,2)))  # Random position of the ball on the tray for this test
        new_vel = int(random.uniform(*vel_range))  # random.randint(0, max_vel) * ((-1)**(random.randint(1,2)))  # Random velocity of the ball for this test

        if random.random() < 1:  # Get it into a smaller range more often, as these values are seen much more
            new_vel = new_vel // 5.0

        trainer.update_tray_angle(new_ang)  # Move the tray to the chosen position
        trainer.add_ball(new_pos, new_vel)  # Add the new ball
        trainer.a = trainer.get_bin(new_ang, trainer.max_ang, trainer.num_bins_ang)
        # print("\n\n\n")
        continue_this_ball = 0
        limit = iteration_limit
        while trainer.is_ball_on_tray() and continue_this_ball < limit and running:  # See what happens after action has taken place
            # Get initial state s
            p, v, a = trainer.get_ball_info()
            p, v, a = trainer.get_bin_info(p, trainer.max_pos, trainer.num_bins_pos, v, trainer.max_vel, trainer.num_bins_vel, a, trainer.max_ang, trainer.num_bins_ang)

            # Use NN to pick and execute best action
            nn_out = net.activate([p, v, a])
            action = np.argmax(nn_out)

            if train:
                if random.random() < explore_rate:
                    action = random.randint(0, 2)

            # print(p, v, a, nn_out, action)
            if abs(nn_out[action]) < action_threshold:
                trainer.step_simulation(False, True, draw_output, draw_speed)
            elif action == 0:
                trainer.do_action(1, 1, slow=False, draw=draw_output, speed=draw_speed)
            elif action == 2:
                trainer.do_action(1, -1, slow=False, draw=draw_output, speed=draw_speed)
            elif action == 1:
                trainer.step_simulation(False, True, draw_output, draw_speed)
            #    trainer.do_action(1, 0, slow=True, draw=draw_output, speed=draw_speed)

            if not train and prnt:
                print(p, v, a, nn_out, action)

            if train:
                # Get new state s'
                p2, v2, a2 = trainer.get_ball_info()
                p2, v2, a2 = trainer.get_bin_info(p2, trainer.max_pos, trainer.num_bins_pos, v2, trainer.max_vel, trainer.num_bins_vel, a2, trainer.max_ang, trainer.num_bins_ang)

                # Calculate reward for being in s'
                reward = trainer.calculate_reward(p2, trainer.num_bins_pos, v2, trainer.num_bins_vel)

                # Get the value of the best action from the new state
                action2_val = max(net.activate([p2, v2, a2]))

                if prnt:
                    print(p, v, a, "/", action, "/", p2, v2, a2, "/", reward)
                    print(nn_out)

                # Update the output vector values
                nn_out[action] = ((1 - learn_rate) * nn_out[action]) + (learn_rate * (reward + discount_factor*action2_val))

                if prnt:
                    print(nn_out, "\n")

                # update the network
                #print("updating")
                net = update_nn(net, [p, v, a], nn_out)
                #print("finished")

            continue_this_ball += 1

            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                elif event.type == KEYDOWN and event.key == K_ESCAPE:
                    running = False

            if continue_this_ball == limit:
                number_completed += 1

        trainer.remove_ball()

        i += 1

    print(number_completed/number_of_trials)
    if train:
        NetworkWriter.writeToFile(net, 'nn_trained_networks/q_learnt_nn.xml')
