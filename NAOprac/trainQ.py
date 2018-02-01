
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

config = configparser.ConfigParser()
config.read('config.ini')


class BallBalancer:
    def __init__(self):

        self.tray_width = 400
        self.tray_height = 20
        self.tray_x_pos = 300
        self.tray_y_pos = 100
        self.tray_angle = -0.05  # np.pi / 24
        self.rotation = 50000
        self.force_distance = self.tray_width/2

        self.ball_radius = 25

        self.NUM_ACTIONS = 2  # Number of actions that can be taken. Normally 2, rotate clockwise or anticlockwise
        self.NUM_X_DIVS = 8  # Number of divisions x plane is split into, for whole tray
        self.NUM_VELOCITIES = 8  # Number of velocity buckets
        self.NUM_ANGLES = 10  # 400  # int(config["nao_params"]["num_angles"]) #2 * np.pi
        self.MAX_VELOCITY = 200  # math.sqrt(2 * mass/100 * 981/100 * math.sin(MAX_ANGLE) * w)  # Using SUVAT and horizontal component of gravity, /100 because of earlier values seem to be *100

        self.MAX_ANGLE = float(config["nao_params"]["left_angle_max"])
        self.MIN_ANGLE = float(config["nao_params"]["right_angle_max"])

        self.NUM_NAO_ANGLES = (self.__get_state(0, 0, self.MAX_ANGLE)[2]-1) - (self.__get_state(0, 0, self.MIN_ANGLE)[2]+1)
        self.first_ang = self.__get_state(0, 0, self.MIN_ANGLE)[2]+1
        self.last_ang = self.__get_state(0, 0, self.MAX_ANGLE)[2]-1

        self.iterations = 1

        self.Q = np.zeros((self.NUM_X_DIVS, self.NUM_VELOCITIES, self.NUM_ANGLES, self.NUM_ACTIONS))

        self.best_time = 0
        self.time_threshold = 0.7
        self.real_time_threshold = 30
        self.total_time = 0

        self.explore_rate = 1
        self.learn_rate = 0.8
        self.discount_factor = 0.9
        self.scale_reduction = 1.7

        self.max_time = 100
        self.threshold_counter = 0
        self.num_thresholds = 15
        self.max_num_iterations = 1000
        self.max_draw_iterations = 500
        self.explore_reduction_freq = 150

        self.train = True  # If train and update Q
        self.start_time = time.time()

        self.num_dists = 0
        self.tot_dist = 0

        self.step_size = 10  # When stepping simulation after taking action, to allow time to see what ball does

        self.trained = False

        self.run_continuously_with_key_press = True  # Used for testing - whether requires a key input to move to next step of simulation

    def perform_episode(self, prnt=False):
        # GET THE STATES
        self.prev_bin_p, self.prev_bin_v, self.prev_bin_a = self.__get_state(self.prev_val_p, self.prev_val_v, self.prev_val_a)

        # DECIDE ON THE BEST ACTION
        self.curr_action = self.__choose_action(self.prev_bin_p, self.prev_bin_v, self.prev_bin_a)

        # GET THE STATES ONCE UPDATED
        self.curr_bin_p, self.curr_bin_v, self.curr_bin_a = self.__change_state(self.prev_bin_a, self.curr_action)

        # UPDATE Q
        if self.train:
            if self.curr_bin_p >= 0 and self.curr_bin_v >= 0:
                reward = self.__calculate_reward()
                old_val = copy.copy(self.Q[self.prev_bin_p][self.prev_bin_v][self.prev_bin_a])
                self.Q[self.prev_bin_p][self.prev_bin_v][self.prev_bin_a][self.curr_action] += \
                    self.learn_rate * (reward + self.discount_factor * max(self.Q[self.curr_bin_p][self.curr_bin_v][self.curr_bin_a]))
                self.Q = self.__normalised(self.curr_bin_p, self.curr_bin_v, self.curr_bin_a)

                if prnt:
                    print("\n\nOld:", self.prev_bin_p, self.prev_bin_v, self.prev_bin_a, ", New:", self.curr_bin_p, self.curr_bin_v, self.curr_bin_a,
                          "\nAction:", self.curr_action, ", Reward:", reward, ", Extra:", self.learn_rate * (reward + self.discount_factor * max(self.Q[self.curr_bin_p][self.curr_bin_v][self.curr_bin_a])),
                          "\nQ_old:", old_val, ", Q_new:", self.Q[self.prev_bin_p][self.prev_bin_v][self.prev_bin_a], ", Q_future:", self.Q[self.curr_bin_p][self.curr_bin_v][self.curr_bin_a])

    def set_up_pygame(self):
        pygame.init()
        self.screen = pygame.display.set_mode((600, 600))
        self.clock = pygame.time.Clock()

        # Physics stuff
        self.space = pymunk.Space()
        self.space._set_gravity(Vec2d(0, -981))
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)

    def create_world(self):
        fp = [(self.tray_width/2, -self.tray_height/2), (-self.tray_width/2, self.tray_height/2), (self.tray_width/2, self.tray_height/2), (-self.tray_width/2, -self.tray_height/2)]
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

        self.__add_ball(400, 200, 1, self.ball_radius, self.tray_width)

    def check_reset(self):
        reset = False
        if abs(self.curr_val_p) > self.tray_width/2:
            self.__remove_ball(self.ball)
            reset = True
        return reset

    def reset_scenario(self):
        self.trayBody.angle = (random.random() * self.MAX_ANGLE * 2) - self.MAX_ANGLE  # random.randrange(-15, 15, 1)/100
        self.trayBody.angular_velocity = 0
        self.__add_ball(self.tray_x_pos, 150, 1, self.ball_radius, self.tray_width)
        self.start_time = time.time()

    def compare_times(self, curr_time):
        t = curr_time - self.start_time
        if t > self.best_time:
            self.best_time = t
        if t > self.time_threshold and self.threshold_counter < self.num_thresholds:
            self.threshold_counter += 1

    def reduce_explore_rate(self):
        t = time.time() - self.start_time
        self.total_time += t
        if self.iterations % self.explore_reduction_freq == 0:
            self.explore_rate /= self.scale_reduction
            print("Iterations:", self.iterations, "Average:", self.total_time / self.iterations)

    def terminate(self, curr_time):
        terminate = False
        for event in pygame.event.get():
            if event.type == KEYDOWN and event.key == K_ESCAPE:
                terminate = True
        if self.threshold_counter >= self.num_thresholds:
            terminate = True
        if self.iterations >= self.max_num_iterations:
            terminate = True
        if curr_time - self.start_time > self.max_time:
            terminate = True
        return terminate

    def __get_state(self, pos, vel, ang):
        p_bin = math.floor(((pos + (self.tray_width/2)) / (self.tray_width) * self.NUM_X_DIVS))
        v_bin = math.floor(((vel + self.MAX_VELOCITY) / (self.MAX_VELOCITY*2)) * self.NUM_VELOCITIES)
        a_bin = math.floor(((ang + self.MAX_ANGLE) / (2 * self.MAX_ANGLE)) * self.NUM_ANGLES)  # (math.floor(((ang + np.pi) / (2*np.pi)) * self.NUM_ANGLES + self.NUM_ANGLES//4))%self.NUM_ANGLES #starting from right is 0, then increases going anticlockwise
        #In terms of normal angles, when horizontal, angle is 0
        #When going from horizontal and turning anticlockwise angle goes from 0 and increases, with pi at other horizontal
        #When going clockwise, angle goes below 0 and then decreases

        if p_bin > self.NUM_X_DIVS-1:
            p_bin = self.NUM_X_DIVS-1
        elif p_bin < 0:
            p_bin = 0

        if vel >= self.MAX_VELOCITY:
            v_bin = self.NUM_VELOCITIES - 1
        elif vel <= -self.MAX_VELOCITY:
            v_bin = 0

        if a_bin > self.NUM_ANGLES - 1:
            a_bin = self.NUM_ANGLES - 1
        elif a_bin < 0:
            a_bin = 0

        return p_bin, v_bin, a_bin

    def get_state(self, pos, vel, ang):
        p_bin = math.floor(((pos + (self.tray_width/2)) / (self.tray_width) * self.NUM_X_DIVS))
        v_bin = math.floor(((vel + self.MAX_VELOCITY) / (self.MAX_VELOCITY*2)) * self.NUM_VELOCITIES)
        a_bin = math.floor(((ang + self.MAX_ANGLE) / (2 * self.MAX_ANGLE)) * self.NUM_ANGLES)  # (math.floor(((ang + np.pi) / (2*np.pi)) * self.NUM_ANGLES + self.NUM_ANGLES//4)) % self.NUM_ANGLES  # starting from right is 0, then increases going anticlockwise

        if p_bin > self.NUM_X_DIVS-1:
            p_bin = self.NUM_X_DIVS-1
        elif p_bin < 0:
            p_bin = 0
        if abs(vel) >= self.MAX_VELOCITY:
            v_bin = -1
        if a_bin > self.NUM_ANGLES - 1:
            a_bin = self.NUM_ANGLES - 1
        elif a_bin < 0:
            a_bin = 0
        return p_bin, v_bin, a_bin

    def __choose_action(self, p_var, v_var, a_var):
        action = 0
        if p_var >= 0 and v_var >= 0:
            if random.random() < self.explore_rate:
                action = random.randint(0, self.NUM_ACTIONS-1)
            else:
                action = np.argmax(self.Q[p_var][v_var][a_var])
        return action

    # dist is distance from centre to apply the force, a_bin is the bin of the current angle of the tray
    def __change_state(self, a_bin, action):
        turn = True
        while turn:
            if action == 0:
                if self.trayBody.angle > self.__get_centre_of_ang_bin((a_bin-1)) and self.trayBody.angle > self.MIN_ANGLE:
                    self.trayBody.apply_force_at_local_point(Vec2d.unit() * self.rotation, (-self.force_distance, 0))  # rotate flipper clockwise
                else:
                    turn = False
            elif action == 1:
                if self.trayBody.angle < self.__get_centre_of_ang_bin((a_bin+1)) and self.trayBody.angle < self.MAX_ANGLE:
                    self.trayBody.apply_force_at_local_point(Vec2d.unit() * self.rotation, (self.force_distance, 0))  # rotate flipper anticlockwise
                else:
                    turn = False
            elif action == 2:
                turn = False  # Do nothing
            self.__step_simulation()

        self.trayBody.angular_velocity = 0  # To stop turning once turn has occured

        # Carry out any extra turns
        for i in range(0, self.step_size):
            self.__step_simulation()

        ppp, vvv, aaa = self.__get_state(self.curr_val_p, self.curr_val_v, self.curr_val_a)

        return ppp, vvv, aaa

    def __step_simulation(self):
        dt = 1.0/60.0/5.
        for x in range(5):
            self.space.step(dt)

        # Update the real values
        self.prev_val_p = self.curr_val_p
        self.prev_val_v = self.curr_val_v
        self.prev_val_a = self.curr_val_a

        self.curr_val_p = self.__get_ball_pos_x(self.ball)
        self.curr_val_v = self.ball.body.velocity[0]
        self.curr_val_a = self.trayBody.angle
        #print(self.curr_val_a)

    def __calculate_reward(self):
        reward = 0

        if abs(self.curr_val_p) < (self.tray_width/2)/3:
            reward += 0.5
        elif abs(self.curr_val_p) < (self.tray_width)/3:
            reward += 0.2
        else:
            reward -= 0.5

        if abs(self.prev_val_v) > abs(self.curr_val_v):
            reward += 0.5
        else:
            reward -= 0.2

        if abs(self.curr_val_a) < abs(self.MAX_ANGLE) / 10:
            reward += 0.2
            # print(self.curr_val_a)

        return reward

    def __normalised(self, p_var, v_var, a_var):
        """
        Maps all values to between -1 and 1
        """
        q_var = copy.copy(self.Q)
        sumup = 0
        for i in q_var[p_var][v_var][a_var]:
            sumup += abs(i)
        if sumup > 0:
            for i in range(0, len(self.Q[p_var][v_var][a_var])):
                q_var[p_var][v_var][a_var][i] = (self.Q[p_var][v_var][a_var][i]/sumup)
        return q_var

    def __add_ball(self, xpos, ypos, mass, radius, tray_width):
        inertia = pymunk.moment_for_circle(mass, 0, radius, (0, 0))
        body = pymunk.Body(1, inertia)
        body.position = random.randint(xpos - tray_width/4, xpos + tray_width/4), ypos
        shape = pymunk.Circle(body, radius, (0, 0))
        shape.elasticity = 0  # 0.95
        self.space.add(body, shape)

        self.prev_val_p = body.position[0]
        self.prev_val_v = 0
        self.prev_val_a = self.trayBody.angle

        self.curr_val_p = 0
        self.curr_val_v = 0
        self.curr_val_a = 0

        self.ball = shape

    def __remove_ball(self, ball_to_remove):
        self.space.remove(ball_to_remove, ball_to_remove.body)

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

    def __get_centre_of_ang_bin(self, b):
        if b < 0:
            b = 0
        elif b > self.NUM_ANGLES - 1:
            b = self.NUM_ANGLES - 1
        jump = (self.MAX_ANGLE * 2.0) / self.NUM_ANGLES  # Size of a bin
        bin_pos = (b + 0.5) % self.NUM_ANGLES  # The bth bin, and then half way through that
        centre_angle = self.MIN_ANGLE + jump * bin_pos

        return centre_angle

    def reset_draw(self):
        self.screen = pygame.display.set_mode((600, 600))
        self.clock = pygame.time.Clock()
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)

    def draw(self, draw_speed=10):
        self.screen.fill(THECOLORS["white"])
        self.space.debug_draw(self.draw_options)

        self.clock.tick(draw_speed)
        pygame.display.flip()

    def continue_running(self, step_by_step=False):
        running = True
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == KEYDOWN and event.key == K_ESCAPE:
                running = False

        if step_by_step:
            run = True
            while run and self.run_continuously_with_key_press:
                for event in pygame.event.get():
                    if event.type == KEYDOWN and event.key == K_RIGHT:
                        run = False
                    elif event.type == KEYDOWN and event.key == K_ESCAPE:
                        self.run_continuously_with_key_press = False
        return running

    def get_reduced_q(self):
        return self.Q[:, :, self.first_ang: self.last_ang]

    def get_ball_info_bins(self):
        return self.__get_state(self.curr_val_p, self.curr_val_v, self.curr_val_a)

    def get_ball_info(self):
        return self.curr_val_p, self.curr_val_v, self.curr_val_a

    def avg_dist(self):
        self.tot_dist += abs(self.curr_val_p)
        self.num_dists += 1
        return self.tot_dist / self.num_dists

    def reset_avg_dist(self):
        self.tot_dist = 0
        self.num_dists = 0


OBSERVE = False
if OBSERVE:
    training_wait = True
    training_show = True
else:
    training_wait = False
    training_show = False


def setup_q_trainer():
    trainer = BallBalancer()
    trainer.set_up_pygame()
    trainer.create_world()
    trainer.max_draw_iterations = 2000
    if OBSERVE:
        trainer.max_draw_iterations = 0
    trainer.explore_rate = 1
    trainer.explore_reduction_freq = 200
    trainer.scale_reduction = 1.5
    trainer.step_size = 30
    trainer.learn_rate = 0.5
    trainer.discount_factor = 0.99
    return trainer


def do_q_learning(trainer, train=True, load_q=False, save_q=False):
    if load_q:
        trainer.Q = np.load("q_mats/q_learn.npy")
        trainer.explore_rate = 0

    if not train:
        trainer.iterations = trainer.max_draw_iterations+1
        trainer.step_size = 1

    running = True
    current_run = 0
    max_run_len = 200
    while running:
        trainer.perform_episode(prnt=training_show)

        if trainer.iterations == trainer.max_draw_iterations:
            trainer.step_size = 1
            trainer.explore_rate = 0
            trainer.a_high = 0
            trainer.a_low = trainer.NUM_ANGLES
            trainer.reset_draw()
            trainer.trained = True
        if trainer.iterations > trainer.max_draw_iterations:
            trainer.draw(10)
            a, b, c = trainer.get_ball_info_bins()
            print("Q:", a, b, c, trainer.curr_action, trainer.Q[a, b, c], trainer.get_state(trainer.prev_val_p, trainer.prev_val_v, trainer.prev_val_a))##trainer.get_ball_info(), trainer.explore_rate)
        if trainer.check_reset() or current_run > max_run_len:
            current_run = 0
            trainer.compare_times(time.time())
            trainer.reduce_explore_rate()
            trainer.reset_scenario()
            trainer.iterations += 1
            trainer.reset_avg_dist()
        running = trainer.continue_running(training_wait)

    if save_q:
        np.save("q_mats/q_learn", trainer.Q)

#TODO
# - Fix jaggedyness
# - Fix locking to one angle when fully tilted