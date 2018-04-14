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
import os
import json

from sklearn.model_selection import ParameterGrid

import configparser
config = configparser.ConfigParser()
config.read('config.ini')


class BallBalancer:
    def __init__(self, p_bins=8, v_bins=8, a_bins=10, ends=True, is_q_not_s=True):

        self.tray_width = 250
        self.tray_height = 20
        self.tray_x_pos = 300
        self.tray_y_pos = 100
        self.tray_angle = -0.05  # np.pi / 24
        self.rotation = 50000
        self.force_distance = self.tray_width/2
        self.ball_radius = 15
        self.ball_mass = 1

        self.num_actions = 2  # Number of actions that can be taken. Normally 2, rotate clockwise or anticlockwise

        self.num_bins_pos = p_bins  # Number of divisions x plane is split into, for whole tray
        self.num_bins_vel = v_bins  # Number of velocity buckets
        self.num_bins_ang = a_bins

        self.max_pos = self.tray_width/2
        self.max_vel = 200
        self.max_ang = 0.1  # float(config["nao_params"]["left_angle_max"])
        self.min_ang = -0.1  # float(config["nao_params"]["right_angle_max"])

        self.q_mats_path = config["trained_models_paths"]["q_mats"]
        self.file_location = self.q_mats_path+"/q_" + str(self.num_bins_pos) + "_" + str(self.num_bins_vel) + "_" + str(self.num_bins_ang) + "_" + str(self.num_actions)
        self.file_location_delay = self.q_mats_path+"/q_DELAY_" + str(self.num_bins_pos) + "_" + str(self.num_bins_vel) + "_" + str(self.num_bins_ang) + "_" + str(self.num_actions)
        self.file_location_er = self.q_mats_path+"/q_er_" + str(self.num_bins_pos) + "_" + str(self.num_bins_vel) + "_" + str(self.num_bins_ang) + "_" + str(self.num_actions)

        self.iterations = 1

        self.Q = np.zeros((self.num_bins_pos, self.num_bins_vel, self.num_bins_ang, self.num_actions))
        # self.Q.fill(0.5)
        self.q_freq = np.zeros((self.num_bins_pos, self.num_bins_vel, self.num_bins_ang))
        self.data_records = []  # Stores the data about each step of the simulation. For use when displaying results
        self.record_cells = False

        self.curr_action = 0
        self.next_action = 0

        self.reward = 0

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
        self.sim_speed = 10  # Speed to step the simulation by. Not draw speed

        self.q_learn = is_q_not_s  # Whether to do q_learning, not SARSA
        self.trained = False
        self.prnt = False
        self.specific = False  # Whether to use a very specific reward, or a more general, simple one
        self.er = False  # Whether experience replay is being done
        self.consensus_choice = False
        self.proportional_choice = False
        self.prev_iteration_num = 1  # Used for recording q_mat cell data

        self.tray_has_ends = ends  # Whether to add ends to the tray
        self.num_end_touches = 0
        self.touched_in_this_iteration = False
        self.no_angle = False

        self.run_continuously_with_key_press = True  # Used for testing - whether requires a key input to move to next step of simulation

    def perform_episode(self):
        """
        Given the current state of the ball, pick an action, carry out
        that action, then update Q based on the state it transitions to
        """

        # Get the states
        self.prev_bin_p, self.prev_bin_v, self.prev_bin_a = self.get_state(self.prev_val_p, self.prev_val_v, self.prev_val_a)

        # Choose best action
        self.curr_action = self.__choose_action(self.prev_bin_p, self.prev_bin_v, self.prev_bin_a)
        if not self.q_learn:  # For SARSA
            self.curr_action = self.next_action

        # Get the new, updated states
        self.curr_bin_p, self.curr_bin_v, self.curr_bin_a = self.__change_state(self.prev_bin_a, self.curr_action)

        if not self.q_learn:  # Get the next action for updating SARSA
            self.next_action = self.__choose_action(self.curr_bin_p, self.curr_bin_v, self.curr_bin_a)

        # Update Q matrix
        if not self.trained:
            if self.curr_bin_p >= 0 and self.curr_bin_v >= 0:  # and (self.curr_bin_p, self.curr_bin_v, self.curr_bin_a) != (self.prev_bin_p, self.prev_bin_v, self.prev_bin_a):
                reward = self.__calculate_reward()
                self.reward = reward
                old_val = copy.copy(self.Q[self.prev_bin_p][self.prev_bin_v][self.prev_bin_a])
                if self.q_learn:
                    self.Q[self.prev_bin_p][self.prev_bin_v][self.prev_bin_a][self.curr_action] = \
                        ((1-self.learn_rate) * old_val[self.curr_action]) + self.learn_rate * (reward + self.discount_factor * max(self.Q[self.curr_bin_p][self.curr_bin_v][self.curr_bin_a]))
                elif self.no_angle:
                    for i in range(0, len(self.Q[self.prev_bin_p][self.prev_bin_v])):
                        self.Q[self.prev_bin_p][self.prev_bin_v][i][self.curr_action] = \
                            ((1-self.learn_rate) * old_val[self.curr_action]) + self.learn_rate * (reward + self.discount_factor * max(self.Q[self.curr_bin_p][self.curr_bin_v][i]))
                else:  # Use SARSA
                    self.Q[self.prev_bin_p][self.prev_bin_v][self.prev_bin_a][self.curr_action] = \
                        ((1-self.learn_rate) * old_val[self.curr_action]) + self.learn_rate * (reward + self.discount_factor * self.Q[self.curr_bin_p][self.curr_bin_v][self.curr_bin_a][self.next_action])

                if self.prnt:
                    print("\n\nOld:", self.prev_bin_p, self.prev_bin_v, self.prev_bin_a, ", New:", self.curr_bin_p, self.curr_bin_v, self.curr_bin_a,
                          "\nAction:", self.curr_action, ", Reward:", reward, ", Extra:", self.learn_rate * (reward + self.discount_factor * max(self.Q[self.curr_bin_p][self.curr_bin_v][self.curr_bin_a])),
                          "\nQ_old:", old_val, ", Q_new:", self.Q[self.prev_bin_p][self.prev_bin_v][self.prev_bin_a], ", Q_future:", self.Q[self.curr_bin_p][self.curr_bin_v][self.curr_bin_a],
                          "\n\n")
            self.q_freq[self.prev_bin_p][self.prev_bin_v][self.prev_bin_a] += 1

    def perform_episode_er(self, prnt=False):
        """
        Carry out an action using information from a action carried out by the real nao
        Returns true if successfully executed
        """

        num_angs = self.num_bins_ang-1
        if self.no_angle:
            num_angs = 0

        p = random.randint(0, self.num_bins_pos-1)
        v = random.randint(0, self.num_bins_vel-1)
        a = random.randint(0, num_angs)
        experience = self.er_mat[p][v][a][0]
        while len(experience) == 0:
            #print(p, v, a)

            p = random.randint(0, self.num_bins_pos-1)
            v = random.randint(0, self.num_bins_vel-1)
            a = random.randint(0, num_angs)
            experience = self.er_mat[p][v][a][0]

        exp = random.choice(experience)
        action = exp["action"]
        new_state = exp["new_state"]
        p_new, v_new, a_new = new_state

        # reward = exp["reward"]
        self.curr_bin_p = p_new
        self.curr_bin_v = v_new
        reward = self.__calculate_reward()
        #reward = exp["reward"]
        old_val = copy.copy(self.Q[p][v][a])

        if not self.no_angle:
            self.Q[p][v][a][action] = \
                ((1-self.learn_rate) * old_val[action]) + self.learn_rate * (reward + self.discount_factor * max(self.Q[p_new][v_new][a_new]))
        else:
            for a_idx in range(0, len(self.Q[p][v])):  # Make the same for all angles - TEMPORARY FIX
                self.Q[p][v][a_idx][action] = \
                   ((1-self.learn_rate) * old_val[action]) + self.learn_rate * (reward + self.discount_factor * max(self.Q[p_new][v_new][a_idx]))

        #if p == 6 and v == 6 and a == 5:
        #    print("Old:", p, v, a, ", New:", p_new, v_new, a_new,
        #            "\nAction:", action, ", Reward:", reward, " Old:", old_val[action], ", Extra:", self.learn_rate * (reward + self.discount_factor * max(self.Q[p_new][v_new][a_new])),
        #            "\nQ_old:", old_val, ", Q_new:", self.Q[p][v][a], ", Q_future:", self.Q[p_new][v_new][a_new], "\n\n")

    def update_state_from_er(self, new_state):
        """
        Moves the ball to a specific state. Does not let the simulation do it itself
        """
        extra = self.step_size
        # Carry out any extra turns
        for i in range(0, extra):
            self.__step_simulation(self.sim_speed)
            self.trayBody.angular_velocity = 0  # Dont want tray to move under balls weight when not moving
        p, v, a = self.__get_vals_from_state(new_state)
        self.ball.body.position = Vec2d(self.tray_x_pos + p, self.ball.body.position.y)  # CHANGE THIS SO THAT DOES ACTUAL DIST ALONG THE TRAY, NOT DIRECT
        #self.ball.body.update_position(self.ball.body, Vec2d(-300, 0))
        self.ball.body.velocity = Vec2d(v, 0)
        self.trayBody.angle = a
        self.__step_simulation(self.sim_speed)
        self.trayBody.angular_velocity = 0  # Dont want tray to move under balls weight when not moving

    def __get_vals_from_state(self, state):
        """
        Given the state bin values, gives the values from the center of each bin
        Maybe randomize within the bin
        """
        multiplier_p = (2 * state[0] - (self.num_bins_pos-1)) / self.num_bins_pos
        multiplier_v = (2 * state[1] - (self.num_bins_vel-1)) / self.num_bins_vel
        multiplier_a = (2 * state[2] - (self.num_bins_ang-1)) / self.num_bins_ang

        val_p = multiplier_p * self.max_pos
        val_v = multiplier_v * self.max_vel
        val_a = multiplier_a * self.max_ang

        return val_p, val_v, val_a

    def load_er_mat(self):
        data_from_file = np.load(config["other"]["q_experiences"] + "/nao_exp_" + str(self.num_bins_pos) + "_" + str(self.num_bins_vel) + "_" + str(self.num_bins_ang) + "_" + str(self.num_actions) + ".npz")
        self.er_mat = data_from_file["exp"]

    def set_up_pygame(self):
        pygame.init()
        self.screen = pygame.display.set_mode((600, 600))
        self.clock = pygame.time.Clock()

        # Physics stuff
        self.space = pymunk.Space()
        self.space._set_gravity(Vec2d(0, -981))
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)

    def create_world(self):
        fp = [(self.tray_width/2, -self.tray_height/2), (-self.tray_width/2, self.tray_height/2), (self.tray_width/2, self.tray_height/2), (-self.tray_width/2, -self.tray_height/2), (-self.tray_width/2, -self.tray_height/2)]

        mass = 100
        moment = pymunk.moment_for_poly(mass, fp[0:2])

        self.trayBody = pymunk.Body(mass, moment)
        self.trayBody.position = self.tray_x_pos, self.tray_y_pos
        self.trayBody.angle = self.tray_angle
        trayShape = pymunk.Poly(self.trayBody, fp)
        if self.tray_has_ends:
            side1 = [(self.tray_width/2, self.tray_height/2), (self.tray_width/2, self.tray_height*4), (self.tray_width/2-1, self.tray_height*4), (self.tray_width/2-1, self.tray_height/2)]
            side2 = [(-self.tray_width/2, self.tray_height/2), (-self.tray_width/2, self.tray_height*4), (-self.tray_width/2+1, self.tray_height*4), (-self.tray_width/2+1, -self.tray_height/2)]
            self.side1_shape = pymunk.Poly(self.trayBody, side1)
            self.side2_shape = pymunk.Poly(self.trayBody, side2)
            self.space.add(self.side1_shape, self.side2_shape)
        self.space.add(self.trayBody, trayShape)

        trayJointBody = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        trayJointBody.position = self.trayBody.position
        j = pymunk.PinJoint(self.trayBody, trayJointBody, (0, 0), (0, 0))
        self.space.add(j)

        self.__add_ball()

    def reset_scenario(self):
        self.trayBody.angle = (random.random() * self.max_ang * 2) - self.max_ang  # random.randrange(-15, 15, 1)/100
        self.trayBody.angular_velocity = 0
        self.__add_ball()
        self.start_time = time.time()

    def reduce_explore_rate(self):
        t = time.time() - self.start_time
        self.total_time += t
        if self.iterations % self.explore_reduction_freq == 0:
            self.explore_rate /= self.scale_reduction
            if self.tray_has_ends:
                print("Iterations:", self.iterations, "Touches freq:", self.num_end_touches / self.iterations, self.explore_rate)
            else:
                print("Iterations:", self.iterations, "Average:", self.total_time / self.iterations, self.explore_rate)

    def get_state(self, pos, vel, ang):
        """
        Returns the bin value given the actual value.
        Bins come from splitting up the range of possible values into discrete
        sections, such that the method works for any variation in range of
        possible distance/velocity/angle values
        """
        p_bin = math.floor(((pos + (self.max_pos)) / (self.max_pos*2) * self.num_bins_pos))
        v_bin = math.floor(((vel + self.max_vel) / (self.max_vel*2)) * self.num_bins_vel)
        a_bin = math.floor(((ang + self.max_ang) / (2 * self.max_ang)) * self.num_bins_ang)  # starting from right is 0, then increases going anticlockwise
        # In terms of normal angles, when horizontal, angle is 0
        # When going from horizontal and turning anticlockwise angle goes from 0 and increases, with pi at other horizontal
        # When going clockwise, angle goes below 0 and then decreases

        if p_bin > self.num_bins_pos-1:
            p_bin = self.num_bins_pos-1
        elif p_bin < 0:
            p_bin = 0

        if vel >= self.max_vel:
            v_bin = self.num_bins_vel - 1
        elif vel <= -self.max_vel:
            v_bin = 0

        if a_bin > self.num_bins_ang - 1:
            a_bin = self.num_bins_ang - 1
        elif a_bin < 0:
            a_bin = 0

        return p_bin, v_bin, a_bin

    def __choose_action(self, p_var, v_var, a_var):
        action = np.argmax(self.Q[p_var][v_var][a_var])

        if self.consensus_choice:
            votes = [action]
            if p_var > 0:
                votes.append(np.argmax(self.Q[p_var-1][v_var][a_var]))
            if p_var < self.num_bins_pos-1:
                votes.append(np.argmax(self.Q[p_var+1][v_var][a_var]))
            if v_var > 0:
                votes.append(np.argmax(self.Q[p_var][v_var-1][a_var]))
            if v_var < self.num_bins_vel-1:
                votes.append(np.argmax(self.Q[p_var][v_var+1][a_var]))
            action = max(set(votes), key=votes.count)
            #if self.iterations >= self.max_num_iterations:
            #    print(votes, action)

        if self.proportional_choice:
            act1 = self.Q[p_var][v_var][a_var][0] 
            act2 = self.Q[p_var][v_var][a_var][1] 
            if (act1 > 0 and act2 > 0) or (act1 < 0 and act2 < 0):  # If they are either both positive reward of both negative reward - i.e. their scores are both good or both bad, then choose proportionally
                if random.random() < abs(act1) / abs(act1 + act2):
                    action = 0
                else:
                    action = 1

        # if p_var >= 0 and v_var >= 0:
        if random.random() < self.explore_rate:
            while action == np.argmax(self.Q[p_var][v_var][a_var]):
                action = random.randint(0, self.num_actions-1)

        # Select action based on the proportion of weight that action has
        #if self.Q[p_var][v_var][a_var][0] > 0 and self.Q[p_var][v_var][a_var][1] > 0:
        # if random.random() < self.Q[p_var][v_var][a_var][0]:
        #     action = 0
        # else:
        #     action = 1
        #if random.random() < self.explore_rate:
        #    action = random.randint(0, self.num_actions-1)

        return int(action)  # int needed for json serialising

    def __change_state(self, a_bin, action):
        """
        Go to the next angle from the current angle, based on the action
        specified
        """
        extra = self.step_size
        # Carry out any extra turns
        for i in range(0, extra):
            self.__step_simulation(self.sim_speed)
            self.trayBody.angular_velocity = 0  # Dont want tray to move under balls weight when not moving
        turn = True
        turn_counter = 0
        while turn:
            if action == 0 and a_bin > 0:
                if self.trayBody.angle > self.__get_centre_of_ang_bin((a_bin-1)) and self.trayBody.angle > self.min_ang:
                    self.trayBody.apply_force_at_local_point(Vec2d.unit() * self.rotation, (-self.force_distance, 0))  # rotate flipper clockwise
                else:
                    turn = False
            elif action == 1 and a_bin < self.num_bins_ang-1:
                if self.trayBody.angle < self.__get_centre_of_ang_bin((a_bin+1)) and self.trayBody.angle < self.max_ang:
                    self.trayBody.apply_force_at_local_point(Vec2d.unit() * self.rotation, (self.force_distance, 0))  # rotate flipper anticlockwise
                else:
                    turn = False
            else:
                turn = False  # Do nothing

            turn_counter += 1
            self.__step_simulation(self.sim_speed)

        self.trayBody.angular_velocity = 0  # To stop turning once turn has occured

        ppp, vvv, aaa = self.get_state(self.curr_val_p, self.curr_val_v, self.curr_val_a)

        if self.trayBody.angle < self.min_ang:
            self.trayBody.angle = self.min_ang
        elif self.trayBody.angle > self.max_ang:
            self.trayBody.angle = self.max_ang

        return ppp, vvv, aaa

    def __step_simulation(self, speed=5):
        """
        Move the simulation on, and update the values of the ball
        """
        dt = 1.0/60.0/5.
        for x in range(speed):
            self.space.step(dt)

        # Update the real values
        self.prev_val_p = copy.copy(self.curr_val_p)
        self.prev_val_v = copy.copy(self.curr_val_v)
        self.prev_val_a = copy.copy(self.curr_val_a)

        self.curr_val_p = self.__get_ball_pos_x(self.ball)
        self.curr_val_v = self.ball.body.velocity[0]
        self.curr_val_a = self.trayBody.angle

        if self.prnt:
            self.draw(60)

        if self.is_in_collision_with_end() and not self.touched_in_this_iteration:
            self.num_end_touches += 1
            self.touched_in_this_iteration = True

    def __calculate_reward(self):
        """
        Generate the reward, given the current state of the ball and how
        it has changed
        """
        reward = 0

        gap_left = 0
        if self.num_bins_pos % 2 == 0:
            gap_left = 1
        gap = 1
        gap2 = 1

        if self.specific:  # A ver specific reward, refined after tinking what the best thing to do is in each situation
            if (self.curr_val_v > 0 and self.curr_bin_a > self.prev_bin_a) or (self.curr_val_v < 0 and self.curr_bin_a < self.prev_bin_a):
                reward += abs(self.tray_x_pos - self.curr_val_p)/self.tray_width
            elif (self.curr_val_v > 0 and abs(self.curr_val_v) < self.max_vel*0.01 and self.curr_val_p > self.tray_x_pos) or (self.curr_val_v < 0 and abs(self.curr_val_v) < self.max_vel*0.01 and self.curr_val_p < self.tray_x_pos):
                reward += (abs(self.tray_x_pos - self.curr_val_p)/self.tray_width)
            else:
                reward += -1 * (abs(self.tray_x_pos - self.curr_val_p)/self.tray_width)
        else:  # The more general reward, a single area
            if self.curr_bin_p >= int((self.num_bins_pos)/2) - gap - gap_left and self.curr_bin_p <= int((self.num_bins_pos)/2) + gap:
                if self.curr_bin_v >= int((self.num_bins_vel)/2) - gap2 - gap_left and self.curr_bin_v <= int((self.num_bins_vel)/2) + gap2:
                    reward = 1
                else:
                    reward = -1
            else:
                reward = -1

        return reward

    def __add_ball(self):
        inertia = pymunk.moment_for_circle(self.ball_mass, 0, self.ball_radius, (0, 0))
        body = pymunk.Body(1, inertia)
        extra = int(self.max_pos / 4)
        body.position = random.randint(self.tray_x_pos - self.max_pos + extra, self.tray_x_pos + self.max_pos - extra), 150  # Y as 150 is arbitrary, jut makes sure is above tray
        shape = pymunk.Circle(body, self.ball_radius, (0, 0))
        shape.elasticity = 0  # 0.95
        self.space.add(body, shape)

        self.prev_val_p = body.position[0]
        self.prev_val_v = random.random() * (self.max_vel/4) * (-1 ** random.randint(1, 2))
        self.prev_val_a = self.trayBody.angle

        self.curr_val_p = 0  # body.position[0]
        self.curr_val_v = 0  # random.random() * (self.max_vel/2) * (-1 ** random.randint(1, 2))
        self.curr_val_a = 0  # self.trayBody.angle

        self.alt_prev_val_p = 0
        self.alt_prev_val_v = 0
        self.alt_prev_val_a = 0

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
        elif b > self.num_bins_ang - 1:
            b = self.num_bins_ang - 1
        jump = (self.max_ang * 2.0) / self.num_bins_ang  # Size of a bin
        bin_pos = (b + 0.5)  # % self.num_bins_ang  # The bth bin, and then half way through that
        centre_angle = self.min_ang + jump * bin_pos
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

    def is_in_collision_with_end(self):
        collide = False
        if self.tray_has_ends:
            if len(self.side1_shape.shapes_collide(self.ball).points) > 0:
                collide = True
            elif len(self.side2_shape.shapes_collide(self.ball).points) > 0:
                collide = True
        return collide

    def load_q(self, delay=False):
        """
        Loads the Q matrix and relevant attributes from a binary numpy file
        """
        data_from_file = None
        if delay:
            data_from_file = np.load(self.file_location_delay + ".npz")
        else:
            data_from_file = np.load(self.file_location + ".npz")
        self.Q = data_from_file["q"]
        #self.Q = normalise_whole_q(self.Q)
        metadata = data_from_file["metadata"].item()
        self.num_bins_pos = metadata["num_pos"]
        self.num_bins_vel = metadata["num_vel"]
        self.num_bins_ang = metadata["num_ang"]
        self.num_actions = metadata["num_actions"]
        self.max_pos = metadata["max_pos"]
        self.max_vel = metadata["max_vel"]
        self.max_ang = metadata["max_ang"]
        self.min_ang = -metadata["max_ang"]

    def save_q(self, delay=False, er=False):
        """
        Saves the Q matrix as a zipped NumPy binary file
        Also includes an array called "metadata" with information on the Q
        matrix
        """
        metadata = {
                        "num_pos": self.num_bins_pos,
                        "num_vel": self.num_bins_vel,
                        "num_ang": self.num_bins_ang,
                        "num_actions": self.num_actions,
                        "max_pos": self.max_pos,
                        "max_vel": self.max_vel,
                        "max_ang": self.max_ang,
                        "specific_reward": self.specific,
                        "step_size": self.step_size,
                        "sim_speed": self.sim_speed
                    }

        q = self.Q
        if delay:
            np.savez(self.file_location_delay, q=q, metadata=metadata)
            print("Saved as:", self.file_location_delay)
        elif er:
            np.savez(self.file_location_er, q=q, metadata=metadata)
            print("Saved as:", self.file_location_er)
        else:
            np.savez(self.file_location, q=q, metadata=metadata)
            print("Saved as:", self.file_location)

    def record_current_state(self):
        state_data = {
            # "iterations": self.iterations,
            "pos": self.curr_val_p,
            "vel": self.curr_val_v,
            "ang": self.curr_val_a,
            "act": self.curr_action,
            # "pos_bin": self.curr_bin_p,  # Can work these out from actual value and some metadata
            # "vel_bin": self.curr_bin_v,
            # "ang_bin": self.curr_bin_a,
            "frames_on_edge": self.num_end_touches / self.iterations
        }
        self.data_records.append(state_data)

    def record_current_q_cells(self):
        """
        Records the data on how Q matrix cells change over time
        """
        store = {}
        if len(self.cell_data_store) > 0:
            prev_data_store = self.cell_data_store[self.prev_iteration_num]  # Get the most recent cell store
            update = False  # Whether to store the data for this iteration
            for c in range(0, len(self.cell_data)):  # For each cell we are recording data about
                cell = self.cell_data[str(c+1)]  # Get the cell state info
                p = cell["p"]
                v = cell["v"]
                a = cell["a"]
                store[c+1] = {
                                0 : copy.copy(self.Q[p][v][a][0]), 
                                1 : copy.copy(self.Q[p][v][a][1])
                            }
                if not np.array_equal(prev_data_store[c+1], store[c+1]):
                    update = True  # Only record the new store if one of the cells is different - save storing reundant data
            
            if update:  # Add the data to the list of stored bits of data
                self.cell_data_store[self.iterations] = store
                self.prev_iteration_num = copy.copy(self.iterations)
                #print(self.iterations, store)

    def generate_cell_data(self):
        """
        Generates the data about the Q matrix cells to record
        This is where you choose which cells to watch
        """
        self.cell_data_store = {}
        self.cell_data =  {
                                "1":
                                    {
                                        'p': 11,
                                        'v': 6,
                                        'a': 0
                                    },
                                "2":
                                    {
                                        'p': 6,
                                        'v': 6,
                                        'a': 5
                                    },
                                "3":
                                    {
                                        'p': 0,
                                        'v': 6,
                                        'a': 8
                                    },
                                "4":
                                    {
                                        'p': 2,
                                        'v': 3,
                                        'a': 3
                                    },
                                "5":
                                    {
                                        'p': 0,
                                        'v': 5,
                                        'a': 7
                                    }
                            }
        store = {}
        for c in range(0, len(self.cell_data)):
            cell = self.cell_data[str(c+1)]
            p = cell["p"]
            v = cell["v"]
            a = cell["a"]
            store[c+1] = {
                            0 : copy.copy(self.Q[p][v][a][0]), 
                            1 : copy.copy(self.Q[p][v][a][1])
                        }
        self.cell_data_store[self.iterations] = store
        self.record_cells = True

    def save_q_cell_data(self, desc=""):
        cwd = os.getcwd() + "/" + config["evaluation_data_paths"]["sim_q"]  # "\..\sim_data"  # Directory of simulation data - do this so matches up with the graphs
        dirs = os.listdir(cwd)  # List of files in that directory
        number_files = len(dirs)-1  # Number of files. -1 because the new position data has already been plotted
        file_name = config["evaluation_data_paths"]["q_cell"] + "/" + config["data_file_prefix"]["q_cell"] + "_" + str(number_files) + ".json"  # ..\q_cell_data
        json_output = open(file_name, 'w')
        json_data = {
                        "metadata": [
                            {
                                "description": desc,
                                "specific": self.specific,
                                "exp_replay": self.er,
                                "num_pos": self.num_bins_pos,
                                "num_vel": self.num_bins_vel,
                                "num_ang": self.num_bins_ang,
                                "max_pos": self.max_pos,
                                "max_vel": self.max_vel,
                                "max_ang": self.max_ang,
                                "num_iterations": self.max_num_iterations,
                                "exp_rate": self.explore_rate_original,
                                "exp_reduction_freq": self.explore_reduction_freq_original,
                                "exp_reduction_scale": self.scale_reduction,
                                "step_size": self.step_size,
                                "sim_speed": self.sim_speed,
                                "learn_rate": self.learn_rate,
                                "discount_factor": self.discount_factor,
                                "final_side_touches": self.num_end_touches / self.iterations
                            }
                        ],
                        "cell_data": self.cell_data,
                        "data": self.cell_data_store,
        }
        #print(self.cell_data, "\n\n", self.cell_data_store)
        json.dump(json_data, json_output)

    def save_state_data(self, desc=""):
        cwd = os.getcwd() + "/" + config["evaluation_data_paths"]["sim_q"]  # Directory of where data is to be saved
        print("Saving data to:", cwd)
        dirs = os.listdir(cwd)  # List of files in that directory
        number_files = len(dirs)  # Number of files
        file_name = config["evaluation_data_paths"]["sim_q"] + "//" + config["data_file_prefix"]["sim_q"] + "_" + str(number_files) + ".json"
        json_output = open(file_name, 'w')
        json_data = {
                        "metadata": [
                            {
                                "description": desc,
                                "specific": self.specific,
                                "exp_replay": self.er,
                                "num_pos": self.num_bins_pos,
                                "num_vel": self.num_bins_vel,
                                "num_ang": self.num_bins_ang,
                                "max_pos": self.max_pos,
                                "max_vel": self.max_vel,
                                "max_ang": self.max_ang,
                                "num_iterations": self.num_iterations_original,
                                "exp_rate": self.explore_rate_original,
                                "exp_reduction_freq": self.explore_reduction_freq_original,
                                "exp_reduction_scale": self.scale_reduction,
                                "step_size": self.step_size,
                                "sim_speed": self.sim_speed,
                                "learn_rate": self.learn_rate,
                                "discount_factor": self.discount_factor,
                                "final_side_touches": self.num_end_touches / self.iterations
                            }
                        ],
                        "data": self.data_records,
        }
        json.dump(json_data, json_output)


def setup_trainer(num_states_p, num_states_v, num_states_a,
                  num_iterations, exp_rate=0.5, num_exp_reductions=20, val_exp_reduction=1.5,
                  step_size=1, learn_rate=0.4, disc_fact=0.9, sim_speed=5):
    trainer = BallBalancer(num_states_p, num_states_v, num_states_a)  # Initialise the trainer object
    trainer.set_up_pygame()  # Set up pygame
    trainer.create_world()  # Create the world - the tray and joint
    trainer.max_num_iterations = num_iterations  # Set how long the simulation will run for
    trainer.explore_rate = exp_rate  # The initial probability that a different action to currently thought optimal will be taken
    trainer.explore_reduction_freq = num_iterations / num_exp_reductions  # Calculate how often the explore rate will drop
    trainer.scale_reduction = val_exp_reduction  # Set the amount to decrease the explore rate by
    trainer.step_size = step_size  # The number of simulation steps to take before an action is executed. Used to simulate the delay in the robot
    trainer.learn_rate = learn_rate  # The amount that the reward and future state affects the current states value
    trainer.discount_factor = disc_fact  # The proportion of the future state value that you keep. Bigger means you value future states more, smaller means only care about immediate states 
    trainer.sim_speed = sim_speed  # The amount to step the simulation by. 5 is default for pygame. Bigger makes simulation run faster.
    trainer.explore_rate_original = exp_rate
    trainer.num_iterations_original = num_iterations
    trainer.explore_reduction_freq_original = num_iterations / num_exp_reductions
    return trainer



def display_simulation(trainer):
    """
    Runs the simulation and displays what goes on. No training is done
    """
    trainer.explore_rate = 0  # Because we want to do the decision thought best by the Q matrix
    trainer.reset_draw()  # Refreshes the drawing canvas. Can error if not.
    trainer.prnt = True  # Print some information to the console 
    trainer.trained = True
    running = True  # Whether to continue running the sinulation
    while running:
        running = trainer.continue_running()  # Check if it is to continue. Usually halted by a key press
        trainer.perform_episode()  # Carry out the next step of the simulation
        p = trainer.curr_bin_p
        v = trainer.curr_bin_v
        a = trainer.curr_bin_a
        #print(p, v, a, trainer.Q[p][v][a])


def train_q(trainer, specific=False, er=False):
    """
    Do Q-learning training with a general reward.
    A general reward means there is a good set of states, determined by position
    and velocity of the ball. Receives a positive reward if ball is in these states.
    Receives a negative reward if not
    If specific is True, then use a more specific reward refined after tinking what
    the best thing to do is in each situation
    If er is True, then train from the experiences recorded from the robot
    """
    trainer.specific = specific
    trainer.er = er
    if er:
        trainer.load_er_mat()  # Load the ER matrix from file. This holds all the experiences in a matrix that can be indexed by pos, vel and ang
        for i in range(1, trainer.max_num_iterations):
            trainer.iterations = i
            if i % 5000 == 0:
                print(i)
            trainer.perform_episode_er()
            trainer.record_current_state()
            if trainer.record_cells:
                trainer.record_current_q_cells()
    else:
        running = True
        while trainer.iterations < trainer.max_num_iterations and running:
            trainer.perform_episode()  # Carry out the next step of the simulation
            trainer.reduce_explore_rate()  # Check if the explore rate is to be reduced, and do so if it is
            running = trainer.continue_running()  # Check whether to continue running the simulation
            trainer.touched_in_this_iteration = False  # Reset the fact that the ball is not touching a wall
            trainer.record_current_state()
            trainer.iterations += 1  # increase the number of iterations by 1
            #if trainer.curr_bin_p == 10 and trainer.curr_bin_v == 5:# and trainer.curr_bin_a == 4:
            #    print(trainer.Q[10][5][trainer.curr_bin_a], trainer.curr_bin_a)


def run_after_trained(tr):
    er_train_steps = copy.copy(tr.max_num_iterations)
    tr.trained = True
    tr.explore_rate = 0
    tr.iterations = 1
    tr.max_num_iterations = 5000  # Do 5000 more - MAYBE SEE WHAT HAPPENS FOR EVEN LONGER
    train_q(tr, er=False)
    tr.max_num_iterations = er_train_steps  # Change for displaying data

# General reward QL
# 100000 iterations works very well, basically perfect
# tr = setup_trainer(12, 12, 10, 100000)
# train_q(tr)
# display_simulation(tr)
# tr.save_state_data()

# Genral reward robot like QL
# 100000 iterations, steps_size and sim_speed 10 works very well
# tr = setup_trainer(12, 12, 10, 100000, step_size=10, sim_speed=10)
# train_q(tr)
# display_simulation(tr)

# Expereince replay with general reward
# 5000000 iters, ss and ss 10 works okish. Quite similar to the robot. Better without sss
# tr = setup_trainer(12, 12, 10, 5000000)#, step_size=10, sim_speed=10)
# train_q(tr, er=True)
# display_simulation(tr)

# Specific reward QL
# 100000 iterations works well. Is very reactive to changing direction, but not so good at gettin into middle
# Very interestingly, with step_size and sim_speed at 10, side toughing freq initially goes up, but eventually comes back down and it finishes doing ok 
# tr = setup_trainer(12, 12, 10, 100000)  # , step_size=10, sim_speed=10)
# train_q(tr, specific=True)
# display_simulation(tr)


df = [0.99, 0.9, 0.5]
er = [0.5, 0.2]

param_grid = {
    #"initial_er":[0.5, 0.2, 0.1], 
    #"num_er":[5, 10, 20],
    #"er_reduction_val":[1.2, 1.5, 2],
    #"lr": [0.4, 0.5, 0.1],
    #"df": [0.9, 0.8, 0.5]
    "not_testing_parameters":[0]
    #"specific": [True, False]
    }
pg = list(ParameterGrid(param_grid))
if False:
    for i in pg:
        print(i)
        tr = setup_trainer(12, 12, 10, 50000)#, step_size=10, sim_speed=10)#, learn_rate=i["lr"], disc_fact=i["df"])

        #tr.explore_rate = i["initial_er"]
        #tr.scale_reduction = i["er_reduction_val"]
        #tr.consensus_choice = False
        #train_q(tr, er=False, specific=True)

        tr.iterations = 1
        er_train_steps = 1000000
        tr.max_num_iterations = er_train_steps
        
        tr.generate_cell_data()
        train_q(tr, er=True, specific=True)
    
        #tr.proportional_choice = True
        tr.trained = True
        tr.explore_rate = 0
        tr.iterations = 1
        tr.max_num_iterations = 5000  # Do 5000 more - MAYBE SEE WHAT HAPPENS FOR EVEN LONGER
        train_q(tr, er=False)
        
        tr.max_num_iterations = er_train_steps  # Change for displaying data
        tr.save_state_data("A trial experience replay" + str(i))
        tr.save_q_cell_data("TEST")
        display_simulation(tr)
        #plot_change_of_q_cells()





# tr = setup_trainer(12, 12, 10, 100000)
# train_q(tr)
# display_simulation(tr)
# tr.save_state_data("General reward, no delay")

# tr = setup_trainer(12, 12, 10, 50000, step_size=10, sim_speed=10)
# train_q(tr)
# display_simulation(tr)
# tr.save_state_data("General reward with delay")

# tr = setup_trainer(12, 12, 10, 5000000)
# train_q(tr, er=True)
# display_simulation(tr)
# tr.save_state_data("ER with no delay")

# tr = setup_trainer(12, 12, 10, 5000000, step_size=10, sim_speed=10)
# train_q(tr, er=True)
# display_simulation(tr)
# tr.save_state_data("ER with delay")

# tr = setup_trainer(12, 12, 10, 100000)
# train_q(tr, specific=True)
# display_simulation(tr)
# tr.save_state_data("Specific reward with no delay")

# tr = setup_trainer(12, 12, 10, 100000, step_size=10, sim_speed=10)
# train_q(tr, specific=True)
# display_simulation(tr)
# tr.save_state_data("Specific reward with delay")
