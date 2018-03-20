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
        self.max_ang = float(config["nao_params"]["left_angle_max"])
        self.min_ang = float(config["nao_params"]["right_angle_max"])

        self.file_location = "q_mats/q_" + str(self.num_bins_pos) + "_" + str(self.num_bins_vel) + "_" + str(self.num_bins_ang) + "_" + str(self.num_actions)
        self.file_location_delay = "q_mats/q_DELAY_" + str(self.num_bins_pos) + "_" + str(self.num_bins_vel) + "_" + str(self.num_bins_ang) + "_" + str(self.num_actions)
        self.file_location_er = "q_mats/q_er_" + str(self.num_bins_pos) + "_" + str(self.num_bins_vel) + "_" + str(self.num_bins_ang) + "_" + str(self.num_actions)

        self.iterations = 1

        self.Q = np.zeros((self.num_bins_pos, self.num_bins_vel, self.num_bins_ang, self.num_actions))
        # self.Q.fill(0.5)
        self.q_freq = np.zeros((self.num_bins_pos, self.num_bins_vel, self.num_bins_ang))

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
                if self.q_learn and not self.no_angle:
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
        # reward = exp["reward"]
        old_val = copy.copy(self.Q[p][v][a])

        if not self.no_angle:
            self.Q[p][v][a][action] = \
                ((1-self.learn_rate) * old_val[action]) + self.learn_rate * (reward + self.discount_factor * max(self.Q[p_new][v_new][a_new]))
        else:
            for a_idx in range(0, len(self.Q[p][v])):  # Make the same for all angles - TEMPORARY FIX
                self.Q[p][v][a_idx][action] = \
                   ((1-self.learn_rate) * old_val[action]) + self.learn_rate * (reward + self.discount_factor * max(self.Q[p_new][v_new][a_idx]))

        # if self.prnt:
        #     print("Old:", p, v, a, ", New:", p_new, v_new, a_new,
        #             "\nAction:", action, ", Reward:", reward, ", Extra:", self.learn_rate * (reward + self.discount_factor * max(self.Q[p_new][v_new][a_new])),
        #             "\nQ_old:", old_val, ", Q_new:", self.Q[p][v][a], ", Q_future:", self.Q[p_new][v_new][a_new], "\n\n")
        #done = True

        #return done

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
        data_from_file = np.load("nao_experiences/nao_exp_" + str(self.num_bins_pos) + "_" + str(self.num_bins_vel) + "_" + str(self.num_bins_ang) + "_" + str(self.num_actions) + ".npz")
        self.er_mat = data_from_file["exp"]

    def load_er_mat_no_angle(self, angs):
        """
        Angs is number of angles in original matrix
        """
        data_from_file = np.load("nao_experiences/nao_exp_" + str(self.num_bins_pos) + "_" + str(self.num_bins_vel) + "_" + str(angs) + "_" + str(self.num_actions) + ".npz")
        er_mat = data_from_file["exp"]
        er_mat_no_ang = np.empty((self.num_bins_pos, self.num_bins_vel, 1), dtype=object)  # Need dtype=object
        for p in range(len(er_mat)):
            for v in range(len(er_mat[p])):
                er_mat_no_ang[p][v][0] = []
                for a in range(len(er_mat[p][v][0])):
                    er_mat_no_ang[p][v][0].append(er_mat[p][v][0][a])
        self.er_mat = er_mat_no_ang

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

    def check_reset(self):
        reset = False
        if abs(self.curr_val_p) > self.max_pos:
            self.__remove_ball(self.ball)
            reset = True
        return reset

    def timed_out(self):
        self.__remove_ball(self.ball)

    def reset_scenario(self):
        self.trayBody.angle = (random.random() * self.max_ang * 2) - self.max_ang  # random.randrange(-15, 15, 1)/100
        self.trayBody.angular_velocity = 0
        self.__add_ball()
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
            if self.tray_has_ends:
                print("Iterations:", self.iterations, "Touches freq:", self.num_end_touches / self.iterations, self.explore_rate)
            else:
                print("Iterations:", self.iterations, "Average:", self.total_time / self.iterations, self.explore_rate)

    def terminate(self, curr_time):
        """
        Check whether the current iteration should be terminated
        """
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
        # if p_var >= 0 and v_var >= 0:
        if random.random() < self.explore_rate:
            while action == np.argmax(self.Q[p_var][v_var][a_var]):
                action = random.randint(0, self.num_actions-1)

        # Select action based on the proportion of weigh that action has
        #if self.Q[p_var][v_var][a_var][0] > 0 and self.Q[p_var][v_var][a_var][1] > 0:
        # if random.random() < self.Q[p_var][v_var][a_var][0]:
        #     action = 0
        # else:
        #     action = 1
        #if random.random() < self.explore_rate:
        #    action = random.randint(0, self.num_actions-1)

        return action

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
            
            #elif action == 2:
            else:
                turn = False  # Do nothing
            #    for i in range(0, 5):
            #        self.__step_simulation()
            # else:
            #     p, v, _ = self.get_state(self.prev_val_p, self.prev_val_v, self.prev_val_a)
            #     p2, v2, _ = self.get_state(self.curr_val_p, self.curr_val_v, self.curr_val_a)
            #     if p != p2 or v != v2:
            #         turn = False
            #         if action == 2 and self.prnt:
            #             print("OK", p, p2, v, v2)

            turn_counter += 1
            self.__step_simulation(self.sim_speed)

        self.trayBody.angular_velocity = 0  # To stop turning once turn has occured

        ppp, vvv, aaa = self.get_state(self.curr_val_p, self.curr_val_v, self.curr_val_a)

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

    def __update_altprev_values(self):
        """
        Used to update the previous values used for reward calculation. Is
        done less frequently than current values in order to allow the impact
        of each action to be seen
        Used only in calculation of reward, as need to remember the speed of
        the ball from a few frames ago
        """
        self.alt_prev_val_p = self.curr_val_p
        self.alt_prev_val_v = self.curr_val_v
        self.alt_prev_val_a = self.curr_val_a

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
            # if abs(self.curr_val_p) < (self.tray_width/2)/3:
            #     reward += 0.5
            # elif abs(self.curr_val_p) < (self.tray_width)/3:
            #     reward += 0.2
            # else:
            #     reward -= 0.5

            # if abs(self.prev_val_p) > abs(self.curr_val_p):
            #     reward += 0.5
            # else:
            #     reward -= 0.2
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

    def __normalise(self, p_var, v_var, a_var):
        """
        Maps all values to between -1 and 1
        """
        q_var = copy.copy(self.Q)
        sumup = 0
        for i in q_var[p_var][v_var][a_var]:
            sumup += abs(i)
        # print(q_var[p_var][v_var][a_var])
        if sumup > 0:
            for i in range(0, len(self.Q[p_var][v_var][a_var])):
                q_var[p_var][v_var][a_var][i] = 2 * (abs(self.Q[p_var][v_var][a_var][i])/sumup) - 1
                # print(2 * (abs(self.Q[p_var][v_var][a_var][i])/sumup) - 1)
        # print(q_var[p_var][v_var][a_var], sumup, "\n\n")
        return q_var

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

    def get_ball_info_bins(self):
        return self.get_state(self.curr_val_p, self.curr_val_v, self.curr_val_a)

    def get_ball_info(self):
        return self.curr_val_p, self.curr_val_v, self.curr_val_a

    def avg_dist(self):
        self.tot_dist += abs(self.curr_val_p)
        self.num_dists += 1
        return self.tot_dist / self.num_dists

    def reset_avg_dist(self):
        self.tot_dist = 0
        self.num_dists = 0

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

    def load_q_npy(self):
        self.Q = np.load("q_mats/q_learn.npy")

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
                        "max_ang": self.max_ang
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


def setup_q_trainer_old(p, v, a, ends=False, is_q_not_s=True):
    trainer = BallBalancer(p, v, a, ends, is_q_not_s)
    trainer.set_up_pygame()
    trainer.create_world()
    trainer.max_draw_iterations = 25000
    if OBSERVE:
        trainer.max_draw_iterations = 0
    trainer.explore_rate = 0.5
    trainer.explore_reduction_freq = 5000
    trainer.scale_reduction = 1.2
    trainer.step_size = 1
    trainer.learn_rate = 0.5
    trainer.discount_factor = 0.99
    trainer.sim_speed = 5
    return trainer


def setup_trainer(num_states_p, num_states_v, num_states_a,
                  num_iterations, exp_rate=0.5, num_exp_reductions=10, val_exp_reduction=1.2,
                  step_size=1, learn_rate=0.4, disc_fact=0.99, sim_speed=5):
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
    return trainer


def do_q_learning(trainer, train=True, prnt=False):
    if not train:
        trainer.iterations = trainer.max_draw_iterations
        trainer.explore_rate = 0

    running = True
    current_run = 0
    max_run_len = 200
    while running:
        trainer.perform_episode(prnt=False)
        if prnt:
            trainer.prnt = True

        if trainer.iterations == trainer.max_draw_iterations:
            trainer.explore_rate = 0
            trainer.reset_draw()
            trainer.prnt = True

        if trainer.check_reset():
            current_run = 0
            trainer.compare_times(time.time())
            trainer.reduce_explore_rate()
            trainer.reset_scenario()
            trainer.iterations += 1
            trainer.reset_avg_dist()
        if current_run > max_run_len:
            trainer.timed_out()
            current_run = 0
            trainer.compare_times(time.time())
            trainer.reduce_explore_rate()
            trainer.reset_scenario()
            trainer.iterations += 1
        running = trainer.continue_running()

        current_run += 1


def do_q_learning_sides(trainer, train=True, prnt=False, iters=100000, no_ang=False):
    trainer.no_angle = no_ang
    trainer.iterations = 1
    trainer.max_num_iterations = iters
    trainer.max_draw_iterations = int(trainer.max_num_iterations*0.95)
    trainer.explore_reduction_freq = iters/10
    if not train:
        trainer.iterations = trainer.max_draw_iterations
        trainer.explore_rate = 0

    running = True
    trainer.prnt = prnt
    while trainer.iterations < trainer.max_num_iterations and running:
        trainer.perform_episode(prnt)

        if trainer.iterations == trainer.max_draw_iterations:
            trainer.explore_rate = 0
            trainer.reset_draw()
            trainer.prnt = True

        trainer.reduce_explore_rate()

        running = trainer.continue_running()
        trainer.iterations += 1
        trainer.touched_in_this_iteration = False


def do_experience_replay(trainer, iter, show=False, prnt=False):
    # trainer.load_er_mat_no_angle(10)
    trainer.load_er_mat()
    trainer.learn_rate = 0.4
    trainer.no_angle = True
    for i in range(0, iter):
        if i % 1000 == 0:
            print(i)
        trainer.perform_episode_er(prnt)  # , no_ang=True)

    if show:  # Show the balancer
        do_q_learning_sides(trainer, False, True)


def display_simulation(trainer):
    """
    Runs the simulation and siaplays what goes on. No training is done
    """
    trainer.explore_rate = 0  # Because we want to do the decision thought best by the Q matrix
    trainer.reset_draw()  # Refreshes the drawing canvas. Can error if not.
    trainer.prnt = True  # Print some information to the console 

    running = True  # Whether to continue running the sinulation
    while running:
        running = trainer.continue_running()  # Check if it is to continue. Usually halted by a key press
        trainer.perform_episode()  # Carry out the next step of the simulation


def train_q_specific(trainer):
    """
    
    """
    trainer.specific = True
    running = True
    while trainer.iterations < trainer.max_num_iterations and running:
        trainer.perform_episode()  # Carry out the next step of the simulation
        trainer.reduce_explore_rate()  # Check if the explore rate is to be reduced, and do so if it is
        running = trainer.continue_running()  # Check whether to continue running the simulation
        trainer.iterations += 1  # increase the number of iterations by 1
        trainer.touched_in_this_iteration = False  # Reset the fact that the ball is not touching a wall


def train_q_general(trainer):
    """
    Do Q-learning training with a general reward.
    A general reward means there is a good set of states, determined by position
    and velocity of the ball. Receives a positive reward if ball is in these states.
    Receives a negative reward if not
    """
    running = True
    while trainer.iterations < trainer.max_num_iterations and running:
        trainer.perform_episode()  # Carry out the next step of the simulation
        trainer.reduce_explore_rate()  # Check if the explore rate is to be reduced, and do so if it is
        running = trainer.continue_running()  # Check whether to continue running the simulation
        trainer.iterations += 1  # increase the number of iterations by 1
        trainer.touched_in_this_iteration = False  # Reset the fact that the ball is not touching a wall


def train_q_general_robot(trainer):
    """
    Do Q-learning training with a general reward, but the simulation is 
    closer to mimicing the robot. This is done by introducing a delay
    between picking an action and executing it, as well as increasing the
    speed that the simulation runs at to mimic the increased speed of 
    the ball in the robot
    """
    running = True
    while trainer.iterations < trainer.max_num_iterations and running:
        trainer.perform_episode()  # Carry out the next step of the simulation
        trainer.reduce_explore_rate()  # Check if the explore rate is to be reduced, and do so if it is
        running = trainer.continue_running()  # Check whether to continue running the simulation
        trainer.iterations += 1  # increase the number of iterations by 1
        trainer.touched_in_this_iteration = False  # Reset the fact that the ball is not touching a wall


def train_q_experience_replay(trainer):
    trainer.load_er_mat()  # Load the ER matrix from file. This stores all the experiences in a matrix that can be indexed by pos, vel and ang
    for i in range(1, trainer.max_num_iterations):
        if i % 1000 == 0:
            print(i)
        trainer.perform_episode_er()  # , no_ang=True)


# General reward QL
# 100000 iterations works very well, basically perfect
# tr = setup_trainer(12, 12, 10, 100000)
# train_q_general(tr)
# display_simulation(tr)

# Genral reward robot like QL
# 100000 iterations, steps_size and sim_speed 10 works very well
# tr = setup_trainer(12, 12, 10, 100000, step_size=10, sim_speed=10)
# train_q_general(tr)
# display_simulation(tr)

# Expereince replay with general reward
# 5000000 iters, ss and ss 10 works okish. Quite similar to the robot. Better without sss
# tr = setup_trainer(12, 12, 10, 5000000)#, step_size=10, sim_speed=10)
# train_q_experience_replay(tr)
# display_simulation(tr)

# Specific reward QL
# 100000 iterations works well. Is very reactive to changing direction, but not so good at gettin into middle
# Very interestingly, with step_size and sim_speed at 10, side toughing freq initially goes up, but eventually comes back down and it finishes doing ok 
# tr = setup_trainer(12, 12, 10, 100000)  # , step_size=10, sim_speed=10)
# train_q_specific(tr)
# display_simulation(tr)