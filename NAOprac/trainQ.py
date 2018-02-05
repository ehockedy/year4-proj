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
    def __init__(self, p_bins=8, v_bins=8, a_bins=10):

        self.tray_width = 250
        self.tray_height = 20
        self.tray_x_pos = 300
        self.tray_y_pos = 100
        self.tray_angle = -0.05  # np.pi / 24
        self.rotation = 50000
        self.force_distance = self.tray_width/2
        self.ball_radius = 25
        self.ball_mass = 1

        self.num_actions = 2  # Number of actions that can be taken. Normally 2, rotate clockwise or anticlockwise

        self.num_bins_pos = p_bins  # Number of divisions x plane is split into, for whole tray
        self.num_bins_vel = v_bins  # Number of velocity buckets
        self.num_bins_ang = a_bins

        self.max_pos = self.tray_width/2
        self.max_vel = 200
        self.max_ang = float(config["nao_params"]["left_angle_max"])
        self.min_ang = float(config["nao_params"]["right_angle_max"])

        self.file_location = "q_mats/q_" + str(self.num_bins_pos) + "_" + str(self.num_bins_vel) + "_" + str(self.num_bins_ang)

        self.iterations = 1

        self.Q = np.zeros((self.num_bins_pos, self.num_bins_vel, self.num_bins_ang, self.num_actions))
        self.Q.fill(0.5)
        self.q_freq = np.zeros((self.num_bins_pos, self.num_bins_vel, self.num_bins_ang))

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

        self.trained = False
        self.prnt = False

        self.run_continuously_with_key_press = True  # Used for testing - whether requires a key input to move to next step of simulation

    def perform_episode(self, prnt=False):
        """
        Given the current state of the ball, pick an action, carry out
        that action, then update Q based on the state it transitions to
        """

        # Get the states
        self.prev_bin_p, self.prev_bin_v, self.prev_bin_a = self.get_state(self.prev_val_p, self.prev_val_v, self.prev_val_a)

        # Choose best action
        self.curr_action = self.__choose_action(self.prev_bin_p, self.prev_bin_v, self.prev_bin_a)

        # Remember the ball values before taking any action
        #self.__update_altprev_values()

        # Get the new, updated states
        self.curr_bin_p, self.curr_bin_v, self.curr_bin_a = self.__change_state(self.prev_bin_a, self.curr_action)

        #if self.curr_bin_a == 0: #and self.curr_bin_p == 3 and self.curr_bin_v == 5:
        #    print(self.curr_bin_p, self.curr_bin_v)

        # Update Q matrix
        if not self.trained:
            if self.curr_bin_p >= 0 and self.curr_bin_v >= 0:
                reward = self.__calculate_reward()
                self.reward = reward
                old_val = copy.copy(self.Q[self.prev_bin_p][self.prev_bin_v][self.prev_bin_a])
                self.Q[self.prev_bin_p][self.prev_bin_v][self.prev_bin_a][self.curr_action] = \
                    ((1-self.learn_rate) * old_val[self.curr_action]) + self.learn_rate * (reward + self.discount_factor * max(self.Q[self.curr_bin_p][self.curr_bin_v][self.curr_bin_a]))
                self.Q = self.__normalise(self.prev_bin_p, self.prev_bin_v, self.prev_bin_a)

                if prnt:
                    print("\n\nOld:", self.prev_bin_p, self.prev_bin_v, self.prev_bin_a, ", New:", self.curr_bin_p, self.curr_bin_v, self.curr_bin_a,
                          "\nAction:", self.curr_action, ", Reward:", reward, ", Extra:", self.learn_rate * (reward + self.discount_factor * max(self.Q[self.curr_bin_p][self.curr_bin_v][self.curr_bin_a])),
                          "\nQ_old:", old_val, ", Q_new:", self.Q[self.prev_bin_p][self.prev_bin_v][self.prev_bin_a], ", Q_future:", self.Q[self.curr_bin_p][self.curr_bin_v][self.curr_bin_a])
            self.q_freq[self.prev_bin_p][self.prev_bin_v][self.prev_bin_a] += 1
        #if self.curr_bin_a == 0 and self.curr_bin_p == 4 and self.curr_bin_a == 5:
        #    print(reward, self.curr_bin_p, self.curr_bin_v, self.curr_bin_a, "  ", self.prev_bin_p, self.prev_bin_v, self.prev_bin_a)

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
        action = random.randint(0, self.num_actions-1)
        # if p_var >= 0 and v_var >= 0:
        if random.random() > self.explore_rate:
            action = np.argmax(self.Q[p_var][v_var][a_var])
        #         action = random.randint(0, self.num_actions-1)
        #     else:
        #         action = np.argmax(self.Q[p_var][v_var][a_var])

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
            self.__step_simulation()

        self.trayBody.angular_velocity = 0  # To stop turning once turn has occured

        #if turn_counter < 2:  # This stops the slow down when at max angle and not changing state
        #    extra = 10
        #    print("EXTRA", self.iterations)

        # Carry out any extra turns
        #for i in range(0, extra):
        #   self.__step_simulation()

        ppp, vvv, aaa = self.get_state(self.curr_val_p, self.curr_val_v, self.curr_val_a)

        return ppp, vvv, aaa

    def __step_simulation(self):
        """
        Move the  simulation on, and update the values of the ball
        """
        dt = 1.0/60.0/5.
        for x in range(5):
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

        gap = 0
        if abs(self.curr_bin_p) >= int((self.num_bins_pos)/2) - gap and abs(self.curr_bin_p) <= int((self.num_bins_pos)/2) + gap:
            reward += 1
            #print(self.curr_bin_p, int((self.num_bins_pos)/2))

        if abs(self.curr_bin_v) >= int((self.num_bins_vel)/2) - gap and abs(self.curr_bin_v) <= int((self.num_bins_vel)/2) + gap:
            reward += 0.5

        if abs(self.curr_val_p) > self.max_pos:
            reward = -1
            #print(self.curr_bin_v, int((self.num_bins_vel)/2))

        #if abs(self.curr_bin_a) == int((self.num_bins_ang)/2):
        #    reward += 0.1

        #print(reward)
        #else:
        #    reward = -1


        # reward = 0.5
        # if self.curr_val_a > 0:
        #     if self.curr_val_v < 0:
        #         if self.curr_val_p < self.max_pos/2:
        #             reward = -0.5
        # elif self.curr_val_a < 0:
        #     if self.curr_val_v > 0:
        #         if self.curr_val_p > -self.max_pos/2:
        #             reward = -0.5

        return reward

    def __normalise(self, p_var, v_var, a_var):
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

    def __normalise_one_zero(self, p_var, v_var, a_var):
        """
        Maps all values to between 0 and 1
        """
        q_var = copy.copy(self.Q)
        sumup = 0
        for i in q_var[p_var][v_var][a_var]:
            sumup += abs(i)
        if sumup > 0:
            for i in range(0, len(self.Q[p_var][v_var][a_var])):
                q_var[p_var][v_var][a_var][i] = (self.Q[p_var][v_var][a_var][i]/sumup)
        return q_var

    def __add_ball(self):
        inertia = pymunk.moment_for_circle(self.ball_mass, 0, self.ball_radius, (0, 0))
        body = pymunk.Body(1, inertia)
        body.position = random.randint(self.tray_x_pos - self.max_pos, self.tray_x_pos + self.max_pos), 150  # Y as 150 is arbitrary, jut makes sure is above tray
        shape = pymunk.Circle(body, self.ball_radius, (0, 0))
        shape.elasticity = 0  # 0.95
        self.space.add(body, shape)

        self.prev_val_p = body.position[0]
        self.prev_val_v = random.random() * (self.max_vel/2) * (-1 ** random.randint(1, 2))
        self.prev_val_a = self.trayBody.angle

        self.curr_val_p = 0
        self.curr_val_v = 0
        self.curr_val_a = 0

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

    def load_q(self):
        """
        Loads the Q matrix and relevant attributes from a binary numpy file
        """
        data_from_file = np.load(self.file_location + ".npz")
        self.Q = data_from_file["q"]
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

    def save_q(self):
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
        np.savez(self.file_location, q=q, metadata=metadata)


OBSERVE = False
if OBSERVE:
    training_wait = True
    training_show = True
else:
    training_wait = False
    training_show = False


def setup_q_trainer(p, v, a):
    trainer = BallBalancer(p, v, a)
    trainer.set_up_pygame()
    trainer.create_world()
    trainer.max_draw_iterations = 20000
    if OBSERVE:
        trainer.max_draw_iterations = 0
    trainer.explore_rate = 0.25
    trainer.explore_reduction_freq = 2500
    trainer.scale_reduction = 1.3
    trainer.step_size = 1
    trainer.learn_rate = 0.6
    trainer.discount_factor = 0.99
    return trainer


def do_q_learning(trainer, train=True):
    if not train:
        trainer.iterations = trainer.max_draw_iterations+1
        #trainer.step_size = 1
        trainer.explore_rate = 0

    running = True
    current_run = 0
    max_run_len = 200
    while running:
        trainer.perform_episode(prnt=False)

        if trainer.iterations == trainer.max_draw_iterations:
            trainer.step_size = 1
            trainer.explore_rate = 0
            trainer.reset_draw()
            trainer.prnt = True
            #trainer.trained = True
        if trainer.iterations > trainer.max_draw_iterations:
            #trainer.draw(10)
            a, b, c = trainer.get_ball_info_bins()
            #if c == 0 or c == 11:
            print("Q:", a, b, c, ",", trainer.curr_action, trainer.reward, ",", str(round(trainer.curr_val_p,2)), str(round(trainer.curr_val_v, 2)), str(round(trainer.curr_val_a, 2)), trainer.Q[a, b, c])#, trainer.get_state(trainer.prev_val_p, trainer.prev_val_v, trainer.prev_val_a))##trainer.get_ball_info(), trainer.explore_rate)

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
        running = trainer.continue_running(training_wait)

        current_run += 1

    # for i in range(0, trainer.num_bins_pos):
    #     for j in range(0, trainer.num_bins_vel):
    #         for k in range(0, trainer.num_bins_ang):
    #             print(i, j, k, ":", trainer.q_freq[i][j][k])


#TODO
# - Fix jaggedyness (not so much of an issue)
# - Fix locking to one angle when fully tilted
# - Make more specific rewards
# - Fix parameters
# - Decide how to integrate learning with robot
# - In calculate reward, it is looking at the previous state for reward,
#   not the one that is 40 or so ago, so doesnt have that long time to
#   see how it has really affected it
