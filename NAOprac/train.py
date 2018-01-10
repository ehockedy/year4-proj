
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

#Q-learning

reset = False
addBall = False


def remove_ball(ball_to_remove, space):
    space.remove(ball_to_remove, ball_to_remove.body)


def get_ball_pos_x(ball, tray_centre_x, tray_centre_y, ball_radius, tray_height):
    pos = -1
    dist_squared = (ball.body.position.x - tray_centre_x)**2 + (ball.body.position.y - tray_centre_y)**2 - (ball_radius+tray_height/2)**2
    if(dist_squared < 0):
        pos = 0
    else:
        pos = math.sqrt(dist_squared)
    if ball.body.position.x < x_pos:
        pos = pos * -1
    return pos


def get_q_idx_pos(pos, tray_width, num_x_divs):  # Get the indices of the Q matrix based on mapping the position and velocity to the buckets
    pos = pos + tray_width/2  # Map from relative to centre of tray to relative from start of tray
    bins = []
    bin_count = 0
    while bin_count < tray_width-1:
        bins.append(bin_count)
        bin_count += round(tray_width/num_x_divs)
    x = [pos]
    pos_bin = np.digitize(x, bins)
    return pos_bin[0]-1


def get_q_idx_vel(vel, max_vel, num_vels):
    vel = vel + max_vel  # Map from relative to centre of tray to relative from start of tray
    bins = []
    bin_count = 0
    while bin_count < max_vel*2-1:
        bins.append(bin_count)
        bin_count += round((max_vel*2)/num_vels)
    x = [vel]
    vel_bin = np.digitize(x, bins)
    return vel_bin[0]-1  # -1 because starts at 0


def get_state(pos, vel, ang, max_vel, num_vels, num_x_divs, num_angs):
    p_bin = math.floor(((pos + (w/2)) / (w)) * num_x_divs)
    
    v_bin = math.floor(((vel + max_vel) / (max_vel*2)) * num_vels)
    a_bin = (math.floor(((ang + np.pi) / (2*np.pi)) * num_angs + num_angs//4))%num_angs #starting from right is 0, then increases going anticlockwise
    #In terms of normal angles, when horizontal, angle is 0
    #When going from horizontal and turning anticlockwise angle goes from 0 and increases, with pi at other horizontal
    #When going clockwise, angle goes below 0 and then decreases
    if abs(pos) >= w/2:
        p_bin = -1
    if abs(vel) >= max_vel:
        v_bin = -1 
    return p_bin, v_bin, a_bin


def get_centre_of_ang_bin(b, num_angs):
    jump = 2*np.pi / num_angs  # Size of a bin
    bin_pos = (b + 0.5) % num_angs  # The bth bin, and then half way through that
    centre_angle = (jump * bin_pos - np.pi - np.pi/2)  # -pi since tray angles go from -pi to +pi, and want 0 to 2pi
    return centre_angle


def add_ball(xpos, ypos, mass, radius, tray_width, space):
    inertia = pymunk.moment_for_circle(mass, 0, radius, (0,0))
    body = pymunk.Body(1, inertia)
    body.position = random.randint(xpos - tray_width/4, xpos + tray_width/4), ypos
    shape = pymunk.Circle(body, radius, (0,0))
    shape.elasticity = 0#0.95
    space.add(body, shape)
    shape.current_velocity = 0
    shape.current_position = 0#get_ball_pos_x(shape)
    shape.current_action = 0
    shape.current_angle = 0
    shape.previous_action = 0
    shape.previous_velocity = 0
    shape.previous_position = 0
    shape.previous_angle = 0
    return shape


def choose_action(p_var, v_var, a_var, q_var, exp_var, num_act):
    action = 0
    if p_var >= 0 and v_var >= 0:
        if random.random() < exp_var:
            action = random.randint(0, num_act-1)
        else:
            action = np.argmax(q_var[p_var][v_var][a_var])
    return action


def change_state(ball_var, tray, strength, dist, a_bin, sp, max_vel, num_vels, num_x_divs, num_angs):
    action = ball_var.current_action
    if action == 0:
        while tray.angle > get_centre_of_ang_bin((a_bin-1)):
            tray.apply_force_at_local_point(Vec2d.unit() * strength, (-dist, 0))  # rotate flipper clockwise
            dt = 1.0/60.0/5.
            for x in range(5):
                sp.step(dt)
            ball_var.current_position = get_ball_pos_x(ball_var)
            ball_var.current_velocity = ball_var.body.velocity[0]
            ball_var.current_angle = tray.angle
            #p_new, v_new, a_new = get_state(ball_var.current_position, ball_var.current_velocity, ball_var.current_angle)
    elif action == 1:
        while tray.angle < get_centre_of_ang_bin((a_bin+1)):
            tray.apply_force_at_local_point(Vec2d.unit() * strength, (dist, 0))  # rotate flipper anticlockwise
            dt = 1.0/60.0/5.
            for x in range(5):
                sp.step(dt)
            ball_var.current_position = get_ball_pos_x(ball_var)
            ball_var.current_velocity = ball_var.body.velocity[0]
            ball_var.current_angle = tray.angle
    ppp, vvv, aaa = get_state(ball_var.current_position, ball_var.current_velocity, ball_var.current_angle, max_vel, num_vels, num_x_divs, num_angs)
    tray.angular_velocity = 0
    return ppp, vvv, aaa


def calculate_reward(ball_var, a_bin, a_bin_new):
    reward = 0
    if ball_var.current_velocity > 0 and a_bin_new > a_bin:
        reward += 1
    if ball_var.current_velocity < 0 and a_bin_new < a_bin:
        reward += 1
    return reward


def normalisedOLD(Q, q_var, p_var, v_var, a_var):
    if sum(Q[p_var][v_var][a_var]) > 0:
        for i in range(0, len(Q[p_var][v_var][a_var])):
            q_var[p_var][v_var][a_var][i] = Q[p_var][v_var][a_var][i]/sum(Q[p_var][v_var][a_var])
    return q_var


def next_params(var_name, var_pos, defaults, params):
    res = copy.copy(defaults)
    if var_pos >= 0:
        res[var_name] = params[var_name][var_pos]
    return res


class BallBalancer:
    def __init__(self):
        
        self.tray_width = 400
        self.tray_height = 0.001
        self.tray_x_pos = 300
        self.tray_y_pos = 100
        self.tray_angle = -0.05#np.pi / 24
        self.rotation = 80000
        
        self.ball_radius = 25

        self.NUM_ACTIONS = 2  # Number of actions that can be taken. Normally 2, rotate clockwise or anticlockwise
        self.NUM_X_DIVS = 2  # Number of divisions x plane is split into, for whole tray
        self.NUM_VELOCITIES = 49  # Number of velocity buckets
        self.NUM_ANGLES = int(config["nao_params"]["num_angles"]) #2 * np.pi
        self.MAX_VELOCITY = 400#math.sqrt(2 * mass/100 * 981/100 * math.sin(MAX_ANGLE) * w)  # Using SUVAT and horizontal component of gravity, /100 because of earlier values seem to be *100

        self.MAX_ANGLE = float(config["nao_params"]["left_angle_max"])
        self.MIN_ANGLE = float(config["nao_params"]["right_angle_max"])
        print(self.MIN_ANGLE, self.MAX_ANGLE)

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

    def perform_episode(self):
        # GET THE STATES
        p, v, a = self.__get_state(self.ball.previous_position, self.ball.previous_velocity, self.ball.previous_angle)
        # DECIDE ON THE BEST ACTION
        self.ball.current_action = self.__choose_action(p, v, a)

        # GET THE STATES ONCE UPDATED
        p_new, v_new, a_new = self.__change_state(self.tray_width/2, a)
        self.a = a_new
        # UPDATE Q
        #print(v, v_new)
        if self.train:
            if p_new >= 0 and v_new >= 0:
                reward = self.__calculate_reward(a, a_new)
                #self.Q[p][v][a][self.ball.current_action] = (self.learn_rate) * self.Q[p][v][a][self.ball.current_action] + self.learn_rate * (reward)   # + 0.01*((max(Q[p2][v2]))))# - Q[p][v][ball.current_action])))
                curr_val = self.Q[p][v][a][self.ball.current_action]
                self.Q[p][v][a][self.ball.current_action] = curr_val + self.learn_rate * (reward + self.discount_factor * max(self.Q[p_new][v_new][a_new]) - curr_val)
                self.Q = self.__normalised(p, v, a)
                self.Qfreq[p][v][a][self.ball.current_action] += 1
                #print(self.Qfreq[p][v][a][self.ball.current_action])
            if a_new > self.a_high:
                self.a_high = a_new
                #print(self.a_high, self.a_low)
            elif a_new < self.a_low:
                self.a_low = a_new
                #print(self.a_high, self.a_low)
        self.v = v_new
        self.p = p_new

    def set_up_pygame(self):
        pygame.init()
        self.screen = pygame.display.set_mode((600, 600))
        self.clock = pygame.time.Clock()

        # Physics stuff
        self.space = pymunk.Space()
        self.space._set_gravity(Vec2d(0, -981))
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)

        print(self.NUM_NAO_ANGLES, self.__get_state(0, 0, 0)[2], self.__get_state(0, 0, self.MIN_ANGLE)[2]+1, self.__get_state(0, 0, self.MAX_ANGLE)[2]-1)

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

    def check_reset(self):
        reset = False
        if abs(self.ball.current_position) > self.tray_width/2:
            remove_ball(self.ball, self.space)
            reset = True
        return reset

    def reset_scenario(self):
        self.trayBody.angle = random.randrange(-15, 15, 1)/100
        self.trayBody.angular_velocity = 0
        # remove_ball(ball)
        self.add_ball(self.tray_x_pos, 150, 1, self.ball_radius, self.tray_width)
        self.start_time = time.time()

    def update_ball(self):
        self.ball.previous_velocity = self.ball.current_velocity
        self.ball.previous_position = self.ball.current_position
        self.ball.previous_action = self.ball.current_action
        self.ball.previous_angle = self.trayBody.angle

    def compare_times(self, curr_time):
        t = curr_time - self.start_time
        #print(self.iterations, t)
        if t > self.best_time:
            self.best_time = t
        if t > self.time_threshold and self.threshold_counter < self.num_thresholds:
            self.threshold_counter += 1
            #print("Threshold", threshold_counter, "Iterations:", iterations, "Time:", t)

    def reduce_explore_rate(self):
        if self.iterations % self.explore_reduction_freq == 0:
            self.explore_rate /= self.scale_reduction
            print("Iterations:", self.iterations, "Best:", self.best_time)

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
        a_bin = (math.floor(((ang + np.pi) / (2*np.pi)) * self.NUM_ANGLES + self.NUM_ANGLES//4))%self.NUM_ANGLES #starting from right is 0, then increases going anticlockwise
        #In terms of normal angles, when horizontal, angle is 0
        #When going from horizontal and turning anticlockwise angle goes from 0 and increases, with pi at other horizontal
        #When going clockwise, angle goes below 0 and then decreases
        #if abs(pos) >= self.tray_width/2:
            #p_bin = -1
        if p_bin > 1:
            #print("OVER", p_bin)
            p_bin = 1
        elif p_bin < 0:
            #print("UNDER", p_bin)
            p_bin = 0
        #print("PBIN", p_bin, pos)
        return p_bin, v_bin, a_bin

    def __choose_action(self, p_var, v_var, a_var):
        action = 0
        if p_var >= 0 and v_var >= 0:
            if random.random() < self.explore_rate:
                action = random.randint(0, self.NUM_ACTIONS-1)
            else:
                action = np.argmax(self.Q[p_var][v_var][a_var])
        return action

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
        if target_bin >= self.NUM_ANGLE_BINS:
            target_bin = self.NUM_ANGLE_BINS - 1
        elif target_bin < 0:
            target_bin = 0
        if target_angle > self.MAX_ANGLE:
            target_angle = self.MAX_ANGLE
        elif target_angle < self.MIN_ANGLE:
            target_angle = self.MIN_ANGLE
        while turn and self.is_ball_on_tray(self.length_on_tray):
            #print(self.trayBody.angle, target_angle, self.MIN_ANGLE, target_bin)
            if direction == -1:
                if self.trayBody.angle > target_angle and self.trayBody.angle > self.MIN_ANGLE:  # Keep rotating util we are past the angle we are aiming for
                    self.trayBody.apply_force_at_local_point(Vec2d.unit() * self.rotation, (-self.tray_width/2, 0))  # rotate flipper clockwise
                else:
                    turn = False
            elif direction == 1:
                #print(self.NUM_ANGLE_BINS, self.trayBody.angle, target_angle, self.MIN_ANGLE, target_bin)
                if self.trayBody.angle < target_angle and self.trayBody.angle < self.MAX_ANGLE:
                    self.trayBody.apply_force_at_local_point(Vec2d.unit() * self.rotation, (self.tray_width/2, 0))  # rotate flipper anticlockwise
                else:
                    turn = False
            #dt = 1.0/60.0/5.
            #for x in range(5):
            #    self.space.step(dt)

            self.step_simulation(slow, record, draw, speed)

            # if self.draw:
            #     self.screen.fill(THECOLORS["white"])
            #     self.space.debug_draw(self.draw_options)

            #     self.clock.tick(50)
            #     pygame.display.flip()
            # c_speed = trainer.speed_of_ball_at_centre()
            # if c_speed is not None:
            #     print(c_speed)
        self.a = target_bin
        self.trayBody.angular_velocity = 0
        self.prev_ball_pos = self.get_pos_ball_along_tray()
        

    # dist is distance from centre to apply the force, a_bin is the bin of the current angle of the tray
    # def __change_state(self, dist, a_bin):
    #     action = self.ball.current_action
    #     #count = 0
    #     turn = True
    #     while turn:
    #         if action == 0:
    #             if self.trayBody.angle > self.__get_centre_of_ang_bin((a_bin-1)%self.NUM_ANGLES) and self.trayBody.angle > self.MIN_ANGLE:
    #                 #print(self.trayBody.angle)
    #                 self.trayBody.apply_force_at_local_point(Vec2d.unit() * self.rotation, (-dist, 0))  # rotate flipper clockwise
    #             else:
    #                 turn = False
    #         elif action == 1:
    #             if self.trayBody.angle < self.__get_centre_of_ang_bin((a_bin+1)%self.NUM_ANGLES) and self.trayBody.angle < self.MAX_ANGLE:
    #                 self.trayBody.apply_force_at_local_point(Vec2d.unit() * self.rotation, (dist, 0))  # rotate flipper anticlockwise
    #             else:
    #                 turn = False
    #         dt = 1.0/60.0/5.
    #         for x in range(5):
    #             self.space.step(dt)

            #print(self.trayBody.angle, action)
                    #print(action, "ANG:", self.trayBody.angle > self.__get_centre_of_ang_bin((a_bin-1)%self.NUM_ANGLES), self.trayBody.angle, self.__get_centre_of_ang_bin((a_bin-1)%self.NUM_ANGLES))
                    #self.ball.current_position = self.__get_ball_pos_x(self.ball)
                    #self.ball.current_velocity = self.ball.body.velocity[0]
                    #self.ball.current_angle = self.trayBody.angle
                    #p_new, v_new, a_new = get_state(self.ball.current_position, self.ball.current_velocity, self.ball.current_angle)
                    #count+=1
            #elif action == 1:
            #    while self.trayBody.angle < self.__get_centre_of_ang_bin((a_bin+1)%self.NUM_ANGLES) and self.trayBody.angle < self.MAX_ANGLE:
            #        self.trayBody.apply_force_at_local_point(Vec2d.unit() * self.rotation, (dist, 0))  # rotate flipper anticlockwise
                    #print(self.trayBody.angle)
            #        dt = 1.0/60.0/5.
            #        for x in range(5):
            #            self.space.step(dt)
            #        count+=1
                #print(action, "ANG:", self.trayBody.angle < self.__get_centre_of_ang_bin((a_bin+1)%self.NUM_ANGLES), self.trayBody.angle, self.__get_centre_of_ang_bin((a_bin+1)%self.NUM_ANGLES))
        #print(count)
        # self.ball.current_position = self.__get_ball_pos_x(self.ball)
        # self.ball.current_velocity = self.ball.body.velocity[0]
        # self.ball.current_angle = self.trayBody.angle
        # ppp, vvv, aaa = self.__get_state(self.ball.current_position, self.ball.current_velocity, self.ball.current_angle)
        # self.trayBody.angular_velocity = 0
        # return ppp, vvv, aaa

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

    def __calculate_reward(self, a_bin, a_bin_new):
        reward = 0

        #Reward states that tilt to decrease speed
        #Just this with initial reward 0 works quite well
        if (self.ball.current_velocity > 0 and a_bin_new > a_bin) or (self.ball.current_velocity < 0 and a_bin_new < a_bin):
            reward += abs(self.tray_x_pos - self.ball.current_position)/self.tray_width
        elif (self.ball.current_velocity > 0 and abs(self.ball.current_velocity) < self.MAX_VELOCITY*0.01 and self.ball.current_position > self.tray_x_pos) or (self.ball.current_velocity < 0 and abs(self.ball.current_velocity) < self.MAX_VELOCITY*0.01 and self.ball.current_position < self.tray_x_pos):
            reward += (abs(self.tray_x_pos - self.ball.current_position)/self.tray_width)
        #    print(self.ball.current_velocity)
        else:
            reward += -1 * (abs(self.tray_x_pos - self.ball.current_position)/self.tray_width)
        #Reward states that have low velocity
        # Maybe change and do with bucket number
        #if abs(self.ball.current_velocity) < self.MAX_VELOCITY * 0.25:
        #    reward += (1 - abs(self.ball.current_velocity) / self.MAX_VELOCITY)
        #elif abs(self.ball.current_velocity) >= self.MAX_VELOCITY * 0.75:
        #    reward -= (1 - abs(self.ball.current_velocity) / self.MAX_VELOCITY)

        #Reward states that are close to centre
        #reward += -1 * ((abs(self.ball.current_position - self.tray_x_pos)/(self.tray_width/2))**2) + 2
        #if abs(self.ball.current_position - self.tray_x_pos) < abs(self.ball.previous_position - self.tray_x_pos):
        #    reward += 1 * abs(self.ball.current_position - self.tray_x_pos)/(self.tray_width/2)
        #else:
        #    reward += -1 * abs(self.ball.current_position - self.tray_x_pos)/(self.tray_width/2)

        # reward = 1 - abs(self.tray_x_pos - self.ball.current_position)/(self.tray_width/2.0) # Start between 0 and 1, determined by how close to centre
        # reward = (reward -0.5) *2
        # reward2 = 1 - abs(self.v - (self.NUM_VELOCITIES/2)) / (self.NUM_VELOCITIES/2)
        # reward2 = (reward2 -0.5) *2
        # reward3 = 1 - abs(self.ball.current_angle)/abs(self.MAX_ANGLE) # If tray flatter, reward higher
        # reward3 = (reward3 -0.5) *2
        #reward = (reward + reward2 + reward3)/3
        if self.trained:
            print(reward, "\n")
        #print(reward, 1 - abs(self.tray_x_pos - self.ball.current_position)/(self.tray_width/2.0), (1.0 - abs(self.v - (self.NUM_VELOCITIES/2.0)) / (self.NUM_VELOCITIES/2.0)), (1.0 - abs(self.ball.current_angle)/abs(self.MAX_ANGLE)))
        return reward

    def __normalised(self, p_var, v_var, a_var):
        q_var = copy.copy(self.Q)
        sumup = 0
        for i in q_var[p_var][v_var][a_var]:
            sumup += abs(i)
        if sumup > 0:
            for i in range(0, len(self.Q[p_var][v_var][a_var])):
                q_var[p_var][v_var][a_var][i] = self.Q[p_var][v_var][a_var][i]/sumup
        #if sum(self.Q[p_var][v_var][a_var]) > 0:
        #    for i in range(0, len(self.Q[p_var][v_var][a_var])):
        #        q_var[p_var][v_var][a_var][i] = self.Q[p_var][v_var][a_var][i]/sum(self.Q[p_var][v_var][a_var])
        return q_var

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
        inertia = pymunk.moment_for_circle(mass, 0, radius, (0,0))
        body = pymunk.Body(1, inertia)
        ang = np.pi/2 - angle - np.arctan2(radius, xdist)  # Right angle - angle of tray - angle from the tray to the line that goes through the centre of the ball
        hyp = np.sqrt(xdist * xdist + radius * radius)  # Dist from centre of tray to centre of ball
        xpos = hyp * np.sin(ang)  # Distance in x direction of ball, from centre of tray
        ypos = hyp * np.cos(ang)
        body.position = self.tray_x_pos + xpos, self.tray_y_pos + ypos + 0.1  # extra 0.1 makes sure does not lie in the tray
        shape = pymunk.Circle(body, radius, (0,0))
        shape.elasticity = 0#0.95
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
        self.prev_ball_pos = xdist
        self.ball = shape
        self.num_frames_off_tray = 0  # Reset the numebr of frams the ball has been off the tray for
        self.ball_changed_direction = False  # THIS AND THE NEXT TWO SHOULD BE MADE A PROPERTY OF THE BALL
        self.was_negative = False
        self.was_positive = False

    def get_pos_ball_along_tray(self):
        #ang = np.pi/2 - self.ball.angle - np.arctan2(self.ball.rad, self.ball.xdist)  # Right angle - angle of tray - angle from the tray to the line that goes through the centre of the ball
        #hyp = np.sqrt(self.ball.xdist * self.ball.xdist + self.ball.rad * self.ball.rad)  # Dist from centre of tray to centre of ball
        #xpos = hyp * np.sin(ang) 
        x = self.ball.body.position[0] - self.tray_x_pos
        return x / np.cos(self.trayBody.angle)

    def __remove_ball(self, ball_to_remove):
        self.space.remove(ball_to_remove, ball_to_remove.body)

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

num_bins_ang = 4 # number of angles above and below
num_bins_pos = 40
num_bins_vel = 30
num_actions = num_bins_ang * 2  # First half is clockwise (-1 dir), second half is anticlockwise (1 dir)

max_ang = 0.1
max_pos = 200
max_vel = 600
#init_vel_range = 50

q_mat = np.zeros((num_bins_pos, num_bins_vel, num_bins_ang, num_actions))

trainer = BallBalancer()
trainer.set_up_pygame()
trainer.create_world()

reduction_freq = 2000
random_action_chance = 1
reduction_amount = 1.5

trainer.length_on_tray = 30
trainer.discount_factor = 0

trainer.NUM_ANGLE_BINS = num_bins_ang  # Make it so dont have to do this

TRAINING = True
TRAINING = False
SAVING =  TRAINING
LOADING = True
if TRAINING:
    i = 0
    while i < 30000 and trainer.continue_running():
        new_ang = (random.randint(0, max_ang*100)/100) * ((-1)**(random.randint(1,2)))  # Random andgle the tray will be for this test
        new_pos = random.randint(0, max_pos) * ((-1)**(random.randint(1,2)))  # Random position of the ball on the tray for this test
        new_vel = random.randint(0, max_vel) * ((-1)**(random.randint(1,2)))  # Random velocity of the ball for this test

        if random.random() < 0.5:
            new_pos = new_pos // 2.0
            new_vel = new_vel // 2.0 

        trainer.update_tray_angle(new_ang)  # Move the tray to the chosen position
        trainer.add_ball(new_pos, new_vel)  # Add the new ball
        
        bin_ang = trainer.get_bin(new_ang, max_ang, num_bins_ang)
        trainer.a = bin_ang
        bin_pos = trainer.get_bin(new_pos, max_pos, num_bins_pos)
        bin_vel = trainer.get_bin(new_vel, max_vel, num_bins_vel)
        
        #num_bins = random.randint(0,trainer.NUM_ANGLE_BINS-1)  # Generate the random number of angles that the tray will move round
        #direction = random.choice([-1, 1])  # Randomly choose a direction. -1 is clockwise, 1 is anticlockwise
        random_action = random.random()
        action = np.argmax(q_mat[bin_pos][bin_vel][bin_ang])  # The action to take, the one with biggest value for the chosen pos, vel, ang values
        if random_action < random_action_chance:
            action = random.choice(range(0, num_actions))  # Do a random action instead

        num_bins, direction = trainer.action_to_num_and_dir(action, num_actions)  # Use that value to get the number of angle bins to rotate round, and in what direction
        #print("ANG:", new_ang, bin_ang, "  POS:", new_pos, bin_pos, "  VEL:", new_vel, bin_vel, "  BINS:", num_bins, "  DIR:", direction)
        #print("ang b4:", trainer.trayBody.angle)
        trainer.do_action(num_bins, direction, draw=False, speed=1000)  # Carry out that action
        #print("angle", trainer.trayBody.angle, bin_ang, "\naction", action, "\ndir", direction, "\nnum bins", num_bins, "\n\n")
        reward = -1
        while trainer.is_ball_on_tray():  # See what happens after action has taken place
            trainer.step_simulation(True, True, draw=False, draw_speed=1000)
            if trainer.speed_past_origin is not None and trainer.ball_changed_direction:
                reward = 1 - (abs(trainer.speed_past_origin) / max_vel)
                #print("Speed:", trainer.speed_past_origin, "Reward:", reward)

        curr_val = copy.copy(q_mat[bin_pos][bin_vel][bin_ang][action])
        q_mat[bin_pos][bin_vel][bin_ang][action] = curr_val + trainer.learn_rate * (reward + trainer.discount_factor * (max(q_mat[bin_pos][bin_vel][bin_ang]) - curr_val))
        q_mat[bin_pos][bin_vel][bin_ang] = trainer.normalised(q_mat, q_mat[bin_pos][bin_vel][bin_ang])
        #print("HI", curr_val, trainer.learn_rate, reward, trainer.discount_factor, max(q_mat[bin_pos][bin_vel][bin_ang]), q_mat[bin_pos][bin_vel][bin_ang][action])

            #trainer.draw_scene()
            #c_speed = trainer.speed_of_ball_at_centre()
            #if c_speed is not None:
            #    print(c_speed)
            #trainer.is_ball_on_tray()
        trainer.remove_ball()

        i+=1

        if i % reduction_freq == 0:
            random_action_chance /= reduction_amount
            print(i)
        #trainer.step_simulation(True, True, True)
        #trainer.draw_scene()

        
if SAVING:
    np.save("q", q_mat)
if LOADING:
    q_mat = np.load("q.npy")

for i in range(0, 5):
    
    # Generate random starting position, velocity and angle for ball
    new_pos = random.randint(0, max_pos) * ((-1)**(random.randint(1,2)))  # Random position of the ball on the tray for this test
    new_vel = 0# random.randint(0, max_vel) * ((-1)**(random.randint(1,2)))  # Random velocity of the ball for this test
    new_ang = (random.randint(0, max_ang*100)/100) * ((-1)**(random.randint(1,2)))  # Random andgle the tray will be for this test

    trainer.update_tray_angle(new_ang)  # Move the tray to the chosen position
    trainer.add_ball(new_pos, new_vel)  # Add the new ball
    print("START", i, new_pos, new_vel, new_ang)
    while trainer.is_ball_on_tray(trainer.length_on_tray) and trainer.continue_running():
        trainer.reset_draw()
        bin_pos = trainer.get_bin(new_pos, max_pos, num_bins_pos)
        bin_vel = trainer.get_bin(new_vel, max_vel, num_bins_vel)
        bin_ang = trainer.get_bin(new_ang, max_ang, num_bins_ang)
        action = np.argmax(q_mat[bin_pos][bin_vel][bin_ang])
        
        num_bins, direction = trainer.action_to_num_and_dir(action, num_actions)
        print("\n\npos:", new_pos, bin_pos, "\nvel:", new_vel, bin_vel, "\nang:", new_ang, bin_ang, "\nact:", action, "\nbins:", num_bins, " dir:", direction, "\nq:", q_mat[bin_pos][bin_vel][bin_ang])
        trainer.do_action(num_bins, direction, draw=True, speed=60)
        #print(q_mat[bin_pos][bin_vel][bin_ang], action)
        #for j in range(0, 1):
        #    trainer.step_simulation(True, True, True, 60)
        new_pos, new_vel, new_ang = trainer.get_ball_info()
        #print(i, new_pos, new_vel, new_ang)
    print("FOT:", trainer.num_frames_off_tray)
    trainer.remove_ball()





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