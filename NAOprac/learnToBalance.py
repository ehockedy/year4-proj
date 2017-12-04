
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


def normalised(Q, q_var, p_var, v_var, a_var):
    if sum(Q[p_var][v_var][a_var]) > 0:
        for i in range(0, len(Q[p_var][v_var][a_var])):
            q_var[p_var][v_var][a_var][i] = Q[p_var][v_var][a_var][i]/sum(Q[p_var][v_var][a_var])
    return q_var
"""
# TODO:
# Allow it to loop different parameters in order to compare training times
# Save to file the results of the training

# PARAMETERS AND MAIN LOOP
NUM_ACTIONS = 2  # Number of actions that can be take. Normally 2, rotate clockwise or anticlockwise
NUM_X_DIVS = 2  # Number of divisions x plane is split into, for whole tray
NUM_VELOCITIES = 2  # Number of velocity buckets
NUM_ANGLES = 200 #2 * np.pi
#MAX_ANGLE = math.atan((y_pos - stop_y)/abs(x_pos - stop_x_left))
MAX_VELOCITY = 400#math.sqrt(2 * mass/100 * 981/100 * math.sin(MAX_ANGLE) * w)  # Using SUVAT and horizontal component of gravity, /100 because of earlier values seem to be *100

#Q = np.zeros((NUM_X_DIVS, NUM_VELOCITIES, NUM_ANGLES, NUM_ACTIONS))

ball_radius = 25
ball = add_ball(400, 200, 1, ball_radius)

iterations = 0

rotation = 10000

TIME_THRESHOLDS = [0.4, 0.6]
SCALE_REDUCTIONS = [1.7]
LEARN_RATES = [0.7]
NUM_ANGLES_OPTIONS = [540, 720]
NUM_VELOCITIES_OPTIONS = [15, 20]
NUM_X_DIVS_OPTIONS = [15, 25]
P_params = [TIME_THRESHOLDS, SCALE_REDUCTIONS, LEARN_RATES, NUM_ANGLES_OPTIONS, NUM_VELOCITIES_OPTIONS, NUM_X_DIVS_OPTIONS]
P_defaults = [0.3, 1.6, 0.5, NUM_ANGLES, NUM_VELOCITIES, NUM_X_DIVS]

Q = np.zeros((P_defaults[5], P_defaults[4], P_defaults[3], NUM_ACTIONS))

best_time = 0
very_best_time = 0
start_time = time.time()
time_threshold = 0.7
real_time_threshold = 30

explore_rate = 0
learnRate = 0.5
scale_reduction = 1.7

threshold_counter = 0
num_thresholds = 15
max_num_iterations = 1000

log = open("log.txt", "a")

defo_running = True
DO_PARAM_TEST = False
DONT_SHOW_ONCE_DONE = False
TRAINING = True
if TRAINING:
    scale_reduction = 1.7
    explore_rate = 1
    log.write("\n\nStarting training\n\n")
else:
    Q = np.load("qGood.npy")

P_vars = [time_threshold, scale_reduction, learnRate, NUM_ANGLES, NUM_VELOCITIES, NUM_X_DIVS]
"""

def next_params(var_name, var_pos, defaults, params):
    res = copy.copy(defaults)
    if var_pos >= 0:
        res[var_name] = params[var_name][var_pos]
    return res


class BallBalancer:
    def __init__(self):
        self.tray_width = 400
        self.tray_height = 20
        self.tray_x_pos = 300
        self.tray_y_pos = 100
        self.tray_angle = np.pi / 24
        self.rotation = 10000
        
        self.ball_radius = 25

        self.NUM_ACTIONS = 2  # Number of actions that can be taken. Normally 2, rotate clockwise or anticlockwise
        self.NUM_X_DIVS = 2  # Number of divisions x plane is split into, for whole tray
        self.NUM_VELOCITIES = 2  # Number of velocity buckets
        self.NUM_ANGLES = 100 #2 * np.pi
        self.MAX_VELOCITY = 400#math.sqrt(2 * mass/100 * 981/100 * math.sin(MAX_ANGLE) * w)  # Using SUVAT and horizontal component of gravity, /100 because of earlier values seem to be *100

        self.iterations = 0

        self.Q = np.zeros((self.NUM_X_DIVS, self.NUM_VELOCITIES, self.NUM_ANGLES, self.NUM_ACTIONS))

        self.best_time = 0
        self.time_threshold = 0.7
        self.real_time_threshold = 30

        self.explore_rate = 0.2
        self.learn_rate = 0.8
        self.scale_reduction = 2#1.7

        self.max_time = 100
        self.threshold_counter = 0
        self.num_thresholds = 15
        self.max_num_iterations = 1000
        self.max_draw_iterations = 500
        self.explore_reduction_freq = 100

        self.start_time = time.time()

        self.Qfreq = copy.copy(self.Q)

        self.a_high = 0
        self.a_low = self.NUM_ANGLES

    def perform_episode(self):
        # GET THE STATES
        p, v, a = self.__get_state(self.ball.previous_position, self.ball.previous_velocity, self.ball.previous_angle)
        # DECIDE ON THE BEST ACTION
        self.ball.current_action = self.__choose_action(p, v, a)

        # GET THE STATES ONCE UPDATED
        p_new, v_new, a_new = self.__change_state(self.tray_width/2, a)

        # UPDATE Q
        if p_new >= 0 and v_new >= 0:
            reward = self.__calculate_reward(a, a_new)
            self.Q[p][v][a][self.ball.current_action] = (1 - self.learn_rate) * self.Q[p][v][a][self.ball.current_action] + self.learn_rate * (reward)# + 0.01*((max(Q[p2][v2]))))# - Q[p][v][ball.current_action])))
            self.Q = self.__normalised(p, v, a)
            self.Qfreq[p][v][a][self.ball.current_action] += 1
            #print(self.Qfreq[p][v][a][self.ball.current_action])
        if a_new > self.a_high:
            self.a_high = a_new
            print(self.a_high, self.a_low)
        elif a_new < self.a_low:
            self.a_low = a_new
            print(self.a_high, self.a_low)

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
        trayJointBody.position = trayBody.position
        j = pymunk.PinJoint(self.trayBody, trayJointBody, (0, 0), (0, 0))
        self.space.add(j)

        self.__add_ball(400, 200, 1, self.ball_radius, self.tray_width)

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
        self.__add_ball(self.tray_x_pos, 150, 1, self.ball_radius, self.tray_width)
        self.start_time = time.time()

    def update_ball(self):
        self.ball.previous_velocity = self.ball.current_velocity
        self.ball.previous_position = self.ball.current_position
        self.ball.previous_action = self.ball.current_action
        self.ball.previous_angle = self.trayBody.angle

    def compare_times(self, curr_time):
        t = curr_time - self.start_time
        print(self.iterations, t)
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
        if abs(pos) >= self.tray_width/2:
            p_bin = -1
        if abs(vel) >= self.MAX_VELOCITY:
            v_bin = -1 
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
    def __change_state(self, dist, a_bin):
        action = self.ball.current_action
        target = 0
        if action == 0:
            while self.trayBody.angle > self.__get_centre_of_ang_bin((a_bin-1)%self.NUM_ANGLES):
                self.trayBody.apply_force_at_local_point(Vec2d.unit() * self.rotation, (-dist, 0))  # rotate flipper clockwise
                dt = 1.0/60.0/5.
                for x in range(5):
                    self.space.step(dt)
                self.ball.current_position = self.__get_ball_pos_x(self.ball)
                self.ball.current_velocity = self.ball.body.velocity[0]
                self.ball.current_angle = self.trayBody.angle
                #p_new, v_new, a_new = get_state(self.ball.current_position, self.ball.current_velocity, self.ball.current_angle)
        elif action == 1:
            while self.trayBody.angle < self.__get_centre_of_ang_bin((a_bin+1)%self.NUM_ANGLES):
                self.trayBody.apply_force_at_local_point(Vec2d.unit() * self.rotation, (dist, 0))  # rotate flipper anticlockwise
                dt = 1.0/60.0/5.
                for x in range(5):
                    self.space.step(dt)
                self.ball.current_position = self.__get_ball_pos_x(self.ball)
                self.ball.current_velocity = self.ball.body.velocity[0]
                self.ball.current_angle = self.trayBody.angle
        elif action == 2:
            while self.trayBody.angle > self.__get_centre_of_ang_bin((a_bin-2)%self.NUM_ANGLES):
                self.trayBody.apply_force_at_local_point(Vec2d.unit() * self.rotation, (-dist, 0))  # rotate flipper anticlockwise
                dt = 1.0/60.0/5.
                for x in range(5):
                    self.space.step(dt)
                self.ball.current_position = self.__get_ball_pos_x(self.ball)
                self.ball.current_velocity = self.ball.body.velocity[0]
                self.ball.current_angle = self.trayBody.angle
        elif action == 3:
            while self.trayBody.angle < self.__get_centre_of_ang_bin((a_bin+2)%self.NUM_ANGLES):
                self.trayBody.apply_force_at_local_point(Vec2d.unit() * self.rotation, (dist, 0))  # rotate flipper anticlockwise
                dt = 1.0/60.0/5.
                for x in range(5):
                    self.space.step(dt)
                self.ball.current_position = self.__get_ball_pos_x(self.ball)
                self.ball.current_velocity = self.ball.body.velocity[0]
                self.ball.current_angle = self.trayBody.angle
        ppp, vvv, aaa = self.__get_state(self.ball.current_position, self.ball.current_velocity, self.ball.current_angle)
        self.trayBody.angular_velocity = 0
        return ppp, vvv, aaa

    def __calculate_reward(self, a_bin, a_bin_new):
        reward = 0
        if self.ball.current_velocity > 0 and a_bin_new > a_bin:
            reward += 1 * (abs(self.tray_x_pos - self.ball.current_position)/self.tray_width)
        if self.ball.current_velocity < 0 and a_bin_new < a_bin:
            reward += 1 * (abs(self.tray_x_pos - self.ball.current_position)/self.tray_width)
        return reward

    def __normalised(self, p_var, v_var, a_var):
        q_var = copy.copy(self.Q)
        if sum(self.Q[p_var][v_var][a_var]) > 0:
            for i in range(0, len(self.Q[p_var][v_var][a_var])):
                q_var[p_var][v_var][a_var][i] = self.Q[p_var][v_var][a_var][i]/sum(self.Q[p_var][v_var][a_var])
        return q_var

    def __add_ball(self, xpos, ypos, mass, radius, tray_width):
        inertia = pymunk.moment_for_circle(mass, 0, radius, (0,0))
        body = pymunk.Body(1, inertia)
        body.position = random.randint(xpos - tray_width/4, xpos + tray_width/4), ypos
        shape = pymunk.Circle(body, radius, (0,0))
        shape.elasticity = 0#0.95
        self.space.add(body, shape)
        shape.current_velocity = 0
        shape.current_position = self.__get_ball_pos_x(shape)
        shape.current_action = 0
        shape.current_angle = 0
        shape.previous_action = 0
        shape.previous_velocity = 0
        shape.previous_position = 0
        shape.previous_angle = 0
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
        if ball.body.position.x < x_pos:
            pos = pos * -1
        return pos

    def __get_centre_of_ang_bin(self, b):
        jump = 2*np.pi / self.NUM_ANGLES  # Size of a bin
        bin_pos = (b + 0.5) % self.NUM_ANGLES  # The bth bin, and then half way through that
        centre_angle = (jump * bin_pos - np.pi - np.pi/2)  # -pi since tray angles go from -pi to +pi, and want 0 to 2pi
        return centre_angle

    def draw(self):
        self.screen.fill(THECOLORS["white"])
        self.space.debug_draw(self.draw_options)

        self.clock.tick(10)
        pygame.display.flip()

running = True
trainer = BallBalancer()
trainer.set_up_pygame()
trainer.create_world()
start = time.time()
print("START")
while running:
    trainer.perform_episode()
    trainer.update_ball()

    if trainer.iterations == trainer.max_draw_iterations:
        trainer.explore_rate = 0
        trainer.a_high = 0
        trainer.a_low = trainer.NUM_ANGLES
    if trainer.iterations > trainer.max_draw_iterations:
        trainer.draw()

    if trainer.check_reset():
        trainer.compare_times(time.time())
        trainer.reduce_explore_rate()
        trainer.reset_scenario()
        trainer.iterations += 1

    if trainer.terminate(time.time()):
        running = False



print("DONE")



def train_single_params():
    running = True
    while running:
        perform_episode()
        update_ball()

        if check_reset():
            best_time, threshold_counter = compare_times()
            explore_rate = reduce_explore_rate()
            reset()

        if terminate():
            running = False








var = 0
val = -1
while var < len(P_params) and defo_running:
    if DO_PARAM_TEST:
        time_threshold, scale_reduction, learnRate, NUM_ANGLES, NUM_VELOCITIES, NUM_X_DIVS = next_params(var, val, P_defaults, P_params)
        Q = np.zeros((NUM_X_DIVS, NUM_VELOCITIES, NUM_ANGLES, NUM_ACTIONS))
        print(time_threshold, scale_reduction, learnRate, NUM_ANGLES, NUM_VELOCITIES, NUM_X_DIVS, var, val)
    while running and defo_running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == KEYDOWN and event.key == K_ESCAPE:
                running = False
                defo_running = False

        # GET THE STATES
        p, v, a = get_state(ball.previous_position, ball.previous_velocity, ball.previous_angle)

        # DECIDE ON THE BEST ACTION
        ball.current_action = choose_action(p, v, a, Q, explore_rate, NUM_ACTIONS)

        # GET THE STATES ONCE UPDATED
        p_new, v_new, a_new = change_state(ball, trayBody, rotation, w/2, a, space)

        # UPDATE Q
        if p_new >= 0 and v_new >= 0 and TRAINING:
            reward = calculate_reward(ball, a, a_new)
            Q[p][v][a][ball.current_action] = (1 - learnRate) * Q[p][v][a][ball.current_action] + learnRate * (reward)# + 0.01*((max(Q[p2][v2]))))# - Q[p][v][ball.current_action])))
            Q = normalised(Q, p, v, a)

        # RESET THE SCENARIO
        if reset:        
            iterations += 1
            trayBody.angle = random.randrange(-15, 15, 1)/100
            trayBody.angular_velocity = 0
            ball = add_ball(400, 150, 1, ball_radius)

            t = time.time() - start_time
            if t > best_time:
                best_time = t
            if t > time_threshold and threshold_counter < num_thresholds:
                threshold_counter += 1
                print("Threshold", threshold_counter, "Iterations:", iterations, "Time:", t)
            if iterations % 100 == 0:
                explore_rate /= scale_reduction
                print("Iterations:", iterations, "Best:", best_time)

            start_time = time.time()
            reset = False

        # IF GOOD ENOUGH
        if (threshold_counter >= num_thresholds) or not TRAINING:
            explore_rate = 0.0
            screen.fill(THECOLORS["white"])
            space.debug_draw(draw_options)

            clock.tick(10)
            if DONT_SHOW_ONCE_DONE:
                running = False

        if best_time > real_time_threshold and DONT_SHOW_ONCE_DONE:
            running = False

        # IF TRAINED TOO MUCH
        if time.time() - start_time > 100:
            best_time = time.time() - start_time
            running = False
        if iterations >= max_num_iterations and DONT_SHOW_ONCE_DONE:
            running = False

        # Update velocities
        ball.previous_velocity = ball.current_velocity
        ball.previous_position = ball.current_position
        ball.previous_action = ball.current_action
        ball.previous_angle = trayBody.angle

        if abs(ball.current_position) > w/2:
            remove_ball(ball)
            reset = True

        # Flip screen
        pygame.display.flip()

        #END OF EPISODE LOOP

    if TRAINING and defo_running:
        spaces = "    "
        if iterations >= 10000:
            spaces = " "
        elif iterations >= 1000:
            spaces = "  "
        elif iterations >= 100:
            spaces = "   "
        log.write("ITERATIONS: " + str(iterations) + spaces + "BEST_TIME: " + str(best_time) + "   TIME_THRESHOLD " + str(time_threshold) + "   SCALE_REDUCTION: " + str(scale_reduction) + "   LEARN_RATE: " + str(learnRate) + "   NUM ACTIONS: " + str(NUM_ACTIONS) + "   NUM_X_DIVS: " + str(NUM_X_DIVS) + "   NUM_VELOCITIES: " + str(NUM_VELOCITIES) + "   NUM_ANGLES: " + str(NUM_ANGLES) + "\n")
        print("WRITING")
    if TRAINING and best_time > very_best_time and defo_running:
        np.save("q", Q)
        very_best_time = best_time
    if DO_PARAM_TEST:
        if val == len(P_params[var])-1:
            val = 0
            var += 1
        else:
            val += 1
        running = True
        best_time = 0
        iterations = 0
        threshold_counter = 0
        explore_rate = 1
        start_time = time.time()
    else:
        defo_running = False
log.close()
