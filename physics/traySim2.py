
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




def remove_ball(ball_to_remove):
    space.remove(ball_to_remove, ball_to_remove.body)


def get_ball_pos_x(ball):
    pos = -1
    dist_squared = (ball.body.position.x - x_pos)**2 + (ball.body.position.y - y_pos)**2 - (ball_radius+h/2)**2
    if(dist_squared < 0):# and abs(dist_squared) < 100):
        pos = 0
    else:
        pos = math.sqrt(dist_squared)
    if ball.body.position.x < x_pos:
        pos = pos * -1
    return pos


def get_q_idx_pos(pos):  # Get the indices of the Q matrix based on mapping the position and velocity to the buckets
    pos = pos + w/2  # Map from relative to centre of tray to relative from start of tray
    bins = []
    bin_count = 0
    while bin_count < w-1:
        bins.append(bin_count)
        bin_count += round(w/NUM_X_DIVS)
    x = [pos]
    pos_bin = np.digitize(x, bins)
    return pos_bin[0]-1


def get_q_idx_vel(vel):
    vel = vel + MAX_VELOCITY  # Map from relative to centre of tray to relative from start of tray
    bins = []
    bin_count = 0
    while bin_count < MAX_VELOCITY*2-1:
        bins.append(bin_count)
        bin_count += round((MAX_VELOCITY*2)/NUM_VELOCITIES)
    x = [vel]
    vel_bin = np.digitize(x, bins)
    return vel_bin[0]-1  # -1 because starts at 0


def get_state(pos, vel, ang):
    p_bin = math.floor(((pos + (w/2)) / (w)) * NUM_X_DIVS)
    
    v_bin = math.floor(((vel + MAX_VELOCITY) / (MAX_VELOCITY*2)) * NUM_VELOCITIES)
    a_bin = (math.floor(((ang + np.pi) / (2*np.pi)) * NUM_ANGLES + NUM_ANGLES//4))%NUM_ANGLES #starting from right is 0, then increases going anticlockwise
    #In terms of normal angles, when horizontal, angle is 0
    #When going from horizontal and turning anticlockwise angle goes from 0 and increases, with pi at other horizontal
    #When going clockwise, angle goes below 0 and then decreases
    if abs(pos) >= w/2:
        p_bin = -1
    if abs(vel) >= MAX_VELOCITY:
        v_bin = -1 
    return p_bin, v_bin, a_bin


def get_centre_of_ang_bin(b):
    jump = 2*np.pi / NUM_ANGLES  # Size of a bin
    bin_pos = (b + 0.5) % NUM_ANGLES  # The bth bin, and then half way through that
    centre_angle = (jump * bin_pos - np.pi - np.pi/2)  # -pi since tray angles go from -pi to +pi, and want 0 to 2pi
    return centre_angle


def add_ball(xpos, ypos, mass, radius):
    inertia = pymunk.moment_for_circle(mass, 0, radius, (0,0))
    body = pymunk.Body(1, inertia)
    body.position = random.randint(x_pos - w/4, x_pos + w/4), ypos
    shape = pymunk.Circle(body, radius, (0,0))
    shape.elasticity = 0#0.95
    space.add(body, shape)
    shape.current_velocity = 0
    shape.current_position = get_ball_pos_x(shape)
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


def change_state(ball_var, tray, strength, dist, a_bin, sp):
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
    ppp, vvv, aaa = get_state(ball_var.current_position, ball_var.current_velocity, ball_var.current_angle)
    tray.angular_velocity = 0
    return ppp, vvv, aaa


def calculate_reward(ball_var, a_bin, a_bin_new):
    reward = 0
    if ball_var.current_velocity > 0 and a_bin_new > a_bin:
        reward += 1
    if ball_var.current_velocity < 0 and a_bin_new < a_bin:
        reward += 1
    return reward


def normalised(q_var, p_var, v_var, a_var):
    if sum(Q[p_var][v_var][a_var]) > 0:
        for i in range(0, len(Q[p_var][v_var][a_var])):
            Q[p_var][v_var][a_var][i] = Q[p_var][v_var][a_var][i]/sum(Q[p_var][v_var][a_var])
    return q_var

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


def next_params(var_name, var_pos, defaults, params):
    res = copy.copy(defaults)
    if var_pos >= 0:
        res[var_name] = params[var_name][var_pos]
    return res

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
            ball = add_ball(400, 150, 1, ball_radius)
            iterations += 1
            trayBody.angle = random.randrange(-15, 15, 1)/100
            trayBody.angular_velocity = 0

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
