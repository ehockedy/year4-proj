import nao_functions as nf
import time
import numpy as np
import random
import copy
import configparser
import json
import os
from msvcrt import getch


"""
This file holds the main functions for running the algorithms on the nao robot
"""

config = configparser.ConfigParser()
config.read('config.ini')


def setup_nao(ip):
    nao_ip = ip
    nao = nf.Nao(nao_ip)
    nao.ip = ip
    return nao


def __get_bin_values_q():
    config_p = config["q_params"]["p"]
    config_v = config["q_params"]["v"]
    config_a = config["q_params"]["a"]
    bin_values = [int(config_p), int(config_v), int(config_a)]
    return bin_values


def __get_bin_values_nn():
    config_p = config["nn_params"]["p"]
    config_v = config["nn_params"]["v"]
    config_a = config["nn_params"]["a"]
    bin_values = [int(config_p), int(config_v), int(config_a)]
    return bin_values


def __get_max_values():
    config_p = config["nao_params"]["p_max"]
    config_v = config["nao_params"]["v_max"]
    config_a = config["nao_params"]["a_max"]
    max_values = [float(config_p), float(config_v), float(config_a)]
    return max_values


def balance_ball_q_mat(nao, wait_time=0.0, run_time=300, ball_update=0.01, delay=False, iser=False, prnt=False):
    """
    Balance the ball on the nao robot using q-matrix to make decisions
    ip - the ip address of the nao robot to connect to
    delay - whether to load the q matrix with delay or not
    wait_time - time to wait between each action
    run_time = total number of actions to make
    """
    bin_values = __get_bin_values_q()
    max_values = __get_max_values()
    nao.num_angles = bin_values[2]

    # Set up the nao
    nao.initial_setup()
    nao.set_up_to_hold_tray()  # move hands out in front
    nao.hands_open()  # Open hands
    raw_input("Press enter to continue: ")  # Wait so the user can position tray
    nao.hands_grab()  # Close the hands and grab the tray
    nao.continually_update_ball_information(ball_update)  # update the current recorded position of the ball every ball_update seconds

    q_mat = nf.load_q_matrix(bin_values[0], bin_values[1], bin_values[2], act=2, is_nao=False, is_delay=delay, is_er=iser)

    ang_val = nao.angle_lr_interpolation

    prev_state = (-1, -1, -1)
    prev_vals = (-1, -1, -1)
    action = 0
    for i in range(0, run_time):
        pos = nao.ball_pos_lr  # Get the NN inputs based off the state of the ball and tray
        vel = nao.ball_vel_lr
        ang = nao.ball_ang_lr
        ball_values = (pos, vel, ang)

        while ball_values == prev_vals:  # Make sure have actually changed
            pos = nao.ball_pos_lr
            vel = nao.ball_vel_lr
            ang = nao.ball_ang_lr
            ball_values = (pos, vel, ang)
        if abs(vel) < 0.05:  # Stop flickering between two bins
            vel = 0

        p, v, a = nf.inputs_to_bins(ball_values, max_values, bin_values)
        p = int(p)
        v = int(v)
        a = int(a)

        action = nf.get_action(q_mat, p, v, a, bin_values[0], bin_values[1], consensus=False)
        move = 0
        if action == 1:  # Anticlockwise
            move = 1
        elif action == 0:  # Clockwise
            move = -1

        if prnt:
            print "State:", p, v, a, "q_mat:", q_mat[p, v, a], "Action:", move

        ang_val += move
        if ang_val < 0:
            ang_val = 0
        if ang_val >= nao.num_angles:
            ang_val = nao.num_angles-1

        nao.interpolate_angles_relative_lr(move, bin_values[2], 2, 7)
        nao.ball_ang_lr = ang_val
        nao.go_to_interpolated_angles_lr()

        # Remember values for next loop
        prev_vals = (pos, vel, ang)
        prev_state = (p, v, a)

        time.sleep(wait_time)

        nao.record_current_state(pos, vel, ang, move)  # Record the data for plotting performance

    nao.hands_open()
    time.sleep(2)
    nao.rest()
    nf.save_q(q_mat, bin_values[0], bin_values[1], bin_values[2], 2, True)
    nao.save_state_data(1, bin_values[0], bin_values[1], bin_values[2], max_values[0], max_values[1], max_values[2], "Performance on the nao with q-matrix", nao.ip)


def collect_experiences(nao, update_experience=True, save_exp=False, wait_time=0.0, run_time=300, ball_update=0.01, delay=False):
    # Set up the nao
    nao.initial_setup()
    nao.set_up_to_hold_tray()  # move hands out in front
    nao.hands_open()  # Open hands
    raw_input("Press enter to continue: ")  # Wait so the user can position tray
    nao.hands_grab()  # Close the hands and grab the tray
    nao.continually_update_ball_information(ball_update)  # update the current recorded position of the ball every ball_update seconds
    
    bin_values = __get_bin_values_q()
    max_values = __get_max_values()
    nao.num_angles = bin_values[2]

    q_mat = nf.load_q_matrix(bin_values[0], bin_values[1], bin_values[2], act=2, is_nao=False, is_delay=delay)

    ang_val = nao.angle_lr_interpolation

    if update_experience:
        experiences = nf.load_exp(bin_values[0], bin_values[1], bin_values[2], 2)
    else:
        experiences = np.empty((bin_values[0], bin_values[1], bin_values[2], 1), dtype=object)  # Need dtype=object
        for i in range(len(experiences)):
            for j in range(len(experiences[i])):
                for k in range(len(experiences[i][j])):
                    experiences[i][j][k][0] = []

    experiences = nf.clean_experiences(experiences)

    prev_state = (-1, -1, -1)
    prev_vals = (-1, -1, -1)
    action = 0
    for i in range(0, run_time):
        pos = nao.ball_pos_lr  # Get the NN inputs based off the state of the ball and tray
        vel = nao.ball_vel_lr
        ang = nao.ball_ang_lr
        ball_values = (pos, vel, ang)

        while ball_values == prev_vals:  # Make sure have actually changed
            pos = nao.ball_pos_lr
            vel = nao.ball_vel_lr
            ang = nao.ball_ang_lr
            ball_values = (pos, vel, ang)
        if abs(vel) < 0.05:  # Stop flickering between two bins
            vel = 0

        p, v, a = nf.inputs_to_bins(ball_values, max_values, bin_values)
        p = int(p)
        v = int(v)
        a = int(a)

        print "State:", p, v, a, q_mat[p, v, a], vel

        reward = nf.get_reward(p, v, bin_values[0], bin_values[1]) * 5

        if i > 1:
            curr_state = (p, v, a)
            exp = {
                    "new_state": curr_state,
                    "action": action,
                    "reward": reward
                  }
            print(len(experiences[prev_state[0], prev_state[1], prev_state[2], 0]))
            if len(experiences[prev_state[0], prev_state[1], prev_state[2], 0]) < 20:
                experiences[prev_state[0], prev_state[1], prev_state[2], 0] = np.append(experiences[prev_state[0], prev_state[1], prev_state[2], 0], exp)
                print(experiences[prev_state[0], prev_state[1], prev_state[2]][0])

        if random.random() < 0.2:  # Only update action 1/2 of the time - allows for series of actions
            action = nf.get_action(q_mat, p, v, int(a), bin_values[0], bin_values[1], consensus=False)
            #action = np.argmax(q_mat[p][v][int(a)])
            if random.random() < 0.2:
                action = (action + 1)%2

        move = 0
        if action == 1:  # Anticlockwise
            move = 1
        elif action == 0:  # Clockwise
            move = -1

        print "Action", move

        ang_val += move
        if ang_val < 0:
            ang_val = 0
        if ang_val >= nao.num_angles:
            ang_val = nao.num_angles-1

        nao.interpolate_angles_relative_lr(move, bin_values[2], 2, 7)  # 2, bin_values[2]-3)
        nao.ball_ang_lr = ang_val
        nao.go_to_interpolated_angles_lr()

        # Remember values for next loop
        prev_vals = (pos, vel, a)
        prev_state = (p, v, int(a))

        time.sleep(wait_time)

        nao.record_current_state(pos, vel, a, move)  # Record the data for plotting performance

    nao.hands_open()
    time.sleep(2)
    nao.rest()
    nf.save_q(q_mat, bin_values[0], bin_values[1], bin_values[2], 2, True)
    nao.save_state_data(1, bin_values[0], bin_values[1], bin_values[2], max_values[0], max_values[1], max_values[2], "Performance on the nao with q-matrix", nao.ip)
    if save_exp:
        sumup = 0
        nf.save_exp(experiences, bin_values[0], bin_values[1], bin_values[2], max_values[0], max_values[1], max_values[2], 2)
        for i in range(0, len(experiences)):
            for j in range(0, len(experiences[i])):
                for k in range(0, len(experiences[i][j])):
                    print i, j, k, experiences[i][j][k][0]
                    sumup += len(experiences[i][j][k][0])
        print "num experiences:", sumup


def balance_ball_nn(nao, only_two_actions=True, qnn=False, trained_on_nao=False, wait_time=0.0, run_time=300, ball_update=0.01):
    """
    Balance the ball on the nao robot using the neural network to make decisions
    only_two_actions - Only use the two actions of tilt left and tilt right
    qnn - Update the network as training goes on
    """
    bin_values = __get_bin_values_nn()
    max_values = __get_max_values()
    nao.num_angles = bin_values[2]

    nao.initial_setup()
    nao.set_up_to_hold_tray()
    nao.hands_open()
    raw_input("Press enter to continue: ")
    nao.hands_grab()
    nao.continually_update_ball_information(ball_update)
    net = nf.load_network(trained_on_nao)

    #max_values = (0.15, 0.3, 0.1)  # The maximum possible values for each of the 3 properties
    #bin_values = (10, 8, 10)  # The number of bins of each value

    action = -1
    nn_out = [0, 0, 0]
    inputs = (0, 0, 0)

    for i in range(0, run_time):
        pos = nao.ball_pos_lr  # Get the NN inputs based off the state of the ball and tray
        vel = nao.ball_vel_lr
        ang = nao.ball_ang_lr
        if abs(vel) < 0.05:  # Stop flickering between two bins
            vel = 0
        ball_values = (pos, vel, ang)

        p, v, a = nf.inputs_to_bins(ball_values, max_values, bin_values)
        prev_inputs = copy.copy(inputs)  # inputs still has the value from the previous action
        inputs = (p, v, a)  # ang not a because it is already in the bin form

        if action >= 0 and qnn:
            reward = nf.get_reward_nn_specific(inputs, prev_inputs, bin_values)  # Reward for the current state
            future_action_value = nf.get_nn_output(net, inputs)  # Outputs for acting from current state
            # nn_out holds the output for the previous state
            # prev_inputs holds the previous state
            # action holds the action taken from the previous state to get to current state
            # reward is reward for being in current state
            if reward != 0:  # Only update if it does something
                print "Curr state:", inputs, " Prev state:", prev_inputs
                print "Prev output:", nf.get_nn_output(net, prev_inputs)
                net = nf.update_nn(net, prev_inputs, action, nn_out, reward, future_action_value, epochs=50, learn_rate_update_func=0.2)
                print "New output:", nf.get_nn_output(net, prev_inputs), "\n"

        nn_out = nf.get_nn_output(net, inputs)  # Feed to the NN
        action = np.argmax(nn_out)
        nao.record_current_state(pos, vel, ang, action)  # Record the data for plotting performance

        move = 0
        if action == 0:  # [1, 0, -1]
            move = 1
        elif action == 2:
            move = -1
        elif only_two_actions:
            if nn_out[0] > nn_out[2]:
                move = 1
            else:
                move = -1
        nao.interpolate_angles_relative_lr(move, 10, 2, 7)
        nao.go_to_interpolated_angles_lr()

        print ball_values, inputs, move, "\n"
        time.sleep(wait_time)
    nf.save_network(net)
    nao.save_state_data(0, bin_values[0], bin_values[1], bin_values[2], max_values[0], max_values[1], max_values[2], "Performance on the nao with nn", nao.ip, two_act=only_two_actions, qnn=qnn)
    nao.hands_open()
    time.sleep(2)
    nao.rest()


def balance_ball_input(nao, append_q=True):
    bin_values = __get_bin_values_nn()
    max_values = __get_max_values()
    nao.num_angles = bin_values[2]

    nao.initial_setup()
    nao.set_up_to_hold_tray()
    nao.hands_open()
    raw_input("Press enter to continue: ")
    nao.hands_grab()
    nao.continually_update_ball_information(0.05)

    balance = True

    training_data_json = []
    filename = str(bin_values[0]) + "_" + str(bin_values[1]) + "_" + str(bin_values[2])
    file_loc = config["other"]["nn_train_data"] + "/training_data_nao_" + filename + ".json"

    if append_q:  # If data is already existing for the current parameters, update that.
        wd = os.getcwd()
        wd = wd.replace("\\", "/")  # Since slash directions are different
        if os.path.exists(file_loc):
            print("UPDATE EXISTING DATA")
            load_json = open(file_loc, "r")
            training_data_json = json.load(load_json)["data"]
        else:  # Data set does not exist for current parameters
            print("NEW DATA SET")
            training_data_json = []
    else:
        training_data_json = []

    while balance:
        move = getch()
        action = 0
        output = [0, 0, 0]
        record = True
        if move == 'k':
            action = 1
            output = [1, 0, 0]
        elif move == 'l':
            action = -1
            output = [0, 0, 1]
        elif move == ' ':
            balance = False
        elif move == 'o':
            action = 1
            record = False
        elif move == 'p':
            action = -1
            record = False
        nao.interpolate_angles_relative_lr(action, 10)#, 2, 7)
        nao.go_to_interpolated_angles_lr()
        p = nao.ball_pos_lr
        v = nao.ball_vel_lr
        a = nao.ball_ang_lr
        if balance and record:    
            new_data = {"pos": p, "vel": v, "ang": a, "out1": output[0], "out2": output[1], "out3": output[2]}
            training_data_json.append(new_data)
        p, v, a = nf.inputs_to_bins((p, v, a), max_values, bin_values)
        print p, v, a
    nao.hands_open()
    time.sleep(2)
    nao.rest()

    json_output = open(file_loc, 'w')
    json_data = {
                    "metadata": [
                        {
                            "num_pos": bin_values[0],
                            "num_vel": bin_values[1],
                            "num_ang": bin_values[2],
                            "max_pos": max_values[0],
                            "max_vel": max_values[1],
                            "max_ang": max_values[2]
                        }
                    ],
                    "data": training_data_json,
    }
    json.dump(json_data, json_output)
