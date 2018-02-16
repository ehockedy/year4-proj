import nao_functions as nf
import time
import numpy as np
import random
from msvcrt import getch

max_values = (0.13, 2, 0.004)  # The maximum possible values for each of the 3 properties
bin_values = (10, 6, 6)  # The number of bins of each value
learn = False

nao = nf.Nao("192.168.1.20", bin_values[2])
#nao = nf.Nao("192.168.1.153", bin_values[2])


def balance_ball_nn():
    nao.initial_setup()
    nao.set_up_to_hold_tray()
    nao.hands_open()
    raw_input("Press enter to continue: ")
    nao.hands_grab()
    nao.continually_update_ball_information(0.01)
    net = nf.load_network()

    wait_time = 0.0
    if wait_time > 0:
        run_time = int(5 / wait_time)
    else:
        run_time = 1000

    action = -1
    nn_out = [0, 0, 0]

    for i in range(0, run_time):
        pos = nao.ball_pos_lr  # Get the NN inputs based off the state of the ball and tray
        vel = nao.ball_vel_lr
        ang = nao.ball_ang_lr
        ball_values = (pos, vel, ang)
        max_values = (0.15, 0.3, 0.1)  # The maximum possible values for each of the 3 properties
        bin_values = (10, 8, 10)  # The number of bins of each value
        p, v, a = nf.inputs_to_bins(ball_values, max_values, bin_values)
        inputs = (p, v, ang)  # ang not a because it is already in the bin form

        if action >= 0:
            reward = nf.get_reward_nn(inputs, max_values)
            net = nf.update_nn(net, inputs, action, nn_out, reward)

        nn_out = nf.get_nn_output(net, inputs)  # Feed to the NN
        action = np.argmax(nn_out)

        move = 0
        if action == 0:  # [1, 0, -1]
            move = 1
        elif action == 2:
            move = -1
        nao.interpolate_angles_relative_lr(move, 10, 2, 7)
        nao.go_to_interpolated_angles_lr()

        print ball_values, inputs, move
        time.sleep(wait_time)
    nao.hands_open()
    time.sleep(2)
    nao.rest()


def balance_ball_q_mat():
    nao.initial_setup()

    nao.set_up_to_hold_tray()
    nao.hands_open()
    raw_input("Press enter to continue: ")
    nao.hands_grab()
    nao.continually_update_ball_information(0.01)

    q_mat = nf.load_q_matrix(bin_values[0], bin_values[1], bin_values[2], 2, False)
    #q_mat = nf.normalise_whole_q(q_mat)

    explore_rate = 0
    if learn:
        explore_rate = 0.5

    wait_time = 0.0
    if wait_time > 0:
        run_time = int(20 / wait_time)
    else:
        run_time = 500

    ang_val = nao.angle_lr_interpolation

    prev_state = (-1, -1, -1)
    prev_vals = (-1, -1, -1)
    action = -1
    for i in range(0, run_time):
        pos = nao.ball_pos_lr  # Get the NN inputs based off the state of the ball and tray
        vel = nao.ball_vel_lr
        ang = nao.ball_ang_lr
        ball_values = (pos, vel, ang)

        while ball_values == prev_vals:  # Make sure have actually changed
            print("SAME")
            pos = nao.ball_pos_lr
            vel = nao.ball_vel_lr
            ang = nao.ball_ang_lr
            ball_values = (pos, vel, ang)

        p, v, _ = nf.inputs_to_bins(ball_values, max_values, bin_values)

        print "State:", int(p), int(v), ang

        # Update Q matrix now that we're in new state
        if learn and i > 1:  # prev_state != (-1, -1, -1):  # True because do want to learn
            curr_state = (int(p), int(v), int(ang))
            reward = nf.get_reward(p, v, bin_values[0], bin_values[1])
            # Action is remembered from previous iteration
            q_mat = nf.update_q(q_mat, prev_state, curr_state, action, reward, 0.3, 0.99, prnt=True)
            print "Prev:", prev_state, "Curr:", curr_state, "Act:", action, "Rew", reward, "\n\n"

        action = np.argmax(q_mat[int(p)][int(v)][int(ang)])
        if learn and random.random() < explore_rate:
            action = random.randint(0, 1)
            print("RANDOM")

        move = 0
        if action == 1:
            move = 1
        elif action == 0:
            move = -1

        print "Action", move

        ang_val += move
        if ang_val < 0:
            ang_val = 0
        if ang_val >= nao.num_angles:
            ang_val = nao.num_angles-1

        #nao.interpolate_angles_relative_lr(move, bin_values[2], 0, 0)  # 2, bin_values[2]-3)
        nao.interpolate_angles_fixed_lr(ang_val, bin_values[2]-1)
        nao.ball_ang_lr = ang_val
        nao.go_to_interpolated_angles_lr()

        #print "\nOld state:", int(p), int(v), int(ang), "\nAction:", move

        # Remember values for next loop
        prev_vals = (pos, vel, ang)
        prev_state = (int(p), int(v), int(ang))

        #time.sleep(wait_time)

        # prev_state = (int(p), int(v), int(nao.ball_ang_lr))
        # prev_vals = (pos, vel, nao.ball_ang_lr)
        # new_ball_values = (nao.ball_pos_lr, nao.ball_vel_lr, nao.ball_ang_lr)
        # p2, v2, _ = nf.inputs_to_bins(new_ball_values, max_values, bin_values)
        # curr_state = (int(p2), int(v2), int(new_ball_values[2]))
        # # Update the Q matrix
        # if learn and i > 1:  # prev_state != (-1, -1, -1):  # True because do want to learn
        #     reward = nf.get_reward(p, v, bin_values[0], bin_values[1])
        #     q_mat = nf.update_q(q_mat, prev_state, curr_state, action, reward, 0.3, 0.99, prnt=True)
        #     print "Prev:", prev_state, "Curr:", curr_state, "Act:", action, "Rew", reward, "\n\n"
            #print "\n\n", ang, ang_val, reward
        #prev_state = curr_state

        freq = run_time/4
        if i % freq == freq-1:
            explore_rate /= 2

    nao.hands_open()
    time.sleep(2)
    nao.rest()
    nf.save_q(q_mat, bin_values[0], bin_values[1], bin_values[2], 2)


def balance_ball_input():
    nao.initial_setup()
    nao.set_up_to_hold_tray()
    nao.hands_open()
    raw_input("Press enter to continue: ")
    nao.hands_grab()
    # time.sleep(10)
    nao.continually_update_ball_information(0.01)
    balance = True
    while balance:
        move = getch()
        action = 0
        if move == 'k':
            action = 1
        elif move == 'l':
            action = -1
        elif move == ' ':
            balance = False
        nao.interpolate_angles_relative_lr(action, 10, 2, 7)
        nao.go_to_interpolated_angles_lr()
        print nao.get_tray_angle()
    nao.hands_open()
    time.sleep(2)
    nao.rest()


def get_instructions():
    continue_running = True
    while continue_running:
        instruction = raw_input("Pick an instruction: ")
        if instruction == "track":
            nao.continually_update_ball_information(0.05)
        elif instruction == "info":
            for i in range(0, 5000):
                print "pos:", nao.ball_pos_lr, "vel:", nao.ball_vel_lr, "ang:", nao.ball_ang_lr
        elif instruction == "exit":
            continue_running = False
        elif instruction == "rest":
            nao.rest()
        elif instruction == "nn":
            balance_ball_nn()
        elif instruction == "q":
            balance_ball_q_mat()
        elif instruction == "move":
            nao.manipulate_limbs(0.1)
        elif instruction == "stand":
            nao.stand(0.5)
        elif instruction == "inter1":
            nao.interpolate_angles_fixed_lr(0, 10)
            nao.go_to_interpolated_angles_lr(0.5)
        elif instruction == "open":
            nao.hands_open()
        elif instruction == "grab":
            nao.hands_grab()
        elif instruction == "tilt":
            balance_ball_input()
        elif instruction == "angs":
            nao.set_up_to_hold_tray()
            nao.hands_open()
            raw_input("Press enter to continue: ")
            nao.hands_grab()
            for i in range(0, bin_values[2]):
                nao.interpolate_angles_fixed_lr(i, bin_values[2])
                nao.go_to_interpolated_angles_lr()
                print(i)
                time.sleep(1)
        elif instruction == "rec":
            nao.set_up_to_hold_tray()
            nao.hands_open()
            raw_input("Press enter to continue: ")
            nao.hands_grab()
            nao.interpolate_angles_fixed_lr(10, bin_values[2])
            nao.go_to_interpolated_angles_lr()
            nao.record_angles()


get_instructions()


# TODO
# - Remove hard coded number of angles and interpolation value, and use config file more
# - Change so that uses actual tray angles in calculation, since it updates
#       the angel value upon doing rotation, hoever because of how it actually
#       moves the tray, means often doesn't acually get to move as is overwritten
#       by next angle
# 6/2 KINDA WORKS - there are hints at learning
# - Maybe keep an angle variable that stores the current angle that is updated based on the rotation. 
#   This may be better than updating each time, and just update when it can due to non-blocking


# Include lag - make sm more like real world - decrease framerate of sim