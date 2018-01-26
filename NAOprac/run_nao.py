import nao_functions as nf
import time
import numpy as np

nao = nf.Nao("169.254.232.52")


def balance_ball_nn():
    nao.initial_setup()
    nao.set_up_to_hold_tray()
    nao.hands_open()
    raw_input("Press enter to continue: ")
    nao.hands_grab()
    nao.continually_update_ball_information()
    net = nf.load_network()
    # prev_action = None
    for i in range(0, 500):
        pos = nao.ball_pos_lr  # Get the NN inputs based off the state of the ball and tray
        vel = nao.ball_vel_lr
        ang = nao.ball_ang_lr
        ball_values = (pos, vel, ang)
        max_values = (0.15, 0.3, 0.1)  # REFINE THESE
        bin_values = (10, 8, 10)
        p, v, a = nf.inputs_to_bins(ball_values, max_values, bin_values)
        inputs = (p, v, ang)  # ang not a because it is already in the bin form

        nn_out = nf.get_nn_output(net, inputs)  # Feed to the NN
        action = np.argmax(nn_out)

        # if abs(action) < 0.3:
        #    action = 0
        move = 0
        if action == 0:  # [1, 0, -1]
            move = 1
        elif action == 2:
            move = -1
        nao.interpolate_angles_relative_lr(move, 10, 2, 7)  # Perform the chosen output
        nao.go_to_interpolated_angles_lr()
        print ball_values, inputs, nn_out, action, move

        time.sleep(0.01)
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


get_instructions()


# TODO
# - Remove hard coded number of angles and interpolation value, and use config file more
