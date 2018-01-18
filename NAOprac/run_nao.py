import nao_functions as nf
import trainNN as nn

nao = nf.Nao("IP")


def balance_ball_nn():
    nao.initial_setup()
    nao.set_up_to_hold_tray()
    input("Press enter to continue")
    nao.hands_grab()
    nao.continually_update_ball_information()
    net = nn.load_network()
    for i in range(0, 100):
        pos = nao.ball_pos_fb  # Get the NN inputs based off the state of the ball and tray
        vel = nao.ball_vel_lr
        ang = nao.ball_ang_lr
        inputs = (pos, vel, ang)

        action = nn.get_nn_output(net, inputs)  # Feed to the NN

        nao.interpolate_angles_relative_lr(action)  # Perform the chosen output
    # nao.hands_open()
    # nao.rest()


nf.load_network()

