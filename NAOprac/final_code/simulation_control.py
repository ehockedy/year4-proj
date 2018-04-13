import simulation_nn as nn
import simulation_q as q


def run_q():
    trainer = q.setup_trainer(12, 12, 10, 50000)
    # tr.generate_cell_data()
    q.train_q(trainer, er=False, specific=False)
    q.run_after_trained(trainer)
    trainer.save_state_data("")
    # trainer.save_q_cell_data("")
    q.display_simulation(trainer)
    # q.save_q(er=False, delay=False)


def run_nn():
    trainer = nn.setup_nn()
    # nn.generate_data(trainer, append_q=True, save_q=True)
    # net = nn.load_network(trainer)

    # Train a network of the given structure for the given number of epochs.
    # The network is saved to file
    nn.train_nn(trainer, structure=(2, 2,), train_time=100)

    # Let the simulation run using the trained network to choose actions
    nn.evaluate_nn(trainer, number_of_trials=3, iteration_limit=2000,
                   draw_output=True, draw_speed=120, record_data=True,
                   two_acts=True)


run_q()
# run_nn()
