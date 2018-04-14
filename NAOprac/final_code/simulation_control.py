import simulation_nn as nn
import simulation_q as q


def run_q(option):
    """
    Options:
     - 0 - no delay, general reward, no ER
     - 1 - no delay, specific reward, no ER
     - 2 - delay, general, no ER
     - 3 - no delay, general, ER
    """
    if option == 0:
        trainer = q.setup_trainer(12, 12, 10, 50000)
        # tr.generate_cell_data()
        q.train_q(trainer, er=False, specific=False)
        q.run_after_trained(trainer)
        trainer.save_state_data("General, no delay, no ER")
        # trainer.save_q_cell_data("")
        q.display_simulation(trainer)
        trainer.save_q(er=False, delay=False)
    elif option == 1:
        trainer = q.setup_trainer(12, 12, 10, 50000)
        q.train_q(trainer, er=False, specific=True)
        q.run_after_trained(trainer)
        trainer.save_state_data("Specific, no delay, no ER")
        q.display_simulation(trainer)
        trainer.save_q(er=False, delay=False)
    elif option == 2:
        trainer = q.setup_trainer(12, 12, 10, 50000, step_size=5, sim_speed=10)
        q.train_q(trainer, er=False, specific=False)
        q.run_after_trained(trainer)
        trainer.save_state_data("General, delay, no ER")
        q.display_simulation(trainer)
        trainer.save_q(er=False, delay=True)
    elif option == 3:
        trainer = q.setup_trainer(12, 12, 10, 50000)
        q.train_q(trainer, er=False, specific=False)
        trainer.iterations = 1
        trainer.max_num_iterations = 10000
        q.train_q(trainer, er=True)
        q.run_after_trained(trainer)
        trainer.save_state_data("General, no delay, ER")
        q.display_simulation(trainer)
        trainer.save_q(er=True, delay=False)


def run_nn():
    trainer = nn.setup_nn(nao_data=True)
    # nn.generate_data(trainer, append_q=True, save_q=True)
    # net = nn.load_network(trainer)

    # Train a network of the given structure for the given number of epochs.
    # The network is saved to file
    nn.train_nn(trainer, structure=(2, 2,), train_time=100)

    # Let the simulation run using the trained network to choose actions
    nn.evaluate_nn(trainer, number_of_trials=3, iteration_limit=2000,
                   draw_output=True, draw_speed=120, record_data=True,
                   two_acts=True)


#run_q(1)
run_nn()
