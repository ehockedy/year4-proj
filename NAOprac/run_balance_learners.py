
import learnToBalance as q
import train as wal
import trainNN as nn


option = 3

if option == 1:
    trainer = q.setup_q_trainer()
    q.do_q_learning(trainer, train=False, load_q=True, save_q=False)
elif option == 2:
    trainer = wal.setup_wal_trainer()
    #wal.do_watch_and_learn_training(trainer, 50000, save_q=True)
    wal.do_watch_and_learn_evaluation(trainer, 5)
elif option == 3:
    trainer = nn.setup_nn(num_vel=8, num_ang=10)
    nn.generate_data_nn(trainer, append_q=True, save_q=True)
    nn.train_nn(trainer, structure=(2,), train_time=200)
    nn.evaluate_nn(trainer, number_of_trials=200, iteration_limit=200, action_threshold=0.5, draw_output=True, draw_speed=120)
