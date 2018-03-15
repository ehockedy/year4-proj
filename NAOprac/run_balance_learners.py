import trainQ as q
import train as wal
import trainNN as nn
import trainQNN as qnn


option = 5

if option == 1:
    trainer = q.setup_q_trainer(12, 12, 7, is_q_not_s=True)
    #trainer.load_q(delay=True)
    q.do_q_learning(trainer, train=True, prnt=False)
    #trainer.save_q(delay=True)
elif option == 2:
    trainer = wal.setup_wal_trainer()
    #wal.do_watch_and_learn_training(trainer, 50000, save_q=True)
    wal.do_watch_and_learn_evaluation(trainer, 5)
elif option == 3:
    trainer = nn.setup_nn(num_vel=8, num_ang=10)
    #nn.generate_data_nn(trainer, append_q=True, save_q=True)
    nn.train_nn(trainer, structure=(2,), train_time=200)
    nn.evaluate_nn(trainer, number_of_trials=200, iteration_limit=200, action_threshold=0.5, draw_output=True, draw_speed=120)
elif option == 4:
    trainer = qnn.setup_nn(num_vel=8, num_ang=10)
    #qnn.evaluate_nn(trainer, number_of_trials=500, iteration_limit=200,
    #                action_threshold=0.0, draw_output=False,
    #                learn_rate=0.2, discount_factor=0.99, 
    #                explore_rate=0.5, explore_rate_reduction=2, explore_rate_freq=50,
    #                train=True, prnt=False)

    net = qnn.load_network()
    qnn.evaluate_nn(trainer, number_of_trials=500, iteration_limit=200,
                    action_threshold=0.0, draw_output=False,
                    learn_rate=0.3, discount_factor=0.99, train=False, net=net)

    net = qnn.load_q_network()  # THIS LEADS TO NO NOTICABLE IMPROVEMENT
    qnn.evaluate_nn(trainer, number_of_trials=500, iteration_limit=200,
                    action_threshold=0.0, draw_output=False,
                    learn_rate=0.3, discount_factor=0.99, train=True, net=net)
elif option == 5:
    trainer = q.setup_q_trainer(12, 12, 10, ends=True)
    trainer.load_q(delay=True)
    #q.do_q_learning_sides(trainer, train=True, prnt=False)
    q.do_experience_replay(trainer, 1000000, True)
    trainer.save_q(delay=True, er=True)
