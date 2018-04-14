import nao_options as nao_opt

#nao_ip = "192.168.1.20"
nao_ip = "192.168.1.153"  # Clank, much better for NN


def get_instructions(ip):
    nao = nao_opt.setup_nao(ip)
    continue_running = True
    while continue_running:
        instruction = raw_input("Pick an instruction: ")
        if instruction == "exit":
            continue_running = False
        elif instruction == "rest":
            nao.rest()
        elif instruction == "nn":
            nao_opt.balance_ball_nn(nao, only_two_actions=True, trained_on_nao=True)
        elif instruction == "q":
            nao_opt.balance_ball_q_mat(nao)
        elif instruction == "er":
            nao_opt.balance_ball_q_mat(nao, iser=True)
        elif instruction == "qd":
            nao_opt.balance_ball_q_mat(nao, delay=True, prnt=True)
        elif instruction == "qnn":
            nao_opt.balance_ball_nn(nao, qnn=True, ball_update=0.1)
        elif instruction == "ctrl":
            nao_opt.balance_ball_input(nao, append_q=True)
        elif instruction == "reset":
            nao_opt.nf.reset_network()
        elif instruction == "exp":
            nao_opt.collect_experiences(nao, update_experience=True, save_exp=True)
        elif instruction == "open":
            nao.hands_open()
        elif instruction == "grab":
            nao.hands_grab()


get_instructions(nao_ip)
