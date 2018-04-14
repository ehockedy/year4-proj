import os
import json
import matplotlib.pyplot as plt
import configparser

config = configparser.ConfigParser()
config.read('config.ini')


def plot_position_against_time(path_key, yrange, i="", sf=1, end=False,
                               end_val=5000):
    """
    Plot data recorded during training or running of one of the algorithms
    Options for path_key are:
     - sim_q
     - sim_nn
     - nao_q
     - nao_nn

    yrange (sim) = 125\n
    yrange (nao) = 0.15

    sf is the scale factor to apply to the y axis. This keeps the labels
    at the values given, but scales the actual data by the given amount

    end is if you want to show only the last end_val iterations
    """
    path = config["evaluation_data_paths"][path_key]
    if i == "":  # Do most recent by default
        cwd = os.getcwd() + "/" + path  # Directory of data to show
        dirs = os.listdir(cwd)  # List of files in that directory
        i = len(dirs)-1  # Number of files
    print("Showing data file", i)

    filename = "/" + config["data_file_prefix"][path_key] + "_"

    data_file = open(path + filename + str(i) + ".json", 'r')
    data_json = json.load(data_file)
    data = data_json["data"]
    x_time = []
    y_pos = []
    # metadata = data_json["metadata"]
    # plt.title(metadata[0]["description"])
    font = {"fontname": "Times New Roman"}
    plt.xlabel("Iterations", font)
    plt.ylabel("Ball position", font)
    ax = plt.gca()
    for tick in ax.get_xticklabels():
        tick.set_fontname("Times New Roman")
    for tick in ax.get_yticklabels():
        tick.set_fontname("Times New Roman")
    if end:
        for idx in range(len(data)-end_val, len(data)):
            x_time.append(idx)
            y_pos.append(data[idx]["pos"]*sf)
    else:
        for idx in range(0, len(data)):
            x_time.append(idx)
            y_pos.append(data[idx]["pos"]*sf)
    axs = plt.gca()
    axs.set_ylim([-yrange, yrange])
    plt.plot(x_time, y_pos)

    plt.show(1)

for i in range(31, 33):
    plot_position_against_time("nao_q", 0.15, i,end=False)  # , sf=115/200)
