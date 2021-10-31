import json
import matplotlib.pyplot as plt
from clustering import *
if __name__ == '__main__':

    # Load trajectories
    trajectories = None
    model_name = 'double_jump_labyrinth'
    with open("../arrays/{}.json".format("{}_trajectories".format(model_name))) as f:
        trajectories = json.load(f)

    print('Clustering')
    print(cluster_trajectories(list(trajectories.values())))

