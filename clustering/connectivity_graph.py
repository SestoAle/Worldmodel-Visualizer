import networkx as nx
from utils import NumpyEncoder

def create_connectivirty_graph(graph: nx.Graph, trajectories:dict,
                               x_min=-50, x_max=50, z_min=-65, z_max=65, y_min=1, y_max=40,
                               traj_to_observe=1000):
    '''
    Create a connectivity graph from dict of trajectories.
    '''
    all_traj = len(trajectories.values())

    for traj in trajectories.values():
        for idx in range(len(traj) - 1):
            # Normalize point of the trajectory for better graph representation
            point_a = (
               int(((traj[idx][0] + 1) / 2) * (x_max - x_min) + x_min),
               int(((traj[idx][1] + 1) / 2) * (z_max - z_min) + z_min),
               int(((traj[idx][2] + 1) / 2) * (y_max - y_min) + y_min)
            )
            point_b = (
                int(((traj[idx + 1][0] + 1) / 2) * (x_max - x_min) + x_min),
                int(((traj[idx + 1][1] + 1) / 2) * (z_max - z_min) + z_min),
                int(((traj[idx + 1][2] + 1) / 2) * (y_max - y_min) + y_min)
            )

            if point_a == point_b:
                continue
            graph.add_edge(point_a, point_b)


if __name__ == '__main__':
    import json
    import numpy as np

    model_name = "double_jump_impossibru_5"

    connectivity_graph = nx.DiGraph()

    trajectories = None
    try:
        with open("../arrays/{}.json".format("{}_trajectories".format(model_name))) as f:
            trajectories = json.load(f)
            print("Trajectories loaded!")

    except Exception as e:
        print("I could not load the trajectories!")
        print(e)

    create_connectivirty_graph(connectivity_graph, trajectories)
    print('Graph created!')
    print('Saving connectiviy graph for visualizing it in Unity')
    print(print(np.shape(np.asarray(connectivity_graph.nodes()))))
    nodes = np.asarray(connectivity_graph.nodes())
    graph_to_save = dict(x=nodes[:, 0], z=nodes[:, 1], y=nodes[:, 2])
    json_str = json.dumps(graph_to_save, cls=NumpyEncoder)
    f = open("../../OpenWorldEnv/OpenWorld/Assets/Resources/graph.json".format(model_name), "w")
    f.write(json_str)
    f.close()
    print('Graph Saved!')
    input('....')
    paths = nx.all_simple_paths(connectivity_graph, (0, 0, 1), (32, 52, 16), cutoff=50)
    print("Paths found!")

    for path in paths:
        path = np.asarray(path)
        traj_to_save = dict(x_s=path[:, 0], z_s=path[:, 1], y_s=path[:, 2], im_values=np.zeros(29), il_values=np.zeros(29))
        json_str = json.dumps(traj_to_save, cls=NumpyEncoder)
        f = open("../../OpenWorldEnv/OpenWorld/Assets/Resources/traj.json".format(model_name), "w")
        f.write(json_str)
        f.close()
        input('...')