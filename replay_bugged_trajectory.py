import os
import tensorflow as tf
import argparse
import numpy as np
from mlagents.envs import UnityEnvironment
import logging as logs
import pickle
import json



os.environ["CUDA_VISIBLE_DEVICES"] = "1"
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


# Parse arguments for training
parser = argparse.ArgumentParser()
parser.add_argument('-gn', '--game-name', help="The name of the game", default=None)
parser.add_argument('-wk', '--work-id', help="Work id for parallel training", default=0)
parser.add_argument('-mt', '--max-timesteps', help="Max timestep per episode", default=80)

args = parser.parse_args()

eps = 1e-12

class BugEnvironment:

    def __init__(self, game_name, no_graphics, worker_id, max_episode_timesteps, pos_already_normed=True):
        self.no_graphics = no_graphics
        self.unity_env = UnityEnvironment(game_name, no_graphics=no_graphics, seed=worker_id, worker_id=worker_id)
        self._max_episode_timesteps = max_episode_timesteps
        self.default_brain = self.unity_env.brain_names[0]
        self.config = None
        self.actions_eps = 0.1
        self.previous_action = [0, 0]
        # Table where we save the position for intrisic reward and spawn position
        self.pos_buffer = dict()
        self.pos_already_normed = pos_already_normed
        self.r_max = 0.5
        self.max_counter = 500
        self.tau = 1 / 40
        self.standard_position = [14, 14, 1]
        self.coverage_of_points = []

        # Dict to store the trajectories at each episode
        self.trajectories_for_episode = dict()
        # Dict to store the actions at each episode
        self.actions_for_episode = dict()
        self.episode = -1

    def execute(self, actions):
        #actions = int(input(': '))

        env_info = self.unity_env.step([actions])[self.default_brain]
        reward = env_info.rewards[0]
        done = env_info.local_done[0]

        self.actions_for_episode[self.episode].append(actions)

        self.previous_action = actions

        state = dict(global_in=env_info.vector_observations[0])

        # Get the agent position from the state to compute reward
        position = state['global_in'][:3]
        self.trajectories_for_episode[self.episode].append(np.concatenate([position, state['global_in'][-2:]]))

        return state, done, reward

    def reset(self):

        self.previous_action = [0, 0]
        logs.getLogger("mlagents.envs").setLevel(logs.WARNING)
        self.coverage_of_points.append(len(env.pos_buffer.keys()))
        self.episode += 1
        self.trajectories_for_episode[self.episode] = []
        self.actions_for_episode[self.episode] = []
        # self.set_spawn_position()

        env_info = self.unity_env.reset(train_mode=True, config=self.config)[self.default_brain]
        state = dict(global_in=env_info.vector_observations[0])
        position = state['global_in'][:3]
        self.trajectories_for_episode[self.episode].append(np.concatenate([position, state['global_in'][-2:]]))
        # print(np.reshape(state['global_in'][7:7 + 225], [15, 15]))
        return state

    def entropy(self, probs):
        entr = 0
        for p in probs:
            entr += (p * np.log(p))
        return -entr

    def set_config(self, config):
        self.config = config

    def close(self):
        self.unity_env.close()

    def multidimensional_shifting(self, num_samples, sample_size, elements, probabilities):
        # replicate probabilities as many times as `num_samples`
        replicated_probabilities = np.tile(probabilities, (num_samples, 1))
        # get random shifting numbers & scale them correctly
        random_shifts = np.random.random(replicated_probabilities.shape)
        random_shifts /= random_shifts.sum(axis=1)[:, np.newaxis]
        # shift by numbers & find largest (by finding the smallest of the negative)
        shifted_probabilities = random_shifts - replicated_probabilities
        return np.argpartition(shifted_probabilities, sample_size, axis=1)[:, :sample_size]

    # Spawn a position from the buffer
    # If the buffer is empty, spawn at standard position
    def set_spawn_position(self):
        values = self.pos_buffer.values()
        if len(values) > 0:
            values = np.asarray(list(values))

            probs = 1 / values

            for i in range(len(values)):
                pos_key = list(self.pos_buffer.keys())[i]
                pos = np.asarray(list(map(float, pos_key.split(" "))))
                if pos[2] == 0:
                    probs[i] = 0

            probs = probs / np.sum(probs)

            index = self.multidimensional_shifting(1, 1, np.arange(len(probs)), probs)[0][0]

            pos_key = list(self.pos_buffer.keys())[index]
            pos = np.asarray(list(map(float, pos_key.split(" "))))
            # re-normalize pos to world coordinates
            pos = (((pos + 1) / 2) * 40) - 20

            self.config['agent_spawn_x'] = pos[0]
            self.config['agent_spawn_z'] = pos[1]
        else:
            self.config['agent_spawn_x'] = self.standard_position[0]
            self.config['agent_spawn_z'] = self.standard_position[1]

    # Insert to the table. Position must be a 2 element vector
    # Return the counter of that position
    def insert_to_pos_table(self, position):

        # Check if the position is already in the buffer
        for k in self.pos_buffer.keys():
            # If position - k < tau, then the position is already in the buffer
            # Add its counter to one and return it

            # The position are already normalized by the environment
            k_value = list(map(float, k.split(" ")))
            k_value = np.asarray(k_value)
            position = np.asarray(position)

            distance = np.linalg.norm(k_value - position)
            if distance < self.tau:
                self.pos_buffer[k] += 1
                return self.pos_buffer[k]

        pos_key = ' '.join(map(str, position))
        self.pos_buffer[pos_key] = 1
        return self.pos_buffer[pos_key]

    # Compute the intrinsic reward based on the counter
    def compute_intrinsic_reward(self, counter):
        return self.r_max * (1 - (counter / self.max_counter))


if __name__ == "__main__":
    game_name = args.game_name
    work_id = int(args.work_id)
    max_episode_timestep = int(args.max_timesteps)

    # Open the environment with all the desired flags
    env = BugEnvironment(game_name=game_name, no_graphics=False, worker_id=work_id,
                         max_episode_timesteps=max_episode_timestep)

    # Set starting position
    env.set_config(dict(agent_spawn_x=0, agent_spawn_z=0))

    # Load the actions to execute
    with open('arrays/actions.json', 'rb') as f:
        actions = json.load(f)['actions']

    # The first reset does not work, I dont know why.
    # This is a workaround
    env.reset()
    env.execute(1)

    while True:
        # Reset
        env.reset()

        # Execute the saved actions
        for a in actions:
            env.execute(a)
