from layers.layers import *

def input_spec():
    input_length = 1031
    global_state = tf.compat.v1.placeholder(tf.float32, [None, input_length], name='state')

    return [global_state]

def obs_to_state(obs):
    global_batch = np.stack([np.asarray(state['global_in']) for state in obs])
    return [global_batch]

def network_spec(states):
    global_state = states[0]

    agent_plane_x, agent_plane_z, agent_jump, is_grounded, depth_buf, goal_weight = \
        tf.split(global_state, [1, 1, 1, 1, 1024, 3], axis=1)

    agent_plane_x = agent_plane_x * 2816
    agent_plane_x = tf.cast(agent_plane_x, tf.int32)
    agent_plane_x = positional_encoding(agent_plane_x, 32)

    agent_plane_z = agent_plane_z * 2752
    agent_plane_z = tf.cast(agent_plane_z, tf.int32)
    agent_plane_z = positional_encoding(agent_plane_z, 32)

    # agent_plane_x = ((agent_plane_x + 1) / 2) * 1600
    # agent_plane_z = ((agent_plane_z + 1) / 2) * 1150

    agent_jump = agent_jump * 100
    agent_jump = tf.cast(agent_jump, tf.int32)
    agent_jump = positional_encoding(agent_jump, 32)

    agent = tf.concat([agent_plane_x, agent_plane_z, agent_jump], axis=1)
    agent = tf.reshape(agent, (-1, 3 * 32))
    # agent = tf.concat([agent, is_grounded, can_double_jump], axis=1)
    agent = linear(agent, 1024, name='global_embs', activation=tf.nn.relu)
    is_grounded = linear(is_grounded, 1024, name='grounded_embs', activation=tf.nn.relu)
    agent = tf.concat([agent, is_grounded], axis=1)

    goal_weight = linear(goal_weight, 1024, name='goal_embs', activation=tf.nn.relu)

    depth_buf = tf.reshape(depth_buf, [-1, 32, 32, 1])
    depth_buf = conv_layer_2d(depth_buf, 32, [3, 3], strides=(2, 2), name='conv1', activation=tf.nn.relu)
    depth_buf = conv_layer_2d(depth_buf, 64, [3, 3], strides=(2, 2), name='conv2', activation=tf.nn.relu)
    depth_buf = tf.reshape(depth_buf, [-1, 8 * 8 * 64])


    global_state = tf.concat([agent, depth_buf, goal_weight], axis=1)

    global_state = linear(global_state, 1024, name='embs', activation=tf.nn.relu)

    return global_state

def input_spec_rnd():
    input_length = 1031
    global_state = tf.compat.v1.placeholder(tf.float32, [None, input_length], name='state')

    return [global_state]

def obs_to_state_rnd(obs):
    global_batch = np.stack([state['global_in'] for state in obs])
    return [global_batch]

def network_spec_rnd_predictor(states):
    global_state = states[0]

    agent_plane_x, agent_plane_z, agent_jump, is_grounded, depth_buf, goal_weight = \
        tf.split(global_state, [1, 1, 1, 1, 1024, 3], axis=1)

    agent_plane_x = agent_plane_x * 2816
    agent_plane_x = tf.cast(agent_plane_x, tf.int32)
    agent_plane_x = positional_encoding(agent_plane_x, 32)

    agent_plane_z = agent_plane_z * 2752
    agent_plane_z = tf.cast(agent_plane_z, tf.int32)
    agent_plane_z = positional_encoding(agent_plane_z, 32)

    # agent_plane_x = ((agent_plane_x + 1) / 2) * 1600
    # agent_plane_z = ((agent_plane_z + 1) / 2) * 1150

    agent_jump = agent_jump * 100
    agent_jump = tf.cast(agent_jump, tf.int32)
    agent_jump = positional_encoding(agent_jump, 32)

    agent = tf.concat([agent_plane_x, agent_plane_z, agent_jump], axis=1)

    global_state = agent
    global_state = tf.reshape(global_state, (-1, 3 * 32))


    global_state = linear(global_state, 1024, name='latent_1', activation=tf.nn.leaky_relu,

                         )

    global_state = linear(global_state, 512, name='latent_2', activation=tf.nn.leaky_relu,

                          )

    global_state = linear(global_state, 128, name='latent_3', activation=tf.nn.relu,

                          )

    global_state = linear(global_state, 128, name='latent_4', activation=tf.nn.relu,

                          )

    global_state = linear(global_state, 64, name='out',
                          )


    return global_state

def network_spec_rnd_target(states):

    global_state = states[0]

    agent_plane_x, agent_plane_z, agent_jump, is_grounded, depth_buf, goal_weight = \
        tf.split(global_state, [1, 1, 1, 1, 1024, 3], axis=1)

    agent_plane_x = agent_plane_x * 2816
    agent_plane_x = tf.cast(agent_plane_x, tf.int32)
    agent_plane_x = positional_encoding(agent_plane_x, 32)

    agent_plane_z = agent_plane_z * 2752
    agent_plane_z = tf.cast(agent_plane_z, tf.int32)
    agent_plane_z = positional_encoding(agent_plane_z, 32)

    # agent_plane_x = ((agent_plane_x + 1) / 2) * 1600
    # agent_plane_z = ((agent_plane_z + 1) / 2) * 1150

    agent_jump = agent_jump * 100
    agent_jump = tf.cast(agent_jump, tf.int32)
    agent_jump = positional_encoding(agent_jump, 32)

    agent = tf.concat([agent_plane_x, agent_plane_z, agent_jump], axis=1)
    global_state = agent
    global_state = tf.reshape(global_state, (-1, 3 * 32))


    global_state = linear(global_state, 1024, name='latent_1', activation=tf.nn.leaky_relu,
                         )

    global_state = linear(global_state, 512, name='latent_2', activation=tf.nn.leaky_relu,
                          )

    global_state = linear(global_state, 64, name='out',
                          )

    return global_state

def input_spec_irl():
    input_length = 1028
    global_state = tf.compat.v1.placeholder(tf.float32, [None, input_length], name='state')

    global_state_n = tf.compat.v1.placeholder(tf.float32, [None, input_length], name='state_n')

    act = tf.compat.v1.placeholder(tf.int32, [None, 1], name='act')

    return [[global_state], act, [global_state_n]]

def obs_to_state_irl(obs):
    if len(obs[0]['global_in']) > 1028:
        global_batch = np.stack([state['global_in'][:-3] for state in obs])
    else:
        global_batch = np.stack([state['global_in'] for state in obs])
    return [global_batch]

def network_spec_irl(states, states_n, act, with_action, actions_size):

    global_state = states[0]
    global_state_n = states_n[0]
    action_state = tf.cast(act, tf.int32)

    agent_plane_x, agent_plane_z, agent_jump, is_grounded, depth_buf_state = \
        tf.split(global_state, [1, 1, 1, 1, 1024], axis=1)

    depth_buf_state = tf.reshape(depth_buf_state, [-1, 32, 32, 1])
    depth_buf = conv_layer_2d(depth_buf_state, 32, [3, 3], strides=(2, 2), name='conv1', activation=tf.nn.relu)
    depth_buf = conv_layer_2d(depth_buf, 64, [3, 3], strides=(2, 2), name='conv2', activation=tf.nn.relu)
    depth_buf = tf.reshape(depth_buf, [-1, 8 * 8 * 64])


    action_state = embedding(action_state, indices=6, size=512, name='action_embs')
    action_state = tf.reshape(action_state, [-1, 512])
    action = action_state

    encoded = tf.concat([depth_buf, action], axis=1)

    global_state = linear(encoded, 1024, name='latent_1', activation=tf.nn.relu,

                         )

    global_state = linear(global_state, 512, name='latent_2', activation=tf.nn.relu,

                          )

    global_state = linear(global_state, 128, name='latent_3', activation=tf.nn.relu,

                          )

    global_state = linear(global_state, 1, name='out',
                          init=tf.compat.v1.keras.initializers.Orthogonal(gain=np.sqrt(2), seed=None,
                                                                          dtype=tf.dtypes.float32)
                          )

    return global_state, depth_buf_state, action_state
