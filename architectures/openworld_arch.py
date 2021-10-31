import tensorflow as tf
from layers.layers import *

def input_spec():
    input_length = 68
    global_state = tf.compat.v1.placeholder(tf.float32, [None, input_length], name='state')

    return [global_state]

def obs_to_state(obs):
    global_batch = np.stack([np.asarray(state['global_in']) for state in obs])
    return [global_batch]

def network_spec(states):
    input_length = 68
    with_circular = False

    global_state = states[0]
    if input_length > 0:
        global_state, rays, coins, obstacles = tf.split(global_state, [7, 12, 28, 21], axis=1)
        #global_state = linear(global_state, 1024, name='embs', activation=tf.nn.relu)

        agent = global_state

        rays = tf.cast(tf.reshape(rays, [-1, 5, 5]), tf.int32)
        rays = embedding(rays, indices=4, size=32, name='rays_embs')
        rays = conv_layer_2d(rays, 32, [3, 3], name='conv_31', activation=tf.nn.relu)
        rays = conv_layer_2d(rays, 64, [3, 3], name='conv_32', activation=tf.nn.relu)
        rays = tf.reshape(rays, [-1, 5 * 5 * 64])

        # There are other things before this point...

        # First list of entities
        # The list of entity must have shape [batch_size, number_of_entities, features]
        coins = tf.reshape(coins, [-1, 14, 2])
        # Create mask manually (not inside the transformer)
        # TODO: The mask will have shape [batch_size, 1, number of entities]
        # TODO: The mask must be done BEFORE the embeddings
        coins_mask = create_mask(coins, 99)
        # Create an embedding of the first list of entities
        coins = linear(coins, 1024, name='entities_obs_1', activation=tf.nn.tanh)

        # Second list of entities
        agent = tf.reshape(agent, [-1, 1, 7])
        # Create mask manually (not inside the transformer)
        # TODO: The mask will have shape [batch_size, 1, number of entities]
        # TODO: The mask must be done BEFORE the embeddings
        agent_mask = create_mask(agent, 99)
        # Create an embedding of the second list of entities
        # Same size of the previous entity embedding
        agent = linear(agent, 1024, name='entities_obs_2', activation=tf.nn.tanh)

        # Concatenate the embeddings
        entity_embeddings = tf.concat([agent, coins], axis=1)
        # Concatenate the mask
        # TODO: the concatenations must be done in the same order of the entity concatenation
        my_mask = tf.concat([agent_mask, coins_mask], axis=2)

        # Apply the transformer. I will pass the concatenated entities and the mask to the layer, without internal
        # embeddings (I already done the entity embeddings) and with max_pool at the end
        entity_embeddings, _ = transformer(entity_embeddings, n_head=4, hidden_size=1024, mask_value=99,
                                                 with_embeddings=False, name='transformer', mask=my_mask, pooling='max')
        # The transformer (with max pooling) will output a tensor with shape [batch_size, 1, hidden_size]
        # I reshape the tensor to be [batch_size, hidden_size]
        entity_embeddings = tf.reshape(entity_embeddings, [-1, 1024])

        # After this, I use the entity_embeddings in some ways...


        obstacles = tf.reshape(obstacles, [-1, 7, 3])
        obstacles = linear(obstacles, 1024, name='embs_obs', activation=tf.nn.relu)
        obstacles, _, mask = transformer(obstacles, n_head=4, hidden_size=1024, mask_value=99, with_embeddings=False,
                                   name='transformer_global')
        obstacles = tf.math.reduce_max(obstacles, axis=2)
        obstacles = tf.reshape(obstacles, [-1, 1024])



        global_state = tf.concat([entity_embeddings], axis=1)

    else:
        # agent, goal, rays, obs = tf.split(global_state, [4, 3, 12, 21], axis=1)
        # Jump
        agent, goal, rays, obs, points = tf.split(global_state, [4, 3, 12, 21, 12], axis=1)

        rays = tf.reshape(rays, [-1, 12, 1])
        rays = circ_conv1d(rays, activation='relu', kernel_size=3, filters=32)
        rays = tf.reshape(rays, [-1, 12 * 32])

        points = tf.reshape(points, [-1, 4, 3])
        points, _ = transformer(points, n_head=4, hidden_size=1024, mask_value=99, with_embeddings=True,
                                   name='transformer_global', pooling='max')
        points = tf.reshape(points, [-1, 1024])

        global_state = tf.concat([goal, rays], axis=1)

        global_state = linear(global_state, 1024, name='embs', activation=tf.nn.relu)

    return global_state
