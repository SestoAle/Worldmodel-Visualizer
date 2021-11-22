import pickle
import numpy as np


folder = 'arrays'
model_name = 'vizdoom_big_2'

with open('../{}/{}/{}.pkl'.format(folder, model_name, model_name), 'rb') as f:
    raw_demonstrations = pickle.load(f)

ep_length = []
i = 0
for j, dem in enumerate(raw_demonstrations['obs']):
    # if dem['global_in'][0] < 0:
    #     ep_length.append(i)
    #     i = 0
    if dem['global_in'][0] > 1:
        dem['global_in'][0] = raw_demonstrations['obs'][j+1]['global_in'][0]
        dem['global_in'][1] = raw_demonstrations['obs'][j+1]['global_in'][1]
        dem['global_in'][2] = raw_demonstrations['obs'][j+1]['global_in'][2]

        ep_length.append(i)
        i = 0
    i += 1
ep_length.append(i)
ep_length = ep_length[1:]


print(raw_demonstrations['obs'][0]['global_in'][0])
raw_demonstrations['episode_len'] = np.asarray(ep_length)
print(len(raw_demonstrations['obs']))
with open('../{}/{}/{}.pkl'.format(folder, model_name, model_name), 'wb') as f:
    pickle.dump(raw_demonstrations, f, pickle.HIGHEST_PROTOCOL)

print(raw_demonstrations['episode_len'])
print('DONE')