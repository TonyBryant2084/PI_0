import numpy as np

observation_state = np.array([
    [
        [1,  2,  3,  4],
        [5,  6,  7,  8],
        [9, 10, 11, 12],
    ],
    [
        [13, 14, 15, 16],
        [17, 18, 19, 20],
        [21, 22, 23, 24],
    ],
])

print(observation_state.shape)  # (2, 3, 4)

batch_shape = observation_state.shape[:-1]

print(batch_shape)  # (2, 3)