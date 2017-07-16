import numpy as np

class ReplayMemory:
    """
    Class to represent the replay memory logic.

    Stores (state, future, action, reward, is_terminal) via add_example.
    Sample mini batches via the sample command.
    Uses a fixed mini batch size to preserve memory.
    """
    def __init__(self, size, image_size, phi_length, minibatch_size):
        # Pre allocate
        self._memory_state = np.zeros(shape=(size, phi_length, image_size[0], image_size[1]), dtype=np.int8)
        self._memory_future_state = np.zeros(shape=(size, phi_length, image_size[0], image_size[1]), dtype=np.int8)
        self._rewards = np.zeros(shape=(size, 1), dtype=np.float32)
        self._is_terminal = np.zeros(shape=(size, 1), dtype=np.bool)
        self._actions = np.zeros(shape=(size, 1), dtype=np.int8)

        self._mini_batch_state = np.zeros(shape=(minibatch_size, phi_length, image_size[0], image_size[1]), dtype=np.float32)
        self._mini_batch_future_state = np.zeros(shape=(minibatch_size, phi_length, image_size[0], image_size[1]), dtype=np.float32)

        self._mini_batch_size = minibatch_size
        self._counter = 0
        self._size = size
        self._counter = 0

    def add_example(self, state, future_state, action, reward, is_terminal):
        position = self._counter % self._size
        self._memory_state[position,:,:,:] = state # Copies values to replay memory
        self._memory_future_state[position,:,:,:] = future_state # Copies values to replay memory
        self._rewards[position] = reward
        self._is_terminal[position] = is_terminal
        self._actions[position] = action
        self._counter += 1

    def replay_full(self):
        return self._counter >= self._size

    def sample(self):
        ind = np.random.choice(self._size, size=self._mini_batch_size)

        # Avoiding a copy action as much as possible
        self._mini_batch_state[:] = self._memory_state[ind,:,:,:]
        self._mini_batch_future_state[:] = self._memory_future_state[ind,:,:,:]

        rewards = self._rewards[ind]
        is_terminal = self._is_terminal[ind]
        actions = self._actions[ind]

        return self._mini_batch_state, self._mini_batch_future_state, actions, rewards, is_terminal
