import numpy as np
import gym


class MovingBlocksEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, shape, num_seq_targets, num_objects, total_numbers=5, add_observation_bits=False,
                 rmoves=4, target_reward=10, step_reward=-0.1, max_steps=1000):
        """
        :param shape: shape (y,x) of the board
        :param num_seq_targets: number of sequential target states, uniformly drawn from [min, max]
        :param num_objects: number of blocks per target state that have to be positioned correclty,
                            uniformly drawn from [min, max]
        :param rmoves: random moves between target positions
        :param total_numbers: amount of different numbers, {0, 1, 2, 3, 4} by default
        :param add_observation_bits: adds bits to signal when targets are reached or output matters
        """
        self._num_seq_targets = num_seq_targets
        self._num_objects = num_objects
        self._total_numbers = total_numbers
        self._shape = shape
        self._rmoves = rmoves
        self._target_reward = target_reward
        self._step_reward = step_reward
        self._max_steps = max_steps
        self._max_action = 2*self._shape[0] + 2*self._shape[1]
        self._observation_bits = 2*self._shape[0]*self._shape[1]
        if add_observation_bits:
            self._get_obs = self._get_obs_with_bits
            self._observation_bits += 2
        else:
            self._get_obs = self._get_obs_no_bits

        self.action_space = gym.spaces.Discrete(self._max_action)
        self.observation_space = gym.spaces.MultiDiscrete([self._total_numbers for _ in range(self._observation_bits)])

        self._board = None
        self._empty_board = None
        self._targets = None
        self._current_target = None
        self._current_steps = None
        self._num_seq_targets_drawn = None
        self._num_objects_drawn = None
        self.reset()

    def _reached_target(self):
        return np.array_equal(self._board, self._targets[self._current_target])

    def _shift(self, a):
        if a < self._shape[0]:
            self._board[a, :] = np.roll(self._board[a, :], shift=1)
            return
        a -= self._shape[0]
        if a < self._shape[0]:
            self._board[a, :] = np.roll(self._board[a, :], shift=-1)
            return
        a -= self._shape[0]
        if a < self._shape[1]:
            self._board[:, a] = np.roll(self._board[:, a], shift=1)
            return
        a -= self._shape[1]
        self._board[:, a] = np.roll(self._board[:, a], shift=-1)

    def _get_obs_no_bits(self, targets, target_reached=0, output_matters=0):
        return np.concatenate((self._board, targets), axis=0).flatten()

    def _get_obs_with_bits(self, targets, target_reached=0):
        output_matters = 0 if self._current_steps < (self._num_objects_drawn - 1) else 1
        return np.concatenate((self._board.flatten(), targets.flatten(), [target_reached, output_matters]), axis=0)

    def step(self, action):
        """
        returns observation, reward, done, info
        where observation is a tuple of (board state, current target),
        and targets will only be shown sequentially, a single time each, in the beginning of the episode
        """
        assert 0 <= action < self._max_action
        info = {}
        done = False
        reward = self._step_reward
        bit_target = 1
        self._current_steps += 1
        if self._current_steps < self._num_seq_targets_drawn:
            return self._get_obs(self._targets[self._current_steps]), reward, done, info
        self._shift(action)
        if self._reached_target():
            bit_target = 1
            self._current_target += 1
            if self._current_target >= self._num_seq_targets_drawn:
                done = True
                reward = self._target_reward
        if self._current_steps >= self._max_steps:
            done = True
        return self._get_obs(self._empty_board, target_reached=bit_target), reward, done, info

    def reset(self):
        self._num_seq_targets_drawn = np.random.randint(self._num_seq_targets[0], self._num_seq_targets[1] + 1)
        self._num_objects_drawn = np.random.randint(self._num_objects[0], self._num_objects[1] + 1)
        self._board = np.zeros(dtype=np.int8, shape=self._shape)

        for i in range(self._num_objects_drawn):
            letter = np.random.randint(1, self._total_numbers)
            d0 = np.random.randint(0, self._shape[0])
            d1 = np.random.randint(0, self._shape[1])
            self._board[d0, d1] = letter

        self._targets = []
        for i in range(self._num_seq_targets_drawn):
            self._targets.append(np.copy(self._board))
            self._current_target = i
            m = 0
            while True:
                self._shift(np.random.randint(0, self._max_action))
                m += 1
                if m >= self._rmoves and not self._reached_target():
                    break

        self._targets = self._targets[::-1]
        self._current_target = 0
        self._current_steps = 0
        self._empty_board = np.zeros_like(self._board)
        return self._get_obs(self._targets[0])

    def render(self, mode='human', close=False):
        print('\n', '-'*15)
        print('board:')
        print(self._board)
        for i in range(len(self._targets)):
            s = ''
            if i == self._current_target:
                s = '(current)'
            print('\ntarget', i+1, s)
            print(self._targets[i])

    def seed(self, seed=None):
        np.random.seed(seed)


class MovingBlocksEnv33Seq12Obj12(MovingBlocksEnv):
    def __init__(self):
        super(MovingBlocksEnv33Seq12Obj12, self).__init__(shape=[3, 3],
                                                          num_seq_targets=[1, 2],
                                                          num_objects=[1, 2])


class MovingBlocksEnv33Seq13Obj13(MovingBlocksEnv):
    def __init__(self):
        super(MovingBlocksEnv33Seq13Obj13, self).__init__(shape=[3, 3],
                                                          num_seq_targets=[1, 3],
                                                          num_objects=[1, 3])


class MovingBitsEnv44Seq12Obj13(MovingBlocksEnv):
    def __init__(self):
        super(MovingBitsEnv44Seq12Obj13, self).__init__(shape=[4, 4],
                                                        num_seq_targets=[1, 2],
                                                        num_objects=[1, 3],
                                                        total_numbers=2)


class MovingBitsEnv44Seq13Obj13(MovingBlocksEnv):
    def __init__(self):
        super(MovingBitsEnv44Seq13Obj13, self).__init__(shape=[4, 4],
                                                        num_seq_targets=[1, 3],
                                                        num_objects=[1, 3],
                                                        total_numbers=2)


class MovingBitsEnv44Seq14Obj13(MovingBlocksEnv):
    def __init__(self):
        super(MovingBitsEnv44Seq14Obj13, self).__init__(shape=[4, 4],
                                                        num_seq_targets=[1, 4],
                                                        num_objects=[1, 3],
                                                        total_numbers=2)


class MovingBitsEnv33Seq11Obj13(MovingBlocksEnv):
    def __init__(self):
        super(MovingBitsEnv33Seq11Obj13, self).__init__(shape=[3, 3],
                                                        num_seq_targets=[1, 1],
                                                        num_objects=[1, 3],
                                                        total_numbers=2)


class MovingBitsEnv33Seq12Obj13(MovingBlocksEnv):
    def __init__(self):
        super(MovingBitsEnv33Seq12Obj13, self).__init__(shape=[3, 3],
                                                        num_seq_targets=[1, 2],
                                                        num_objects=[1, 3],
                                                        total_numbers=2)


class MovingBitsEnv33Seq11Obj13v1(MovingBlocksEnv):
    def __init__(self):
        super(MovingBitsEnv33Seq11Obj13v1, self).__init__(shape=[3, 3],
                                                          num_seq_targets=[1, 1],
                                                          num_objects=[1, 3],
                                                          total_numbers=2,
                                                          add_observation_bits=True)


class MovingBitsEnv33Seq12Obj13v1(MovingBlocksEnv):
    def __init__(self):
        super(MovingBitsEnv33Seq12Obj13v1, self).__init__(shape=[3, 3],
                                                          num_seq_targets=[1, 2],
                                                          num_objects=[1, 3],
                                                          total_numbers=2,
                                                          add_observation_bits=True)


class MovingBitsEnv33Seq13Obj13v1(MovingBlocksEnv):
    def __init__(self):
        super(MovingBitsEnv33Seq13Obj13v1, self).__init__(shape=[3, 3],
                                                          num_seq_targets=[1, 3],
                                                          num_objects=[1, 3],
                                                          total_numbers=2,
                                                          add_observation_bits=True)


class MovingBitsEnv33Seq11Obj13NTRv1(MovingBlocksEnv):
    def __init__(self):
        super(MovingBitsEnv33Seq11Obj13NTRv1, self).__init__(shape=[3, 3],
                                                             num_seq_targets=[1, 1],
                                                             num_objects=[1, 3],
                                                             total_numbers=2,
                                                             add_observation_bits=True,
                                                             target_reward=0)


class MovingBitsEnv33Seq12Obj13NTRv1(MovingBlocksEnv):
    def __init__(self):
        super(MovingBitsEnv33Seq12Obj13NTRv1, self).__init__(shape=[3, 3],
                                                             num_seq_targets=[1, 2],
                                                             num_objects=[1, 3],
                                                             total_numbers=2,
                                                             add_observation_bits=True,
                                                             target_reward=0)
