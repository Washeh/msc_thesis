# based on https://github.com/keon/deep-q-learning/blob/master/ddqn.py
# results: even after 15000+ iterations, the agent is stuck with <25 average score

from collections import deque
from rl_dnc.model.dnc_wrapper import DNCWrapper
import tensorflow as tf
import random
import numpy as np
import gym
from copy import deepcopy


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = 1
        self.memory = deque(maxlen=5000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.99
        self.learning_rate = 0.001
        self.model = DNCWrapper(self.state_size, self.action_size, self.batch_size, self.cost_fun, scope="dnc_wrapper_model")
        self.target_model = DNCWrapper(self.state_size, self.action_size, self.batch_size, self.cost_fun, scope="dnc_wrapper_target")
        self.state3d = np.zeros((1, 1, state_size))
        self.default_mask = np.ones((1, 1), dtype=np.float32)

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def __state_to_3d(self, state):
        self.state3d[0, :, :] = state
        return self.state3d

    def compile(self, session, save_dir='./../saves/dqn_keon/'):
        self.model.compile(session, save_dir, save_name='model')
        self.target_model.compile(session, save_dir, save_name='target')
        self.update_target_model()

    """
    @staticmethod
    def cost_fun(prediction, target, mask):
        return tf.reduce_mean(tf.square(tf.subtract(prediction, target)))
    """

    @staticmethod
    def cost_fun(prediction, target, mask):
        diff = tf.squared_difference(prediction, target)
        loss_time_batch = tf.reduce_sum(diff, axis=2)
        loss_batch = tf.reduce_sum(loss_time_batch * mask, axis=0)
        batch_size = tf.cast(tf.shape(prediction)[1], dtype=loss_time_batch.dtype)
        loss = tf.reduce_sum(loss_batch) / batch_size
        return loss

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, model_state, action, reward, next_state, done):
        self.memory.append((state, model_state, action, reward, next_state, done))

    def act(self, state, model_state):
        if np.random.rand() <= self.epsilon:
            rand_action = random.randrange(self.action_size)
            return rand_action, model_state, -1
        act_values, model_state = self.model.predict_with_state(model_state, state)
        best_action = np.argmax(act_values[0, 0])
        return best_action, model_state, [int(i) for i in act_values[0, 0, :]]

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, model_state, action, reward, next_state, done in minibatch:
            target, next_model_state = self.model.predict_with_state(model_state, state)
            #print(target)
            if done:
                target[0, 0, action] = reward
            else:
                #a, _ = self.model.predict_with_state(next_model_state, next_state)
                a, _ = self.model.predict_with_state(model_state, next_state)
                #t, _ = self.target_model.predict_get_state(next_model_state, next_state)
                #t, _ = self.model.predict_get_state(next_model_state, next_state)
                t, _ = self.target_model.predict_with_state(model_state, next_state)
                a, t = a[0, 0], t[0, 0]
                target[0, 0, action] = reward + self.gamma * t[np.argmax(a)]
            #print(target, reward, '\n')
            self.model.train_with_state(model_state, state, target, self.default_mask, iterations=1)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self):
        self.model.restore()

    def save(self):
        self.model.save()


EPISODES = 10000  #50000
STEPS_PER_EPISODE = 200  #500
SAVE_INTERVAL = 100
REPLAY_BATCH_SIZE = 32
MIN_REPLAYS_FOR_TRAINING = 1000


def main(unused_args):
    tf.flags.FLAGS.learning_rate = 1e-3
    tf.flags.FLAGS.clip_value = 50000

    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)

    with tf.Session() as sess:
        agent.compile(sess)
        try:
            agent.load()
            agent.set_epsilon(0.2)
            print('Loaded agent weights, reduced exploration rate')
        except:
            pass

        for e in range(EPISODES):
            state = env._reset_dnc_states()
            state = np.reshape(state, [1, 1, state_size])
            model_state = None
            expectations  =[]
            for time in range(STEPS_PER_EPISODE):
                # env.render()
                # state = observations (environment state), model_state = state of the dnc model
                action, model_state_new, q_value = agent.act(state, model_state)
                next_state, reward, done, _ = env.step(action)
                reward = reward if not done else -10
                next_state = np.reshape(next_state, [1, 1, state_size])
                agent.remember(state, deepcopy(model_state), action, reward, next_state, done)
                state = next_state
                model_state = model_state_new
                expectations.append(q_value)
                if done:
                    agent.update_target_model()
                    print("episode: {}/{}, e: {:.2}, score: {}".format(e, EPISODES, agent.epsilon, time), expectations)
                    break
            if len(agent.memory) > MIN_REPLAYS_FOR_TRAINING:
                agent.replay(REPLAY_BATCH_SIZE)
            if (e+1) % SAVE_INTERVAL == 0:
                agent.save()
                print('Saved agent state')

if __name__ == "__main__":
    main(None)
