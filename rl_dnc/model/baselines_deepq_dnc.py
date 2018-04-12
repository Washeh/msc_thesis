# originated from DeepMind's train.py
# does not seem to work since the agent's internal state is omitted

import tensorflow as tf
import dnc

import gym
from baselines import deepq

FLAGS = tf.flags.FLAGS

# Model parameters
tf.flags.DEFINE_integer("hidden_size", 8, "Size of LSTM hidden layer.")
tf.flags.DEFINE_integer("memory_size", 8, "The number of memory slots.")
tf.flags.DEFINE_integer("word_size", 8, "The width of each memory slot.")
tf.flags.DEFINE_integer("num_write_heads", 1, "Number of memory write heads.")
tf.flags.DEFINE_integer("num_read_heads", 1, "Number of memory read heads.")
tf.flags.DEFINE_integer("clip_value", 20, "Maximum absolute value of controller and dnc outputs.")
tf.flags.DEFINE_integer("batch_size", 1, "Batch size.")


class DNCDeepqWrapper:
    def __init__(self, batch_size=1):
        self.batch_size = batch_size

    def __call__(self, input_var, output_size, scope, reuse=False):
        print('>'*5, "scope", scope)
        #self.scopes.append(scope)
        with tf.variable_scope(scope, reuse=reuse):
            access_config = {
                "memory_size": FLAGS.memory_size,
                "word_size": FLAGS.word_size,
                "num_reads": FLAGS.num_read_heads,
                "num_writes": FLAGS.num_write_heads,
            }
            controller_config = {
                "hidden_size": FLAGS.hidden_size,
            }
            clip_value = FLAGS.clip_value
            self.dnc_model = dnc.DNC(access_config, controller_config, output_size, clip_value)
            self.dnc_model_state = self.dnc_model.initial_state(self.batch_size, dtype=tf.float32)

            reshaped_input_var = tf.reshape(input_var, [1, -1, int(input_var.shape[1])], name="reshape_in")

            prediction, self.dnc_model_state = tf.nn.dynamic_rnn(
                cell=self.dnc_model,
                inputs=reshaped_input_var,
                time_major=True,
                initial_state=self.dnc_model_state)

            #trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
            #self.saver = tf.train.Saver(trainable_variables)

            # should be [self.batch_size, output_size] ?
            reshaped_output = tf.reshape(prediction, [-1, output_size], name="reshape_out")
            return reshaped_output


def callback(lcl, glb):
    # stop training if reward exceeds 199
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
    return is_solved


def main():
    env = gym.make("CartPole-v0")
    batch_size = FLAGS.batch_size
    model = DNCDeepqWrapper(batch_size=batch_size)
    act = deepq.learn(
        env,
        q_func=model,
        lr=1e-3,
        max_timesteps=100000,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        print_freq=10,
        callback=callback,
        batch_size=batch_size
    )
    #print("Saving model to cartpole_model.pkl")
    #act.save("cartpole_model.pkl")


if __name__ == '__main__':
    main()

# TODO when saving
# "Sonnet AbstractModule instances cannot be serialized. You should "
# sonnet.python.modules.base.NotSupportedError: Sonnet AbstractModule instances cannot be serialized. You should instead serialize all necessary configuration which will allow modules to be rebuilt.
