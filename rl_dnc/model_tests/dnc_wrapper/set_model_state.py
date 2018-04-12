from collections import Iterable

import numpy as np
import tensorflow as tf

from rl_dnc.model.dnc_wrapper import DNCWrapper
from rl_dnc.tasks.dnc_wrapper.mod2_full_seq import cost_fun, generate_data


def print_state(state, name):
    if type(state) is Iterable:
        print(name)
        for v in state.keys():
            print(v, '\n', state[v], '\n')
        print('\n', '-'*150, '\n')


def main(unused_argv):
    input_size = 1
    output_size = 1
    batch_size = 1
    iterations = 3
    sequence_length = 20

    inputs, outputs, mask = generate_data(iterations, sequence_length)

    wrapper = DNCWrapper(input_size, output_size, batch_size, cost_fun, scope="dnc_wrapper")
    tf.logging.set_verbosity(3)  # Print INFO log messages.

    # Train.
    # with tf.train.SingularMonitoredSession() as sess:  # does not work with custom saver? only hooks?
    with tf.Session() as sess:

        wrapper.compile(sess, save_dir='./../../saves/simple/mod2_full_seq/', save_name="mod2_seq")
        try:
            wrapper.restore()
        except:
            pass

        #state = None  #wrapper.get_state_init()
        #print_state(state, 'init state:')

        #print_results_for(wrapper, inputs, outputs, count=iterations)
        for i in range(iterations):
            state = None  # resetting initial state
            observation = inputs[i]
            lower, upper = 0, 5
            for j in range(4):
                obs_part = observation[lower:upper]
                print(obs_part[:,0,0])
                prediction, state = wrapper.predict_with_state(state, obs_part)
                # state = deepcopy(state)
                # print(type(state))
                print(np.round(prediction[:,0,0]))
                print('\n')
                lower = upper
                upper = upper + 5
            print('-'*100, '\n')


if __name__ == "__main__":
    tf.app.run()
