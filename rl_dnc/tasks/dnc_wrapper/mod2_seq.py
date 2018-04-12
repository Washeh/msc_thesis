import numpy as np
import tensorflow as tf
import random

from rl_dnc.model.dnc_wrapper import DNCWrapper


sequence_length = 13
# used so that only the last value in the target is of importance
sequence_mask = np.array([[0] for i in range(sequence_length-1)] + [[1]])


# simple count(1s)%2 == ?
# only the last value of the outputs matters, rest is 0
def generate_data(count):
    inputs = np.zeros((count, sequence_length, 1, 1))
    outputs = np.zeros_like(inputs)
    for i in range(count):
        sequence = [random.randint(0, 1) for i in range(sequence_length)]
        observation = np.zeros((sequence_length, 1, 1))
        observation[:, 0, 0] = sequence
        expected = np.zeros_like(observation)
        expected[-1, 0, 0] = sum(sequence) % 2
        inputs[i] = observation
        outputs[i] = expected
    return inputs, outputs


def cost_fun(prediction, target, mask):
    diff = tf.squared_difference(prediction, target)
    loss_time_batch = tf.reduce_sum(diff, axis=2)
    loss_batch = tf.reduce_sum(loss_time_batch * mask, axis=0)
    batch_size = tf.cast(tf.shape(prediction)[1], dtype=loss_time_batch.dtype)
    loss = tf.reduce_sum(loss_batch) / batch_size
    return loss


"""
def cost_fun(prediction, target, mask):
    diff = tf.squared_difference(prediction * mask, target * mask)
    return tf.reduce_sum(diff)
"""

def xent_cost_fun(prediction, target, mask):
    xent = tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=prediction)
    loss_time_batch = tf.reduce_sum(xent, axis=2)
    loss_batch = tf.reduce_sum(loss_time_batch * mask, axis=0)
    batch_size = tf.cast(tf.shape(prediction)[1], dtype=loss_time_batch.dtype)
    loss = tf.reduce_sum(loss_batch) / batch_size
    return loss


def main(unused_argv):
    input_size = 1
    output_size = 1
    batch_size = 1
    iterations = 1500
    report_interval = 100

    inputs, outputs = generate_data(iterations)

    # changing some dnc settings
    # tf.flags.FLAGS.hidden_size = 3
    tf.flags.FLAGS.learning_rate = 1e-4

    wrapper = DNCWrapper(input_size, output_size, batch_size, cost_fun, scope="dnc_wrapper")
    tf.logging.set_verbosity(3)  # Print INFO log messages.

    # Train.
    # with tf.train.SingularMonitoredSession() as sess:  # does not work with custom saver? only hooks?
    with tf.Session() as sess:
        def results_for(w, count=5):
            random_slice = np.random.randint(0, len(inputs), count)
            sliced_outputs = outputs[random_slice]
            sliced_inputs = inputs[random_slice]
            results = np.zeros((count, 2))
            results[:, 0] = [o[-1, 0, 0] for o in sliced_outputs]
            for i in range(count):
                observation = sliced_inputs[i]
                predicted = w.predict(observation)
                #print('observation: ', observation)
                #print('predicted: ', predicted)
                #print('\n')
                results[i, 1] = predicted[-1, 0, 0]
            return results

        def train(w, num_iterations):
            total_loss = 0
            for i in range(0, num_iterations):
                observation = inputs[i]
                target = outputs[i]
                loss, prediction, output = w.train(observation, target, sequence_mask)

                total_loss += loss
                if (i+1) % report_interval == 0:
                    tf.logging.info("%d: Avg training loss %f.", i, total_loss / report_interval)
                    total_loss = 0

        wrapper.compile(sess, save_dir='./../../saves/simple/mod2_seq/', save_name="mod2_seq")
        try:
            wrapper.restore()
        except:
            pass

        train(wrapper, iterations)
        wrapper.save()

        print('results:')
        print(results_for(wrapper))


if __name__ == "__main__":
    tf.app.run()
