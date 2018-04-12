import numpy as np
import tensorflow as tf
import random

from rl_dnc.model.dnc_wrapper import DNCWrapper


sequence_length = 3
# used so that only the last value in the target is of importance
sequence_mask = np.array([[0] for i in range(sequence_length-1)] + [[1]])


# simple count(1s)%2 == ?
def generate_data(count):
    inputs = np.zeros((count, sequence_length, 1, 1))
    outputs = np.zeros_like(inputs)
    for i in range(count):
        sequence = [random.randint(0, 1) for i in range(sequence_length)]
        observation = np.zeros((sequence_length,1,1))
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
    iterations = 5
    report_interval = 100

    inputs, outputs = generate_data(iterations)
    costfun_prediction = tf.placeholder(shape=inputs[0].shape, dtype=tf.float32)
    costfun_target = tf.placeholder(shape=outputs[0].shape, dtype=tf.float32)
    costfun_mask = tf.placeholder(shape=sequence_mask.shape, dtype=tf.float32)

    costfun_node = cost_fun(costfun_prediction, costfun_target, costfun_mask)
    xent_costfun_node = xent_cost_fun(costfun_prediction, costfun_target, costfun_mask)

    with tf.Session() as sess:

        for i in range(iterations):
            obs = inputs[i]
            target = outputs[i]
            out, xent_out = sess.run([costfun_node, xent_costfun_node], feed_dict={
                costfun_prediction: obs,
                costfun_target: target,
                costfun_mask: sequence_mask,
            })
            print('observation: ', obs[:,0,0])
            print('target: ', target[:,0,0])
            print('mask: ', sequence_mask[:,0])
            print('costfun: ', out)
            print('xentcostfun: ', xent_out)
            print('\n')


if __name__ == "__main__":
    tf.app.run()
