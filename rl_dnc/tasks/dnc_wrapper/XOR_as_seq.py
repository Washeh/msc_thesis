import numpy as np
import tensorflow as tf

from rl_dnc.model.dnc_wrapper import DNCWrapper


# simple XOR problem
inputs = [np.array([[[0]], [[0]]], dtype=np.float32),
          np.array([[[0]], [[1]]], dtype=np.float32),
          np.array([[[1]], [[0]]], dtype=np.float32),
          np.array([[[1]], [[1]]], dtype=np.float32)]
outputs = [np.array([[[0]], [[1]]], dtype=np.float32),
           np.array([[[0]], [[0]]], dtype=np.float32),
           np.array([[[0]], [[0]]], dtype=np.float32),
           np.array([[[0]], [[1]]], dtype=np.float32)]
# mask is used, only last value is important
mask = np.array([[0], [1]])


def cost_fun(prediction, target, mask):
    return tf.reduce_mean(tf.square(tf.subtract(prediction, target)))


def cost_fun2(prediction, target, mask):
    diff = tf.squared_difference(prediction, target)
    loss_time_batch = tf.reduce_sum(diff, axis=2)
    loss_batch = tf.reduce_sum(loss_time_batch * mask, axis=0)
    batch_size = tf.cast(tf.shape(prediction)[1], dtype=loss_time_batch.dtype)
    loss = tf.reduce_sum(loss_batch) / batch_size
    return loss


def main(unused_argv):
    input_size = 1
    output_size = 1
    batch_size = 1
    iterations = 2500
    report_interval = 100
    index_seq = np.random.randint(0, len(inputs), iterations)

    # changing some dnc settings
    # tf.flags.FLAGS.hidden_size = 3
    tf.flags.FLAGS.learning_rate = 1e-3

    wrapper = DNCWrapper(input_size, output_size, batch_size, cost_fun2, scope="dnc_wrapper")
    tf.logging.set_verbosity(3)  # Print INFO log messages.

    global_init = tf.global_variables_initializer()

    # Train.
    # with tf.train.SingularMonitoredSession() as sess:  # does not work with custom saver? only hooks?
    with tf.Session() as sess:
        def results_for(w):
            results = np.zeros((len(inputs), 2))
            results[:, 0] = [o[1, 0, 0] for o in outputs]
            for i in range(len(inputs)):
                observation = inputs[i]
                predicted = w.predict(observation)
                results[i, 1] = predicted[1, 0, 0]
            return results

        def train(w, num_iterations):
            total_loss = 0
            for i in range(0, num_iterations):
                index = index_seq[i]
                observation = inputs[index]
                target = outputs[index]
                loss, prediction, output = w.train(observation, target, mask)

                total_loss += loss
                if (i+1) % report_interval == 0:
                    tf.logging.info("%d: Avg training loss %f.", i, total_loss / report_interval)
                    total_loss = 0

        wrapper.compile(sess, save_dir='./../../saves/simple/xor_as_seq/', save_name="xor_seq")
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
