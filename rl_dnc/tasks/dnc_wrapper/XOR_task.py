import numpy as np
import tensorflow as tf

from rl_dnc.model.dnc_wrapper import DNCWrapper

# TODO remove
a = tf.Variable(4, trainable=True, name="a")
b = tf.Variable(3, trainable=True, name="b")


# simple XOR problem
inputs = [np.array([[0, 0]], dtype=np.float32),
          np.array([[0, 1]], dtype=np.float32),
          np.array([[1, 0]], dtype=np.float32),
          np.array([[1, 1]], dtype=np.float32)]
outputs = [np.array([[1]], dtype=np.float32),
           np.array([[0]], dtype=np.float32),
           np.array([[0]], dtype=np.float32),
           np.array([[1]], dtype=np.float32)]
# mask will not be used here
mask = np.ones((1,1), dtype=np.float32)


def cost_fun(prediction, target, mask):
    return tf.reduce_mean(tf.square(tf.subtract(prediction, target)))


def main(unused_argv):
    input_size = 2
    output_size = 1
    batch_size = 1
    iterations = 2000
    report_interval = 100
    index_seq = np.random.randint(0, len(inputs), iterations)

    # changing some dnc settings
    # tf.flags.FLAGS.hidden_size = 3
    tf.flags.FLAGS.learning_rate = 1e-3

    wrapper = DNCWrapper(input_size, output_size, batch_size, cost_fun, scope="dnc_wrapper")
    tf.logging.set_verbosity(3)  # Print INFO log messages.

    global_init = tf.global_variables_initializer()

    # Train.
    # with tf.train.SingularMonitoredSession() as sess:  # does not work with custom saver? only hooks?
    with tf.Session() as sess:
        def results_for(w):
            results = np.zeros((len(inputs), 2))
            results[:, 0] = [o[0, 0] for o in outputs]
            for i in range(len(inputs)):
                observation = inputs[i]
                predicted = w.predict([observation])
                results[i, 1] = predicted
            return results

        def train(w, num_iterations):
            total_loss = 0
            for i in range(0, num_iterations):
                index = index_seq[i]
                observation = inputs[index]
                target = outputs[index]
                loss, prediction, output = w.train([observation], [target], mask)

                total_loss += loss
                if (i+1) % report_interval == 0:
                    tf.logging.info("%d: Avg training loss %f.", i, total_loss / report_interval)
                    total_loss = 0

        wrapper.compile(sess, save_dir='./../../saves/simple/xor/', save_name="xor")
        try:
            wrapper.restore()
        except:
            pass

        train(wrapper, iterations)

        print('results:')
        print(results_for(wrapper))

        wrapper.save()


if __name__ == "__main__":
    tf.app.run()
