# based on DeepMind's train.py

import tensorflow as tf

import repeat_copy
from rl_dnc.model.dnc_wrapper import DNCWrapper

FLAGS = tf.flags.FLAGS

# Task parameters
tf.flags.DEFINE_integer("batch_size", 16, "Batch size for training.")
tf.flags.DEFINE_integer("num_bits", 4, "Dimensionality of each vector to copy")
tf.flags.DEFINE_integer("min_length", 1,
    "Lower limit on number of vectors in the observation pattern to copy")
tf.flags.DEFINE_integer("max_length", 2,
    "Upper limit on number of vectors in the observation pattern to copy")
tf.flags.DEFINE_integer("min_repeats", 1,
                        "Lower limit on number of copy repeats.")
tf.flags.DEFINE_integer("max_repeats", 2,
                        "Upper limit on number of copy repeats.")

# Training options.
tf.flags.DEFINE_integer("num_training_iterations", 10000,
                        "Number of iterations to train for.")
tf.flags.DEFINE_integer("report_interval", 100,
                        "Iterations between reports (samples, valid loss).")
tf.flags.DEFINE_string("checkpoint_dir", "/tmp/tf/dnc",
                       "Checkpointing directory.")
tf.flags.DEFINE_integer("checkpoint_interval", -1,
                        "Checkpointing step interval.")


def main(unused_argv):
    dataset = repeat_copy.RepeatCopy(FLAGS.num_bits, FLAGS.batch_size,
                                   FLAGS.min_length, FLAGS.max_length,
                                   FLAGS.min_repeats, FLAGS.max_repeats)

    input_size = FLAGS.num_bits + 2
    output_size = FLAGS.num_bits + 1
    batch_size = FLAGS.batch_size
    wrapper = DNCWrapper(input_size, output_size, batch_size, dataset.cost)
    tf.logging.set_verbosity(3)  # Print INFO log messages.
    # sending one batch of observations
    # -> TODO single observation
    dataset_tensors = dataset()

    # Train.
    with tf.Session() as sess:
        wrapper.compile(sess, save_dir="./../../saves/simple/repeat_copy/", save_name = "repeat_copy")
        try:
            wrapper.restore()
        except:
            pass

        total_loss = 0
        for i in range(1, FLAGS.num_training_iterations+1):
            dt = sess.run(dataset_tensors)
            loss, prediction, output = wrapper.train(dt.observations, dt.target, dt.mask)

            total_loss += loss
            if i % FLAGS.report_interval == 0:
                dataset_string = "" #dataset.to_human_readable(dataset_tensors_np, output_np)
                tf.logging.info("%d: Avg training loss %f.\n%s",
                                i, total_loss / FLAGS.report_interval,
                                dataset_string)
                total_loss = 0

        wrapper.save()


if __name__ == "__main__":
    tf.app.run()