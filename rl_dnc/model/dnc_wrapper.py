# originated from DeepMind's train.py

import tensorflow as tf
import dnc
import numpy as np
import os

FLAGS = tf.flags.FLAGS

# Model parameters
tf.flags.DEFINE_integer("hidden_size", 64, "Size of LSTM hidden layer.")
tf.flags.DEFINE_integer("memory_size", 16, "The number of memory slots.")
tf.flags.DEFINE_integer("word_size", 16, "The width of each memory slot.")
tf.flags.DEFINE_integer("num_write_heads", 1, "Number of memory write heads.")
tf.flags.DEFINE_integer("num_read_heads", 4, "Number of memory read heads.")
tf.flags.DEFINE_integer("clip_value", 20, "Maximum absolute value of controller and dnc outputs.")

# Optimizer parameters.
tf.flags.DEFINE_float("max_grad_norm", 50, "Gradient clipping norm limit.")
tf.flags.DEFINE_float("learning_rate", 1e-4, "Optimizer learning rate.")
tf.flags.DEFINE_float("optimizer_epsilon", 1e-10,
                      "Epsilon used for RMSProp optimizer.")


class DNCWrapper:

    def __init__(self, input_size, output_size, batch_size, cost_fun, scope="dnc_wrapper"):
        self.batch_size = batch_size
        self.scope = scope

        # to be set in the compile method
        self.save_dir = None
        self.save_name = None
        self.session = None
        self.dtype = tf.float32

        with tf.variable_scope(scope):
            # dnc model
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
            self.dnc_model_state_init = self.dnc_model.initial_state(batch_size, dtype=tf.float32)

            # further tf objects
            # learning rate can be a tensor object, fed with feed_dict, if wanted
            optimizer = tf.train.RMSPropOptimizer(
                FLAGS.learning_rate,
                epsilon=FLAGS.optimizer_epsilon)
            global_step = tf.get_variable(
                name="global_step",
                shape=[],
                dtype=tf.int64,
                initializer=tf.zeros_initializer(),
                trainable=False,
                collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.GLOBAL_STEP])

            # prediction and training
            # dynamic_rnn input: [max_time, batch_size, depth]
            # print(input)   #Tensor("repeat_copy/Reshape_32:0", shape=(?, 16, 6), dtype=float32)
            # print(target)  #Tensor("repeat_copy/Reshape_33:0", shape=(?, 16, 5), dtype=float32)
            # print(mask)    #Tensor("repeat_copy/transpose:0", shape=(?, 16), dtype=float32)
            self.input_placeholder = tf.placeholder(shape=(None, batch_size, input_size), dtype=tf.float32, name='input_placeholder')
            self.target_placeholder = tf.placeholder(shape=(None, batch_size, output_size), dtype=tf.float32, name='target_placeholder')
            self.mask_placeholder = tf.placeholder(shape=(None, batch_size), dtype=tf.float32, name='mask_placeholder')

            self.prediction, self.dnc_model_state = tf.nn.dynamic_rnn(
                cell=self.dnc_model,
                inputs=self.input_placeholder,
                time_major=True,
                initial_state=self.dnc_model_state_init)

            # Used for visualization.
            self.output_node = tf.round(tf.expand_dims(self.mask_placeholder, -1) * tf.sigmoid(self.prediction))

            self.train_loss_node = cost_fun(self.prediction,
                                            self.target_placeholder,
                                            self.mask_placeholder)

            # used for weight setting
            weight_setter_nodes = []
            self.weight_setter_feeds = []
            self.trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)
            for i in range(len(self.trainable_variables)):
                v = self.trainable_variables[i]
                var = tf.Variable(np.zeros(v.shape), dtype=tf.float32)
                weight_setter_nodes.append(v.assign(var))
                self.weight_setter_feeds.append(var)
            self.weight_setter_node = tf.group(*weight_setter_nodes)
            print(self.scope, 'has a total of', len(self.trainable_variables), 'trainable vars')

            # Set up optimizer with global norm clipping.
            grads, _ = tf.clip_by_global_norm(
                tf.gradients(self.train_loss_node, self.trainable_variables),
                FLAGS.max_grad_norm)

            self.train_step_node = optimizer.apply_gradients(
                zip(grads, self.trainable_variables),
                global_step=global_step)

            self.saver = tf.train.Saver(self.trainable_variables)
            self.var_init = tf.variables_initializer(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope))

    def compile(self, session, save_dir="./saves/dncwrapper/", save_name = "dncwrapper"):
        """ making the wrapper usable """
        self.session = session
        self.save_dir = save_dir
        self.save_name = save_name
        self.session.run([self.var_init, self.dnc_model_state_init])

    def _complete_save_dir_name(self, save_dir, save_name):
        if not save_dir:
            save_dir = self.save_dir
        if not save_name:
            save_name = self.save_name
        return save_dir, save_name

    def save(self, save_dir=None, save_name=None):
        save_dir, save_name = self._complete_save_dir_name(save_dir, save_name)
        os.makedirs(save_dir, exist_ok=True)
        os.remove(save_dir, exist_ok=True)  # removes unnecessary last folder that is created
        self.saver.save(self.session, save_dir+save_name)

    def restore(self, save_dir=None, save_name=None):
        save_dir, save_name = self._complete_save_dir_name(save_dir, save_name)
        self.saver.restore(self.session, self.save_dir+self.save_name)

    def get_weights(self):
        return np.array([w.eval() for w in self.trainable_variables])

    def set_weights(self, weights):
        self.session.run(self.weight_setter_node, feed_dict=dict(zip(self.weight_setter_feeds, weights)))

    # evaluates the tensors of the initial state, not necessary for intermediate steps
    def _get_state_as_dict(self, state, state_dict={}):
        for s in state:
            if not (type(s) == tf.Tensor):
                state_dict = self._get_state_as_dict(s, state_dict)
            else:
                state_dict[s] = s.eval()
        return state_dict

    def get_state_init(self):
        return self._get_state_as_dict(self.dnc_model_state_init)

    def predict(self, input_sequence):
        """ predicts the output based on a zero internal state """
        return self.session.run(self.prediction, feed_dict={self.input_placeholder: input_sequence})

    def predict_with_state(self, model_state, input_sequence):
        """ predicts the output based on an optionally given state dict/DNCState, returns prediction and state """
        feed_dict = {self.input_placeholder: input_sequence}
        if type(model_state) is dnc.DNCState:
            feed_dict[self.dnc_model_state_init] = model_state
            pass
        elif type(model_state) is dict:
            # required when feeding something something for the first time
            feed_dict.update(model_state)
        prediction, state = self.session.run([self.prediction, self.dnc_model_state], feed_dict=feed_dict)
        return prediction, state

    def train(self, input_seq, target_seq, mask, iterations=1):
        total_loss, last_prediction, last_output = 0, False, False
        feed_dict = {self.input_placeholder: input_seq,
                     self.target_placeholder: target_seq,
                     self.mask_placeholder: mask}
        for i in range(iterations):
            last_prediction, _, loss, last_output = self.session.run([self.prediction,
                                                                      self.train_step_node,
                                                                      self.train_loss_node,
                                                                      self.output_node],
                                                                     feed_dict=feed_dict)
            total_loss += loss
        return total_loss, last_prediction, last_output

    def train_with_state(self, model_state, input_seq, target_seq, mask, iterations=1):
        total_loss, last_prediction, last_output = 0, False, False
        feed_dict = {self.input_placeholder: input_seq,
                     self.target_placeholder: target_seq,
                     self.mask_placeholder: mask}
        if type(model_state) is dnc.DNCState:
            feed_dict[self.dnc_model_state_init] = model_state
            pass
        elif type(model_state) is dict:
            feed_dict.update(model_state)
        for i in range(iterations):
            last_prediction, _, loss, last_output = self.session.run([self.prediction,
                                                                      self.train_step_node,
                                                                      self.train_loss_node,
                                                                      self.output_node],
                                                                     feed_dict=feed_dict)
            total_loss += loss
        return total_loss, last_prediction, last_output
