import tensorflow as tf
import dnc
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt


FLAGS = tf.flags.FLAGS

# Model parameters
tf.flags.DEFINE_integer("hidden_size", 64, "Size of LSTM hidden layer.")
tf.flags.DEFINE_integer("memory_size", 16, "The number of memory slots.")
tf.flags.DEFINE_integer("word_size", 16, "The width of each memory slot.")
tf.flags.DEFINE_integer("num_write_heads", 1, "Number of memory write heads.")
tf.flags.DEFINE_integer("num_read_heads", 4, "Number of memory read heads.")
tf.flags.DEFINE_integer("clip_value", 200000, "Maximum absolute value of controller and dnc outputs.")

# RL parameters
tf.flags.DEFINE_float("discount", 0.999, "Discount factor (gamma) for future rewards.")

# Optimizer parameters.
tf.flags.DEFINE_float("max_grad_norm", 50, "Gradient clipping norm limit.")
tf.flags.DEFINE_float("learning_rate", 1e-3, "Optimizer learning rate.")
tf.flags.DEFINE_float("optimizer_epsilon", 1e-10,
                      "Epsilon used for RMSProp optimizer.")


class StatsHandler:

    def __init__(self, file_name="stats.csv"):
        self._path = ""
        self._file_name = file_name
        self._column_names = ['episode_rewards', 'actor_losses', 'critic_losses']
        self._file_exists = False
        self._stored_data = []
        pass

    def set_path(self, path):
        self._path = path + 'meta/'
        try:
            os.makedirs(self._path)
        except FileExistsError:
            pass

    def on_model_save(self, path=False):
        if path:
            self.set_path(path)
        for df in self._stored_data:
            df.to_csv(self._path+self._file_name,
                      mode='a',
                      header=(not self._file_exists),
                      index=False)
            self._file_exists = True
        self._stored_data = []

    def on_model_load(self, path=False):
        if path:
            self.set_path(path)
        if os.path.exists(self._path+self._file_name):
            self._file_exists = True

    def on_model_training(self, feed):
        df = pd.DataFrame(dict(zip(self._column_names,
                                   [feed['episode_rewards'], feed['actor_losses'], feed['critic_losses']])),
                          columns=self._column_names)
        self._stored_data.append(df)
        return False, False  # as this is used as a training feedback function, return False to continue training

    def plot_columns(self, show=False):
        df = pd.read_csv(self._path+self._file_name)
        plt.figure(1)
        for i in range(len(self._column_names)):
            col_name = self._column_names[i]
            plt.subplot(len(self._column_names), 1, i+1)
            plt.plot(df[col_name], 'bo')
            plt.plot(df[col_name].rolling(window=20).mean(), 'r', linewidth=3)
            plt.ylabel(col_name)
        if show:
            plt.show()


class AC_DNC_Wrapper:

    def __init__(self, environments, episodes_per_update, scope="ac_dnc_wrapper"):
        self._dtype = tf.float32
        self._gamma = FLAGS.discount
        self._environments = environments
        self._scope = scope
        self._actor_scope = "actor"
        self._critic_scope = "critic"
        self._actor_save_path = self._actor_scope + '/actor'
        self._critic_save_path = self._critic_scope + '/critic'

        self._input_size = self._environments[0].observation_space.shape[0]
        self._output_size = self._environments[0].action_space.n
        self._batch_size = 1  # len(self._environments)
        self._episodes_per_update = episodes_per_update

        # set in the reset methods
        self._current_actor_state = None
        self._current_critic_state = None
        self._episode_transitions = []

        # set in the compile method
        self._stats_handler = StatsHandler()  # just path
        self._save_dir = None
        self._session = None
        self._actor_zero_state_input = None
        self._critic_zero_state_input = None

        # set in add feedback functions
        self._training_feedback_functions = []

        with tf.variable_scope(scope):

            # ------- actor ------- #

            # model
            self._actor_input,\
                self._actor_state_input,\
                self._actor_probabilities,\
                self._actor_state_output = self._get_actor_model(self._actor_scope)

            # step var
            self._actor_global_step = tf.get_variable(
                name="actor_global_step",
                shape=[],
                dtype=tf.int64,
                initializer=tf.zeros_initializer(),
                trainable=False,
                collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.GLOBAL_STEP])

            # saver
            self._trainable_actor_variables =\
                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope+'/'+self._actor_scope)
            self._actor_saver = tf.train.Saver(self._trainable_actor_variables)

            # optimizing
            self._actor_choices_input,\
                self._actor_advantages_input,\
                self._actor_training_loss,\
                self._actor_gradients,\
                self._actor_training_step = self._get_actor_optimizer(self._actor_scope)

            # ------- critic ------- #

            # model
            self._critic_obs_input,\
                self._critic_act_input,\
                self._critic_state_input,\
                self._critic_expectation,\
                self._critic_state_output = self._get_critic_model(self._critic_scope)

            # step var
            self._critic_global_step = tf.get_variable(
                name="critic_global_step",
                shape=[],
                dtype=tf.int64,
                initializer=tf.zeros_initializer(),
                trainable=False,
                collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.GLOBAL_STEP])

            # saver
            self._trainable_critic_variables =\
                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope+'/'+self._critic_scope)
            self._critic_saver = tf.train.Saver(self._trainable_critic_variables)

            # optimizing
            self._critic_target_input,\
                self._critic_training_loss,\
                self._critic_gradients,\
                self._critic_training_step = self._get_critic_optimizer(self._critic_scope)

        self._global_init = tf.global_variables_initializer()
        self._reset_dnc_states()

    @staticmethod
    def _get_dnc_configs():
        access_config = {
            "memory_size": FLAGS.memory_size,
            "word_size": FLAGS.word_size,
            "num_reads": FLAGS.num_read_heads,
            "num_writes": FLAGS.num_write_heads,
        }
        controller_config = {
            "hidden_size": FLAGS.hidden_size,
        }
        return access_config, controller_config, FLAGS.clip_value

    def _get_actor_model(self, scope):
        """ output size refers to the action distribution """
        with tf.variable_scope(scope):
            access_config, controller_config, clip_value = self._get_dnc_configs()
            dnc_model = dnc.DNC(access_config, controller_config, self._output_size, clip_value)
            dnc_model_state_init = dnc_model.initial_state(self._batch_size, dtype=self._dtype)
            input_ph = tf.placeholder(shape=(None, self._batch_size, self._input_size), dtype=self._dtype, name='input_ph')
            dnc_output, dnc_model_state = tf.nn.dynamic_rnn(
                cell=dnc_model,
                inputs=input_ph,
                time_major=True,
                initial_state=dnc_model_state_init)
            soft_output = tf.nn.softmax(logits=dnc_output, dim=-1, name="output_softmax")
            return input_ph, dnc_model_state_init, soft_output, dnc_model_state

    def _get_actor_optimizer(self, scope):
        with tf.variable_scope(scope):
            choices = tf.placeholder(shape=(None, self._batch_size, self._output_size),
                                     dtype=self._dtype, name='choices_input_ph')
            advantages = tf.placeholder(shape=(None, self._batch_size, 1),
                                        dtype=self._dtype, name='advantages_input_ph')
            chosen_probabilities = tf.reduce_sum(tf.multiply(self._actor_probabilities, choices), axis=2)
            eligibility = tf.log(chosen_probabilities) * advantages
            loss_batch = -tf.reduce_mean(eligibility, axis=0)
            loss = tf.reduce_sum(loss_batch)

            # optimizer, gradient clipping
            grads, _ = tf.clip_by_global_norm(
                tf.gradients(loss, self._trainable_actor_variables),
                FLAGS.max_grad_norm)
            optimizer = tf.train.RMSPropOptimizer(
                FLAGS.learning_rate,
                epsilon=FLAGS.optimizer_epsilon)
            # optimizer = tf.train.SyncReplicasOptimizer(optimizer, replicas_to_aggregate=self._episodes_per_update)
            train_step = optimizer.apply_gradients(
                zip(grads, self._trainable_actor_variables),
                global_step=self._actor_global_step)

            return choices, advantages, loss, grads, train_step

    def _get_critic_model(self, scope):
        with tf.variable_scope(scope):
            access_config, controller_config, clip_value = self._get_dnc_configs()
            dnc_model = dnc.DNC(access_config, controller_config, 1, clip_value)
            dnc_model_state_init = dnc_model.initial_state(self._batch_size, dtype=self._dtype)
            obs_input_ph = tf.placeholder(shape=(None, self._batch_size, self._input_size),
                                          dtype=self._dtype, name='obs_input_ph')
            act_input_ph = tf.placeholder(shape=(None, self._batch_size, self._output_size),
                                          dtype=self._dtype, name='act_input_ph')
            concat_input = tf.concat([obs_input_ph, act_input_ph], 2)
            dnc_output, dnc_model_state = tf.nn.dynamic_rnn(
                cell=dnc_model,
                inputs=concat_input,
                time_major=True,
                initial_state=dnc_model_state_init)
            return obs_input_ph, act_input_ph, dnc_model_state_init, dnc_output, dnc_model_state

    def _get_critic_optimizer(self, scope):
        with tf.variable_scope(scope):
            target_input_ph = tf.placeholder(shape=(None, self._batch_size, 1),
                                             dtype=self._dtype, name='target_input_ph')
            # calculate loss
            diff = tf.squared_difference(self._critic_expectation, target_input_ph)
            loss_time_batch = tf.reduce_sum(diff, axis=2)
            loss_batch = tf.reduce_mean(loss_time_batch, axis=0)
            loss = tf.reduce_sum(loss_batch) / self._batch_size

            # optimizer, gradient clipping
            grads, _ = tf.clip_by_global_norm(
                tf.gradients(loss, self._trainable_critic_variables),
                FLAGS.max_grad_norm)
            optimizer = tf.train.RMSPropOptimizer(
                FLAGS.learning_rate,
                epsilon=FLAGS.optimizer_epsilon)
            # optimizer = tf.train.SyncReplicasOptimizer(optimizer, replicas_to_aggregate=self._episodes_per_update)
            train_step = optimizer.apply_gradients(
                zip(grads, self._trainable_critic_variables),
                global_step=self._critic_global_step)

            return target_input_ph, loss, grads, train_step

    def _reset_dnc_states(self):
        self._current_actor_state = self._actor_zero_state_input
        self._current_critic_state = self._critic_zero_state_input

    def _reset_episode_transitions(self):
        self._episode_transitions = []

    def _add_episode_transitions(self, observations, actor_states, actions, rewards, total_reward):
        """ sequences of observation, actor_state, action, reward, total reward of episode
            action: one-hot vectors for the picked action """
        future_rewards = np.zeros_like(rewards)
        future_rewards[-1] = total_reward
        for i in range(len(rewards)-2, -1, -1):
            future_reward_in_step = future_rewards[i+1] - rewards[i+1]
            future_rewards[i] = rewards[i] + self._gamma * future_reward_in_step
        self._episode_transitions.append({
            'observations': np.array(observations),
            'actor_states': actor_states,
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'future_rewards': future_rewards
        })

    def add_training_feedback_function(self, fun):
        """ functions are expected to return (bool, bool) to (stop training, save model progress) """
        self._training_feedback_functions.append(fun)

    def compile(self, session, save_dir):
        self._session = session
        self._save_dir = save_dir
        _, self._actor_zero_state_input, self._critic_zero_state_input =\
            self._session.run([self._global_init,
                               self._actor_state_input,
                               self._critic_state_input])
        step_a, step_c = self._session.run([self._actor_global_step, self._critic_global_step])
        # adding the stats handler
        self._stats_handler.set_path(self._save_dir)
        self.add_training_feedback_function(self._stats_handler.on_model_training)

        tf.logging.info('Compiled AC DNC Model:')
        tf.logging.info('  save directory: %s' % self._save_dir)
        tf.logging.info('  %d trainable actor variables' % len(self._trainable_actor_variables))
        tf.logging.info('  %d trainable critic variables' % len(self._trainable_critic_variables))
        tf.logging.info('  global step %d, %d' % (step_a, step_c))

    def save(self, save_dir=None):
        if not save_dir:
            save_dir = self._save_dir
        try:
            os.makedirs(save_dir)
            os.remove(save_dir)  # removes unnecessary last folder that is created
        except:
            pass
        self._actor_saver.save(self._session, save_dir + self._actor_save_path, self._actor_global_step)
        self._critic_saver.save(self._session, save_dir + self._critic_save_path, self._actor_global_step)
        self._stats_handler.on_model_save()

    def load(self, save_dir=None):
        if not save_dir:
            save_dir = self._save_dir
        self._stats_handler.on_model_load()
        actor_path = save_dir+self._actor_save_path
        if os.path.exists(actor_path+'.index'):
            self._actor_saver.restore(self._session, actor_path)
            tf.logging.info('Loaded actor variables from: %s' % actor_path)
        critic_path = save_dir+self._critic_save_path
        if os.path.exists(critic_path+'.index'):
            self._critic_saver.restore(self._session, critic_path)
            tf.logging.info('Loaded critic variables from: %s' % critic_path)

    def get_stats_handler(self):
        return self._stats_handler

    def play_episodes(self, count, max_steps, render=False):
        """ plays episodes """
        # TODO just for batch size 1 now...
        # TODO parallelize multiple environments -> multiple a+c copies?

        batch = 0
        env = self._environments[batch]
        episode_rewards = []

        for i in range(count):
            self._reset_dnc_states()
            observation = env.reset()
            total_episode_reward = 0
            transition_observations = []
            transition_actor_states = []
            transition_actions = []
            transition_rewards = []

            for step in range(max_steps):

                transition_actor_states.append(self._current_actor_state)
                transition_observations.append([observation])

                # predict action distribution
                feed_dict = {self._actor_input: [[observation]]}
                if self._current_actor_state:
                    feed_dict[self._actor_state_input] = self._current_actor_state
                action_probabilities, self._current_actor_state =\
                    self._session.run([self._actor_probabilities, self._actor_state_output], feed_dict=feed_dict)

                # pick random action of the distribution
                action_index = np.random.choice(list(range(self._output_size)), p=action_probabilities[0, 0, :])

                # act in the environment
                observation, reward, done, _ = env.step(action_index)
                total_episode_reward += reward

                # adding to the transitions, sequence of (observation, actor_state, action, reward)
                action = np.zeros(shape=self._output_size, dtype=np.float32)
                action[action_index] = 1

                transition_actions.append([action])
                transition_rewards.append([[reward]])

                if render:
                    env.render()
                if done:
                    if render:
                        tf.logging.info('episode reward: %.1f' % total_episode_reward)
                    break

            # appending the transitions for improving upon them
            self._add_episode_transitions(transition_observations,
                                          transition_actor_states,
                                          transition_actions,
                                          transition_rewards,
                                          total_episode_reward)
            episode_rewards.append(total_episode_reward)

        return episode_rewards

    def train(self, max_iterations, max_steps):
        # TODO apply gradients only after all episodes?
        iterations = int(max_iterations / self._episodes_per_update)
        tf.logging.info('Training %d*%d episodes' % (iterations, self._episodes_per_update))
        for i in range(iterations):
            # clear old info, play new episodes
            self._reset_episode_transitions()
            episode_rewards = self.play_episodes(self._episodes_per_update, max_steps, render=False)
            actor_losses = []
            critic_losses = []

            for transition in self._episode_transitions:
                critic_outputs, c_loss, _ = self._session.run([self._critic_expectation,
                                                               self._critic_training_loss,
                                                               self._critic_training_step],
                                                              feed_dict={
                        self._critic_target_input: transition['future_rewards'],
                        self._critic_obs_input: transition['observations'],
                        self._critic_act_input: transition['actions'],
                    })

                advantages = transition['future_rewards'] - critic_outputs

                a_loss, _ = self._session.run([self._actor_training_loss, self._actor_training_step], feed_dict={
                    self._actor_choices_input: transition['actions'],
                    self._actor_advantages_input: advantages,
                    self._actor_input: transition['observations'],
                })

                actor_losses.append(a_loss)
                critic_losses.append(c_loss)

            stop, save = False, False
            feed = {
                'cur_iteration': i,
                'played_episodes': i*self._episodes_per_update,
                'max_iterations': iterations*self._episodes_per_update,
                'episode_rewards': episode_rewards,
                'actor_losses': actor_losses,
                'critic_losses': critic_losses,
            }
            for fun in self._training_feedback_functions:
                stop_fb, save_fb = fun(feed)
                stop, save = stop or stop_fb, save or save_fb
            if save:
                self.save()
            if stop:
                break
