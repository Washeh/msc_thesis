import gym
import tensorflow as tf

import matplotlib.pyplot as plt
import pandas as pd

from rl_dnc.model.AC_dnc_wrapper import AC_DNC_Wrapper, StatsHandler


def print_feedback_fun(feed):
    num_episodes = len(feed['episode_rewards'])
    avg_reward = sum(feed['episode_rewards']) / num_episodes
    avg_actor_loss = sum([abs(l) for l in feed['actor_losses']]) / num_episodes
    avg_critic_loss = sum([abs(l) for l in feed['critic_losses']]) / num_episodes
    print('iteration %d/%d, avg reward: %.2f' % (feed['cur_iteration']+1, feed['max_iterations'], avg_reward))
    print('  avg abs a loss: %.2f' % avg_actor_loss)
    print('  avg abs c loss: %.2f' % avg_critic_loss)
    return False, False


def cancel_feedback_fun(feed):
    num_episodes = len(feed['episode_rewards'])
    avg_reward = sum(feed['episode_rewards']) / num_episodes
    avg_actor_loss = sum([abs(l) for l in feed['actor_losses']]) / num_episodes
    # cancel training when the reward is high, and the actor is quite sure about its actions
    if avg_reward > 199 and avg_actor_loss < 250:
        return True, False
    return False, False


def saver_feedback_fun(feed):
    # save each 10 iterations
    if (feed['cur_iteration']+1) % 10 == 0:
        return False, True
    return False, False


def main():
    tf.logging.set_verbosity(3)  # Print INFO log messages.
    batch_size = 1
    environments = [gym.make("CartPole-v0") for _ in range(batch_size)]

    path = "./../../saves/ac_dnc_wrapper/Cartpole_solved/"
    max_iterations = 0
    episodes_per_update = 5
    max_steps = 200
    play_episodes = 5
    render = True

    sh = None

    if max_iterations > 0 or play_episodes > 0:
        ac = AC_DNC_Wrapper(environments, episodes_per_update, scope="ac_dnc_wrapper")
        ac.add_training_feedback_function(print_feedback_fun)
        ac.add_training_feedback_function(cancel_feedback_fun)
        ac.add_training_feedback_function(saver_feedback_fun)

        with tf.Session() as sess:
            ac.compile(sess, save_dir=path)
            ac.load()
            ac.train(max_iterations, max_steps)
            ac.play_episodes(play_episodes, max_steps, render=render)
            ac.save()
            sh = ac.get_stats_handler()
    else:
        sh = StatsHandler()
        sh.set_path(path)

    sh.plot_columns(show=True)


if __name__ == '__main__':
    main()
