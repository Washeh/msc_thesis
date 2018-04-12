import os
import logging
import gym
import numpy as np
import tensorflow as tf
import time
from baselines.acktr.utils import conv, fc, dense, conv_to_fc, sample, kl_div
from baselines import logger
from baselines.common import set_global_seeds
from baselines import bench
from baselines.acktr.acktr_disc import learn
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
import baselines.common.tf_util as U

from baselines.a2c.utils import batch_to_seq, seq_to_batch, lstm


# based on https://github.com/openai/baselines/blob/master/baselines/acktr/policies.py
class CustomPolicyLstm(object):
    def __init__(self, sess, ob_space, ac_space, nenv, nsteps, nstack, nlstm=128, reuse=False):
        scope = "model"
        nbatch = nenv*nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc*nstack)
        nact = ac_space.n
        X = tf.placeholder(tf.uint8, ob_shape, name="observations") #obs
        M = tf.placeholder(tf.float32, [nbatch], name="mask") #mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, nlstm*2], name="states") #states
        with tf.variable_scope(scope, reuse=reuse):
            h = conv(tf.cast(X, tf.float32)/255., 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2))
            h2 = conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2))
            h3 = conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2))
            h3 = conv_to_fc(h3)
            h4 = fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2))
            xs = batch_to_seq(h4, nenv, nsteps)
            ms = batch_to_seq(M, nenv, nsteps)
            h5, snew = lstm(xs, ms, S, 'lstm1', nh=nlstm)
            h5 = seq_to_batch(h5)
            pi = fc(h5, 'pi', nact, act=lambda x:x)
            vf = fc(h5, 'v', 1, act=lambda x:x)

            trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
            self._saver = tf.train.Saver(trainable_vars)

        v0 = vf[:, 0]
        a0 = sample(pi)
        self.initial_state = np.zeros((nenv, nlstm*2), dtype=np.float32)

        def step(ob, state, mask):
            a, v, s = sess.run([a0, v0, snew], {X:ob, S:state, M:mask})
            return a, v, s

        def value(ob, state, mask):
            return sess.run(v0, {X:ob, S:state, M:mask})

        def save(path, name):
            try:
                os.makedirs(path)
            except FileExistsError:
                pass
            self._saver.save(sess, path+name)

        def load(path, name):
            if os.path.exists(path+name+'.index'):
                self._saver.restore(sess, path+name)
            else:
                tf.logging.warn('Failed restoring vars from %s' % path)

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value
        self.save = save
        self.load = load


# based on https://github.com/openai/baselines/blob/master/baselines/acktr/policies.py
class CustomPolicyCnn(object):
    def __init__(self, sess, ob_space, ac_space, nenv, nsteps, nstack, reuse=False):
        scope = "model"
        nbatch = nenv*nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc*nstack)
        nact = ac_space.n
        X = tf.placeholder(tf.uint8, ob_shape) #obs
        with tf.variable_scope(scope, reuse=reuse):
            h = conv(tf.cast(X, tf.float32)/255., 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2))
            h2 = conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2))
            h3 = conv(h2, 'c3', nf=32, rf=3, stride=1, init_scale=np.sqrt(2))
            h3 = conv_to_fc(h3)
            h4 = fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2))
            pi = fc(h4, 'pi', nact, act=lambda x:x)
            vf = fc(h4, 'v', 1, act=lambda x:x)

            trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
            self._saver = tf.train.Saver(trainable_vars)

        v0 = vf[:, 0]
        a0 = sample(pi)
        self.initial_state = [] #not stateful

        def step(ob, *_args, **_kwargs):
            a, v = sess.run([a0, v0], {X:ob})
            return a, v, [] #dummy state

        def value(ob, *_args, **_kwargs):
            return sess.run(v0, {X:ob})

        def save(path, name):
            os.makedirs(path, exist_ok=True)
            self._saver.save(sess, path+name)

        def load(path, name):
            if os.path.exists(path+name+'.index'):
                self._saver.restore(sess, path+name)
            else:
                tf.logging.warn('Failed restoring vars from %s' % path)

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value
        self.save = save
        self.load = load


class PolicyWrapper:
    def __init__(self, path, policy_type, try_load=True):
        self._policy_type = policy_type
        self._try_load = try_load
        self.step_model = False
        self.train_model = False
        self._step_model_path_append = 'step_model/'
        self._train_model_path_append = 'train_model/'
        self._path = path
        self._model_name = 'model'

    def __call__(self, sess, ob_space, ac_space, nenv, nsteps, nstack, reuse=False):
        policy = self._policy_type(sess, ob_space, ac_space, nenv, nsteps, nstack, reuse=reuse)
        if nsteps == 1:
            self.step_model = policy
            if self._try_load:
                policy.load(self._path + self._step_model_path_append, self._model_name)
        else:
            self.train_model = policy
            if self._try_load:
                policy.load(self._path + self._train_model_path_append, self._model_name)
        return policy

    def save(self):
        if self.step_model:
            self.step_model.save(self._path + self._step_model_path_append, self._model_name)
        if self.train_model:
            self.step_model.save(self._path + self._train_model_path_append, self._model_name)


def train(env_id, policy_fn, num_timesteps, seed, num_cpu):
    def make_env(rank):
        def _thunk():
            env = make_atari(env_id)
            env.seed(seed + rank)
            env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))
            gym.logger.setLevel(logging.WARN)
            return wrap_deepmind(env)
        return _thunk
    set_global_seeds(seed)
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    learn(policy_fn, env, seed, total_timesteps=int(num_timesteps * 1.1), nprocs=num_cpu)
    env.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='BreakoutNoFrameskip-v4')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(10e6))
    args = parser.parse_args()
    logger.configure()

    tf.logging.set_verbosity(3)  # Print INFO log messages.
    policy = PolicyWrapper('./../saves/baselines/acktr/test/', CustomPolicyLstm)
    tstart = time.time()
    # train(args.env, policy_fn, num_timesteps=args.num_timesteps, seed=args.seed, num_cpu=8)
    train(args.env, policy, num_timesteps=1000, seed=args.seed, num_cpu=8)
    policy.save()
    tf.logging.info('Training done after %.2f seconds' % (time.time()-tstart))
    # 1000 -> 5, 6.13
    # 10000 -> 22s
    # 100000 -> 202s


if __name__ == '__main__':
    main()


# TODO problem when stateful?
