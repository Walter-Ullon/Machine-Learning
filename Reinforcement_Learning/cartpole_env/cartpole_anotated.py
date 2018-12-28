#!/usr/bin/env python3
# ======================================================================================================================
#                                       CROSS-ENTROPY METHOD  --- PROCEDURE
# ======================================================================================================================
#               The core of the cross-entropy method is to throw away bad episodes and train on better ones.
# 1. Play (NUM_EPISODES) number of episodes using current model and environment.
# 2. Calculate the total reward for every episode and decide on a reward boundary. Usually, some percentile of all
#    rewards such as 50th of 70th.
# 3. Throw away all episodes with rewards below the established percentile boundary.
# 4. Train on the remaining "elite" episodes i.e. above the percentiles, using observations as the input and issued
#    actions as the desired output (basically, turn it into a supervised learning problem).
# 5. Repeat until satisfied with the results.
#
# NOTE: "batch" is comprised of up to 16 episodes, and each episode is comprised of n number of steps --> (observation,
#        action, reward)

# ======================================================================================================================
#                                            IMPORT LIBRARIES
# ======================================================================================================================
import gym
from collections import namedtuple
import numpy as np
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim


# ======================================================================================================================
#                                       NEURAL NETWORK CONSTRUCTOR CLASS
# ======================================================================================================================
class Net(nn.Module):
    # num. of inputs ----> obs_size
    # num. of nodes in first hidden layer ----> hidden_size
    # num. of outputs  ----> n_actions
    # activation fn ----> ReLU
    # !!!NOTE: softmax to be applied to outputs later in the process...(increases numerical stability)
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
            # sm = nn.Softmax(dim=1)
        )

    # forward pass function (takes in "observations" as tensor and send it through the network):
    def forward(self, observations_tensor):
        return self.net(observations_tensor)


# ======================================================================================================================
#                                   BATCH GENERATOR FUNCTION (W/ EPISODES)
# ======================================================================================================================
# GENERATOR FUNCTION:
# accepts the environment (cartpole), as well as our neural network and the number of episodes it should generate in
# every iteration:
def iterate_batches(env, net, NUM_EPISODES):
    # accumulates list of "Episode" object instances:
    batch = []
    # current episode reward counter:
    episode_reward = 0.0
    # list of steps ("EpisodeStep" objects)
    episode_steps = []
    # reset env. and obtain first observation:
    obs = env.reset()
    # create softmax layer to turn network output to probability distribution over actions:
    sm = nn.Softmax(dim=1)

    # environment loop:
    while True:
        # turn observation vector into torch tensor:
        obs_v = torch.FloatTensor([obs])

        # pass observation to neural network:
        act_probs_raw = net(obs_v)

        # get probability distribution over actions by applying softmax:
        act_probs_v = sm(act_probs_raw)

        # unpack tensor data field to extract probabilities and turn to numpy array:
        act_probs = act_probs_v.data.numpy()[0]

        # sample random action from distribution:
        action = np.random.choice(len(act_probs), p=act_probs)

        # pass sampled action to environment to obtain next observation, reward, and episode status:
        next_obs, reward, is_done, _ = env.step(action)

        # append reward to list:
        episode_reward += reward

        # append "EpisodeStep" object to list after recording observation and action:
        # NOTE: we append the "obs" that we used to choose the action, not the "obs" that is the result of the action.
        episode_steps.append(EpisodeStep(observation=obs, action=action))

        # if "is_done" ---> pole fell down, game over!
        if is_done:
            # append total reward plus collection of "EpisodeStep" objects to "Episode" tuple:
            batch.append(Episode(reward=episode_reward, steps=episode_steps))
            # reset episode reward:
            episode_reward = 0.0
            # reset episode steps:
            episode_steps = []
            # reset env (to get 1st obs. again)
            next_obs = env.reset()

            # if we have played enough episodes:
            if len(batch) == NUM_EPISODES:
                # return control to outer iteration loop and yield "batch" iterator object:
                yield batch

                # clean up batch:
                batch = []

        # capture next observation:
        obs = next_obs

# One very important fact to understand in this function logic is that the training of our network and the generation
# of our episodes are performed at the same time. They are not completely in parallel, but every time our loop
# accumulates enough episodes (16), it passes control to this function caller, which is supposed to train the network
# using the gradient descent. So, when yield is returned, the network will have different, slightly better behavior.


# ======================================================================================================================
#                                       BATCH FILTERING FUNCTION
# ======================================================================================================================
# This function is at the core of the cross-entropy method: from the given batch of episodes and percentile value,
# it calculates a boundary reward, which is used to filter elite episodes to train on. To obtain the boundary reward,
# we're using NumPy's percentile function, which from the list of values and the desired percentile, calculates the
# percentile's value. Then we will calculate mean reward, which is used only for monitoring.

def filter_batch(batch, percentile):
    # from "batch" list of "Episode" objects, extract their rewards and populate a list:
    rewards = list(map(lambda s: s.reward, batch))

    # obtain (70th) percentile from list of rewards:
    reward_bound = np.percentile(rewards, percentile)

    # obtain mean of rewards:
    reward_mean = float(np.mean(rewards))

    # set lists to hold observations and actions for training (i.e. input and target):
    train_obs = []
    train_act = []

    # check each episode's reward against boundary:
    for episode in batch:
        if episode.reward < reward_bound:
            continue
        # append observation lists:
        train_obs.extend(map(lambda step: step.observation, episode.steps))
        # append actions taken from the observations:
        train_act.extend(map(lambda step: step.action, episode.steps))

    # turn action and observation lists into torch tensors:
    train_obs_v = torch.FloatTensor(train_obs)
    train_act_v = torch.LongTensor(train_act)

    # return 4-tuple:
    return train_obs_v, train_act_v, reward_bound, reward_mean


# ======================================================================================================================
#                                             TRAINING LOOP
# ======================================================================================================================
# first hidden layer size:
HIDDEN_SIZE = 128

# number of episodes in every iteration:
NUM_EPISODES = 30

# reward percentile cutoff/filter:
PERCENTILE = 70

# (object) records single step taken during episode, captures env. observation and action (pair) taken:
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])

# (object) records single, undiscounted episode reward and collection of EpisodeStep:
Episode = namedtuple('Episode', field_names=['reward', 'steps'])

# begin loop:
if __name__ == "__main__":
    # make Cartpole environment instance:
    env = gym.make("CartPole-v0")

    # set environment monitoring directory (saves movies of training):
    env = gym.wrappers.Monitor(env, directory="mon", force=True)

    # get the size of the observation space (for determining network input size):
    obs_size = env.observation_space.shape[0]

    # get the size of the action space (for determining network output size):
    n_actions = env.action_space.n

    # create neural network instance (from "Net" class):
    net = Net(obs_size, HIDDEN_SIZE, n_actions)

    # loss measure function (combines both softmax and cross-entropy in a single, more numerically stable expression):
    objective = nn.CrossEntropyLoss()

    # create optimizer function:
    optimizer = optim.Adam(params=net.parameters(), lr=0.01)

    # create SummaryWriter instance for TensorBoard:
    writer = SummaryWriter(comment="-cartpole")

    # set threshold for the mean of reward, used to kill loop once satisfied:
    reward_mean_thresh = 199

    # keep "yielding" batch instances until reward_mean_thresh is exceeded:
    for iter_no, batch in enumerate(iterate_batches(env, net, NUM_EPISODES)):
        # for every batch received as input, filter it according to percentile and return 4-tuple:
        obs_v, acts_v, reward_b, reward_m = filter_batch(batch, PERCENTILE)

        # "zero" the network gradients in anticipation of training:
        optimizer.zero_grad()

        # pass observations to network and obtain action scores:
        action_scores_v = net(obs_v)

        # calculate cross-entropy between the network output and the actions that the agent took
        loss_v = objective(action_scores_v, acts_v)

        # calculate gradients on the loss:
        loss_v.backward()

        # ask optimizer to adjust the network:
        optimizer.step()

        # print monitoring stats:
        print("%d: loss=%.3f, reward_mean=%.1f, reward_bound=%.1f" % (
            iter_no, loss_v.item(), reward_m, reward_b))
        # write values to tensorboard:
        writer.add_scalar("loss", loss_v.item(), iter_no)
        writer.add_scalar("reward_bound", reward_b, iter_no)
        writer.add_scalar("reward_mean", reward_m, iter_no)

        # check to see if mean of reward values has exceeded threshold:
        if reward_m > reward_mean_thresh:
            print("Solved!")
            break

    # kill writer:
    writer.close()

