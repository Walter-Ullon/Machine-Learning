#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 09:11:49 2017

@author: walterullon
"""

import pandas as pd
import numpy as np

# define number of draws for the simulation:
n_draws = 10000

# populate a list with n_draws # of probabilities from a uniform dist: 
# (non-informative prior, we assume all probs equally likely)
prior = pd.Series(np.random.uniform(0,1, size = n_draws))
# plot probability dist:
prior.hist()


# define generative model (binomial dist(n_obs, prob)):
def gen_model(prob):
    return (np.random.binomial(16, prob))

# simulate # of susbcribers using the gen_model:
subscribers = []

# loop through the probs list in 'prior' and pass them into the gen_model:
# append to subs list:
for p in prior:
    subscribers.append(gen_model(p))

# true number of observed subscriptions:
observed = 10

# find the number of simulations in 'subs' that match the observations (6):
post_rate = prior[list(map(lambda x: x == observed, subscribers))]
# plot the posterior:
post_rate.hist()

# find the number of signups if methods is used on 100 ppl:
signups = pd.Series([np.random.binomial(n = 100, p = p) for p in post_rate])
signups.hist()


