#!/usr/bin/env python3

# author: zatkins

import numpy as np

# inputs
exclude_temps = [0.0,]

# get metadata
d = np.load('metadata.npy', allow_pickle = True).item()

# file names
fbase = '/home/zatkins/so/data/smurf_data/{}/outputs/'
goodchans_fn = '{}_goodchans.txt'

# initialize object to save
out = {}

# iterate over band
for band in d:
    out[band] = {}

    # iterate over tbath
    for tbath in d[band]:

        # initialize all possible channels
        minset = np.arange(0, 4096)

        # iterate over cl temps
        for cl in d[band][tbath]:

            # skip
            if cl in exclude_temps: continue

            # get metadata
            dirs, ctime = d[band][tbath][cl]

            # get goodchans
            with open(fbase.format(dirs) + goodchans_fn.format(ctime)) as f:
                goodchans = np.array(f.readline().split(',')).astype(int)

            # update minset
            minset = np.intersect1d(minset, goodchans)

        # add minset to dict
        out[band][tbath] = minset

# add exclude temps
out['exclude_temps'] = exclude_temps

# save 
np.save('min_goodchans.npy', np.array(out))