#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 07:54:02 2019

@author: cryo
"""

import os
import glob
import struct
import numpy as np


def read_stream_data_gcp_save(datafile, channel=None,
                              unwrap=True, downsample=1, n_samp=None):
    """
        Reads the special data that is designed to be a copy of the GCP data.

        Args:
        -----
        datafile (str): The full path to the data made by stream_data_on
        
        Opt Args:
        ---------
        channel (int or int array): Channels to load.
        unwrap (bool) : Whether to unwrap units of 2pi. Default is True.
        downsample (int): The amount to downsample.

        Ret:
        ----
        t (float array): The timestamp data
        d (float array): The resonator data in units of phi0
        m (int array): The maskfile that maps smurf num to gcp num
        """
    try:
        datafile = glob.glob(datafile + '*')[-1]
    except:
        print('datafile=%s' % datafile)

    keys = ['protocol_version', 'crate_id', 'slot_number', 'number_of_channels',
            'rtm_dac_config0', 'rtm_dac_config1', 'rtm_dac_config2',
            'rtm_dac_config3', 'rtm_dac_config4', 'rtm_dac_config5',
            'flux_ramp_increment', 'flux_ramp_start', 'rate_since_1Hz',
            'rate_since_TM', 'nanoseconds', 'seconds', 'fixed_rate_marker',
            'sequence_counter', 'tes_relay_config', 'mce_word',
            'user_word0', 'user_word1', 'user_word2'
            ]

    data_keys = [f'data{i}' for i in range(528)]

    keys.extend(data_keys)
    keys_dict = dict(zip(keys, range(len(keys))))  # Read in all channels by default
    if channel is None:
        channel = np.arange(512)

    channel = np.ravel(np.asarray(channel))
    n_chan = len(channel)

    # Indices for input channels
    channel_mask = np.zeros(n_chan, dtype=int)
    for i, c in enumerate(channel):
        channel_mask[i] = keys_dict['data{}'.format(c)]

    eval_n_samp = False
    if n_samp is not None:
        eval_n_samp = True

    # Make holder arrays for phase and timestamp
    phase = np.zeros((n_chan, 0))
    timestamp2 = np.array([])
    counter = 0
    n = 20000  # Number of elements to load at a time
    tmp_phase = np.zeros((n_chan, n))
    tmp_timestamp2 = np.zeros(n)
    with open(datafile, mode='rb') as file:
        while True:
            chunk = file.read(2240)  # Frame size is 2240
            if not chunk:
                # If frame is incomplete - meaning end of file
                phase = np.hstack((phase, tmp_phase[:, :counter % n]))
                timestamp2 = np.append(timestamp2, tmp_timestamp2[:counter % n])
                break
            elif eval_n_samp:
                if counter >= n_samp:
                    phase = np.hstack((phase, tmp_phase[:, :counter % n]))
                    timestamp2 = np.append(timestamp2,
                                           tmp_timestamp2[:counter % n])
                    break
            frame = struct.Struct('3BxI6Q8I5Q528i').unpack(chunk)
            # 3 unsigned char+pad byte+ unsigned int+6 unsigned longlong+8 unsigned int
            # 5 unsinged longlong +528 int

            # Extract detector data
            for i, c in enumerate(channel_mask):
                tmp_phase[i, counter % n] = frame[c]
            # Timestamp data
            tmp_timestamp2[counter % n] = frame[keys_dict['rtm_dac_config5']]

            # Store the data in a useful array and reset tmp arrays
            if counter % n == n - 1:
                phase = np.hstack((phase, tmp_phase))
                timestamp2 = np.append(timestamp2, tmp_timestamp2)
                tmp_phase = np.zeros((n_chan, n))
                tmp_timestamp2 = np.zeros(n)
            counter = counter + 1

    phase = np.squeeze(phase)
    phase = phase.astype(float) / 2 ** 15 * np.pi  # where is decimal?  Is it in rad?

    rootpath = os.path.dirname(datafile)
    filename = os.path.basename(datafile)
    timestamp = filename.split('.')[0]

    mask = make_mask_lookup(os.path.join(rootpath, '{}_mask.txt'.format(timestamp)))
    # mask = self.make_mask_lookup(datafile.replace('.dat','_mask.txt'))

    return timestamp2, phase, mask


def make_mask_lookup(mask_file):
    """
        Makes an n_band x n_channel array where the elements correspond
        to the smurf_to_mce mask number. In other workds, mask[band, channel]
        returns the GCP index in the mask that corresonds to band, channel.
    
        Args:
        -----
        mask_file (str): The full path the a mask file
    
        Ret:
        ----
        mask_lookup (int array): An array with the GCP numbers.
        """
    mask = np.atleast_1d(np.loadtxt(mask_file))
    bands = np.unique(mask // 512).astype(int)
    ret = np.ones((np.max(bands) + 1, 512), dtype=int) * -1

    for gcp_chan, smurf_chan in enumerate(mask):
        ret[int(smurf_chan // 512), int(smurf_chan % 512)] = gcp_chan

    return ret
