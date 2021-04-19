import numpy as np
import matplotlib.pyplot as plt
import pysmurf
import time
import os

S = pysmurf.SmurfControl(setup=False,
    cfg_file='/usr/local/controls/Applications/smurf/pysmurf/' + 
    'pysmurf/cfg_files/experiment_kx_mapodaq.cfg',
    make_logfile=False)

savedir = '/home/cryo/ey/nonlinear_lms'
bias_vals = np.arange(8, 1, -.5)
lms_freqs = np.arange(12200, 12801, 50)
lms_freqs_tmp = np.arange(12250, 12601, 50)
bias_line_resistance = S.bias_line_resistance


def plot_sib(channels=None, show_plot=True, rs=.003):
    if show_plot:
        plt.ion()
    else:
        plt.ioff()
    s = np.zeros((len(lms_freqs_tmp), len(bias_vals), 123))
    for i, lms in enumerate(lms_freqs_tmp):
        s[i] = np.load(os.path.join(savedir, 'sibs_{}.npy'.format(lms)))

    if channels is None:
        channels = np.arange(123)
    else:
        channels = np.ravel(np.array(channels))

    for c in channels:
        print(s[0,:5,c])
        if np.median(s[0,:5,c]) > 0 :
            s[:,:,c] *= -1

    r = np.abs(rs * (1-1./s))
    siq = (2*s-1)/(rs*np.atleast_3d(bias_vals/bias_line_resistance)) * 1.0E6/1.0E12

    cm = plt.get_cmap('viridis')
    
    for c in channels:
        fig, ax = plt.subplots(3, figsize=(5,8), sharex=True)
        for i, l in enumerate(lms_freqs_tmp):
            color = cm(i/len(lms_freqs_tmp))
            ax[0].plot(bias_vals, s[i,:,c], color=color, label='lms {}'.format(l))
            ax[1].plot(bias_vals, 1.0E3*r[i,:,c], color=color, label='lms {}'.format(l))
            ax[2].plot(bias_vals, siq[i,:,c], color=color)

        ax[0].legend()
        ax[2].set_xlabel('Bias Voltage [V]')
        ax[0].set_ylabel(r'$S_{IB}$')
        ax[1].set_ylabel(r'$R$' + ' ' + '$[m \Omega]$')
        ax[2].set_ylabel(r'$S_{IQ}$' + ' ' + '$[\mu A/pW]$')
        fig.suptitle('{:03}'.format(c))

        ax[0].set_ylim((-1.2, 1.2))
        ax[1].set_ylim((0,75))

        plt.savefig(os.path.join(savedir, 'S_vs_lms{:03}'.format(c)))
        if not show_plot:
            plt.close()
    



def test(S):    
    #n_bias = len(bias_vals)
    #n_chans = len(S.which_on(2))
    for b in bias_vals:
        print('Bias {}'.format(b))
        S.overbias_tes_all(bias_groups=np.array([2]), overbias_voltage=8, tes_bias=b, 
                           high_current_mode=True, overbias_wait=1.5, cool_wait=90)
        for lms in lms_freqs:
            print(lms)
            S.tracking_setup(2, fraction_full_scale=.72,lms_freq_hz=lms)
            r = S.bias_bump(2, step_size=.03)
            np.save(os.path.join(savedir, 'r_bias{}_lms{}'.format(b, lms)), r)
        #resps = np.zeros((n_bias, n_chans))
        #sibs = np.zeros((n_bias, n_chans))
        #for i, bv in enumerate(bias_vals):
        #    resps[i], sibs[i] = get_sib(S, np.array([1,2,3]), start_bias=bv)
        #np.save(os.path.join(savedir, 'resp_{}'.format(lms)), resps)
        #np.save(os.path.join(savedir, 'sibs_{}'.format(lms)), sibs)
        


def get_sib(S, bias_group, wait_time=1, step_size=.015, duration=10, fs=180,
            start_bias=None, make_plot=False, skip_samp_start=50, skip_samp_end=10):
    bias_group = np.ravel(np.array(bias_group))
    if start_bias is None:
        start_bias = S.get_tes_bias_bipolar(bias_group)

    n_step = int(np.floor(duration / wait_time / 2))

    filename = S.stream_data_on()

    for i in np.arange(n_step):
        for bg in bias_group:
            S.set_tes_bias_bipolar(bg, start_bias + step_size)
        time.sleep(wait_time)
        for bg in bias_group:
            S.set_tes_bias_bipolar(bg, start_bias)
        time.sleep(wait_time)

    S.stream_data_off()

    t, d, m = S.read_stream_data(filename)
    d *= S.pA_per_phi0/(2*np.pi*1.0E6) # Convert to microamps
    i_amp = step_size / S.bias_line_resistance * 1.0E6 # also uA


    n_demod = int(np.floor(fs*wait_time))
    demod = np.tile(np.append(np.ones(n_demod),-np.ones(n_demod)),2)

    bands, channels = np.where(m!=-1)
    resp = np.zeros(len(bands))
    sib = np.zeros(len(bands))*np.nan

    for i, (b, c) in enumerate(zip(bands, channels)):
        mm = m[b, c]
        conv = np.convolve(d[mm], demod)
        start_idx = (len(demod) + np.where(conv == np.max(conv))[0][0])%(2*n_demod)

        high = np.tile(np.append(np.append(np.nan*np.zeros(skip_samp_start), np.ones(n_demod-skip_samp_start-skip_samp_end)),
               np.nan*np.zeros(skip_samp_end+n_demod)),4)
        low = np.tile(np.append(np.append(np.nan*np.zeros(n_demod+skip_samp_start), np.ones(n_demod-skip_samp_start-skip_samp_end)),
              np.nan*np.zeros(skip_samp_end)),4)
        x = np.arange(len(low)) + start_idx

        h = np.nanmean(high*d[mm,start_idx:start_idx+len(high)])
        l = np.nanmean(low*d[mm,start_idx:start_idx+len(low)])

        resp[i] = h-l


        if np.abs(resp[i]) > 1E-2:
            sib[i] = resp[i] / i_amp
            if make_plot:
                plt.figure()
                plt.plot(d[mm])
                plt.axvline(start_idx, color='k', linestyle=':')
                plt.plot(x, h*high)
                plt.plot(x, l*low)
                plt.title(resp[i])
                plt.show()

    return resp, sib



