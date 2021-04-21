import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import csv
import scipy.signal as signal

colors = ["windows blue", "amber", "greyish", "faded green", "dusty purple"]
colors = sns.xkcd_palette(colors)

data_dir = '/mnt/d/Google Drive/Work/uMux/analysis/noise_pole_vs_stanford/1585774029/outputs/psd'
# plot_dir =

datafile_file = '1585891006_datafiles.txt'
bias_file = '1585891006_bias.txt'

datafile = np.genfromtxt(os.path.join(data_dir, datafile_file), dtype='str')
timestamp = [os.path.split(d)[1].split('.dat')[0] for d in datafile]
bias = np.loadtxt(os.path.join(data_dir, bias_file))
pole_datadir = '/mnt/d/Google Drive/Work/uMux/analysis/noise_pole_vs_stanford/pole_normal'
plot_dir = '/mnt/d/Google Drive/Work/uMux/analysis/noise_pole_vs_stanford/plots'

def get_data_stanford(b, ch, f_min=1, f_max=5, make_plot=True, show_plot=False,
    save_plot=True):
    """
    """
    if show_plot:
        plt.ion()
    else:
        plt.ioff()

    normal_idx = np.array([0,1,2])
    sc_idx = np.array([-1, -2])

    if make_plot:
        plt.figure(figsize=(6,4))
    normal_wn = np.zeros(len(normal_idx))
    for i, idx in enumerate(normal_idx):
        f, pxx = np.loadtxt(os.path.join(data_dir,
            f'{timestamp[idx]}_psd_b{b}ch{ch:03}.txt'))

        f_idx = np.where(np.logical_and(f>f_min, f<f_max))
        normal_wn[i] = np.mean(pxx[f_idx])

        if make_plot:
            plt.loglog(f, pxx, alpha=.5, color=colors[1])

    sc_wn = np.zeros(len(sc_idx))
    for i, idx in enumerate(sc_idx):
        f, pxx = np.loadtxt(os.path.join(data_dir,
            f'{timestamp[idx]}_psd_b{b}ch{ch:03}.txt'))

        f_idx = np.where(np.logical_and(f>f_min, f<f_max))
        sc_wn[i] = np.mean(pxx[f_idx])
        if make_plot:
            plt.loglog(f, pxx, alpha=.5, color=colors[0])

    if make_plot:
        plt.axhline(np.min(normal_wn), color=colors[1], linestyle='--',
            label=f'Normal {np.min(normal_wn):3.1f}')
        plt.axhline(np.min(sc_wn), color=colors[0], linestyle='--',
            label=f'SC {np.min(sc_wn):3.1f}')

        plt.legend(loc='upper right')
        print(f'{np.min(normal_wn):3.2f} {np.min(sc_wn):3.2f}')

        plt.title(f'Normal/SC Stanford b{b}ch{ch:03}')

        plt.ylim(20, 250)
        plt.xlabel("Freq [Hz]")
        plt.ylabel("Resp [pA/rtHz]")

        plt.tight_layout()

        if show_plot:
            plt.show()
        else:
            plt.close()

    return np.min(normal_wn), np.min(sc_wn)

def get_data_pole(b, ch, t, d, m, fs=180, f_min=1, f_max=5, make_plot=True,
    show_plot=False, save_plot=True):
    """
    """
    idx = m[b, ch]

    f, pxx = signal.welch(d[idx], fs=fs, nperseg=2**14)
    pxx = np.sqrt(pxx)

    f_idx = np.where(np.logical_and(f>f_min, f<f_max))
    wn = np.mean(pxx[f_idx])

    if make_plot:
        plt.figure(figsize=(4,2.5))
        plt.semilogy(f,pxx, color=colors[0], alpha=.5)
        plt.axhline(wn, color=colors[0], linestyle='--', label=f'{wn:3.2f}')
        plt.legend(loc='upper right')
        plt.tight_layout()

        if save_plot:
            plt.savefig(os.path.join(plot_dir, f'pole_b{b}ch{ch:03}.png'),
                bbox_inches='tight')

        if show_plot:
            plt.show()
        else:
            plt.close()

    return wn

def plot_stanford_vs_pole_normal(f_min, f_max, save_plot=True):
    stanford = np.array([])
    pole = np.array([])

    # Read CSV data with stanford and pole noise
    # with open(os.path.join(pole_datadir,
    #     'stanford_pole_normal_noise.csv')) as csv_file:
    #     csv_reader = csv.reader(csv_file, delimiter=",")
    #     for row in csv_reader:
    #         stanford = np.append(stanford, row[0])
    #         pole = np.append(pole, row[1])

    # s = np.ones(len(stanford)) * -1
    # p = np.ones_like(s) * -1

    # for i in np.arange(len(s)):
    #     if stanford[i] != '':
    #         s[i] = float(stanford[i])
    #     if pole[i] != '':
    #         p[i] = float(pole[i])

    tag = f'_{int(f_min*1.0E3):03}_{int(f_max*1.0E3):03}'

    p = np.loadtxt(os.path.join(pole_datadir, 'normal_noise' + tag))
    s = np.loadtxt(os.path.join(data_dir, 'normal_wn' + tag))
    s = s[:-1]

    # mask bad solutions
    p[p<1] = np.nan
    s[s<1] = np.nan
    p[p>250] = np.nan
    s[s>250] = np.nan

    plt.figure(figsize=(5,4))

    plt.plot([0,200], [0,200], color='k', linestyle=':')
    plt.plot(s, p, '.', color=colors[0])
    plt.xlabel('Stanford noise [pA/rtHz]')
    plt.ylabel('Pole noise [pA/rtHz]')
    plt.title(f'Normal noise : {f_min} - {f_max} Hz')
    plt.grid()

    plt.gca().set_aspect('equal', adjustable='box')

    plt.xlim(0,200)
    plt.ylim(0,200)

    if save_plot:
        plt.savefig(os.path.join(plot_dir, f'noise_stanford_vs_pole{tag}.png'),
            bbox_inches='tight')
        plt.close()

    ratio = p/s
    med = np.nanmedian(ratio)
    idx = ~np.isnan(ratio)
    plt.figure(figsize=(5,4))
    plt.hist(ratio[idx], bins=np.arange(.5, 5, .25))
    plt.axvline(med, color='k', linestyle='--', label=f'Med {med:2.2f}')
    plt.legend(loc='upper right')
    plt.title(f'Noise Pole/Stanford : {f_min} - {f_max} Hz')

    if save_plot:
        plt.savefig(os.path.join(plot_dir,
            f'hist_noise_stanford_vs_pole{tag}.png'),
            bbox_inches='tight')
        plt.close()



if __name__ == "__main__":
    pole = True
    stanford = True
    f_min = 5
    f_max = 10

    tag = f'_{int(f_min*1.0E3):03}_{int(f_max*1.0E3):03}'

    if pole:
        band = np.array([])
        channel = np.array([])
        with open(os.path.join(data_dir, 'channel_list.csv')) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            for row in csv_reader:
                band = np.append(band, row[0])
                channel = np.append(channel, row[1])

        band = band.astype(int)
        channel = channel.astype(int)

        normal_wn = np.zeros(len(band))
        sc_wn = np.zeros(len(band))

        for i, (b, ch) in enumerate(zip(band, channel)):
            print(b, ch)
            normal_wn[i], sc_wn[i] = get_data_stanford(b, ch, make_plot=False,
                f_min=f_min, f_max=f_max)

        np.savetxt(os.path.join(data_dir, 'normal_wn' + tag), normal_wn)
        np.savetxt(os.path.join(data_dir, 'sc_wn' + tag), sc_wn)

    if stanford:
        import pysmurf.client
        S = pysmurf.client.SmurfControl(offline=True)

        t, d, m = S.read_stream_data(os.path.join(pole_datadir,
            '1563581403.dat.part_00001'), n_samp=40000, gcp_mode=True)

        # Convert to pA
        pA_per_phi0 = 9E6
        d *= pA_per_phi0/(2*np.pi)

        band = np.array([])
        channel = np.array([])
        with open(os.path.join(pole_datadir, 'pole_channel.csv')) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            for row in csv_reader:
                band = np.append(band, row[0])
                channel = np.append(channel, row[1])

        wn = np.ones(len(band)) * -1
        for i, (b, ch) in enumerate(zip(band, channel)):
            if b != '':
                b = int(b)
                ch = int(ch)
                print(b, ch)
                wn[i] = get_data_pole(b, ch, t, d, m, f_min=f_min, f_max=f_max,
                    make_plot=False)

        np.savetxt(os.path.join(pole_datadir, 'normal_noise' + tag), wn)

    plot_stanford_vs_pole_normal(f_min=f_min, f_max=f_max)