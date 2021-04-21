from pathlib import Path
import numpy as np
import pysmurf.client
import scipy.signal as signal
import os
import matplotlib.pyplot as plt

rootdir = "/mnt/d/Google Drive/Work/uMux/20200219"
min_chan = 400
nperseg = 2**13
pA_per_phi0 = 9.0E6
fs = 200

# Find number of channels in each data file
def get_data_dirs():
    datadirs = {}
    for path in Path(rootdir).rglob("*_mask.txt"):
        pp = path.absolute().as_posix()
        datadirs[pp] = len(np.loadtxt(pp))
    return datadirs


def calculate_psd(datadirs):

    S = pysmurf.client.SmurfControl(offline=True)

    keys = np.sort(list(datadirs.keys()))
    for k in keys:
        if datadirs[k] > min_chan:
            print(datadirs[k], k)
        t, d, m = S.read_stream_data(k.replace("_mask.txt", ".dat"))
        d *= pA_per_phi0/(2*np.pi)

        psd_dict = {}  # Holds all the PSDs
        band, channel = np.where(m != -1)

        for b in np.unique(band):
            psd_dict[b] = {}

        # Calculate PSDs
        for i, (b, ch) in enumerate(zip(band, channel)):
            idx = m[b, ch]
            f, pxx = signal.welch(d[idx], fs=fs, nperseg=nperseg)
            psd_dict[b][ch] = pxx

        np.save(os.path.join(k.replace("_mask.txt", "_psd")),
            psd_dict)
        np.save(os.path.join(k.replace("_mask.txt", "_f")), f)


def plot_data(datafile, make_timestream_plot=False, f_min=1, f_max=2,
    bins=np.arange(0, 500, 10)):
    """
    """
    s = os.path.split(datafile)
    plot_dir = s[0].replace('outputs', 'plots/psd')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    timestamp = s[1].split('_')[0]

    psd = np.load(datafile, allow_pickle=True).item()
    f = np.load(datafile.replace("psd.npy", "f.npy"), allow_pickle=True)

    f_idx = np.where(np.logical_and(f > f_min, f < f_max))

    band = np.array(list(psd.keys()))
    wn = {}

    fig, ax = plt.subplots(2, 2, figsize=(5.75, 4.25), sharex=True)
    for b in band:
        # Extract channels
        channel = np.array(list(psd[b].keys()))

        if make_timestream_plot:
            for ch in channel:
                fig2 = plt.figure()
                plt.semilogy(f, np.sqrt(psd[b][ch]))
                plt.savefig(os.path.join(plot_dir,
                    f'{timestamp}_b{b}ch{ch:03}.png'),
                    bbox_inches='tight')
                plt.close(fig2)

        wn[b] = np.zeros(len(channel))
        for i, ch in enumerate(channel):
            wn[b][i] = np.mean(np.sqrt(psd[b][ch][f_idx]))

        med = np.median(wn[b])

        x = b % 2
        y = b // 2
        ax[y,x].hist(wn[b], bins=bins)
        ax[y,x].text(.02, .975, f'Band {b}', transform=ax[y,x].transAxes,
            va='top', ha='left')
        ax[y,x].text(.98, .975, r'$n_{chan}$ : ' +
            f'{np.sum(wn[b]<np.max(bins))}/{len(wn[b])}' + '\n' +
            f'med : {med:3.1f}',
            transform=ax[y,x].transAxes,
            va='top', ha='right')


        ax[y,x].axvline(med, color='k', linestyle=':')

        if y == 1:
            ax[y,x].set_xlabel('Noise [pA/rtHz]')

    fig.suptitle(timestamp)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_name = os.path.join(plot_dir, f'{timestamp}_hist.png')
    plt.savefig(plot_name, bbox_inches='tight')
    plt.close(fig)

    # Make summary plot
    wn_all = np.array([])

    for b in band:
        wn_all = np.append(wn_all, wn[b])

    med = np.nanmedian(wn_all)
    print(med)

    fig, ax = plt.subplots(1, figsize=(4,2.5))
    ax.hist(wn_all, bins=bins)
    ax.set_xlabel('Noise [pA/rtHz]')
    ax.axvline(med, color='k', linestyle='--')
    ax.set_title(f'White Noise : {f_min:1.1f} - {f_max:1.1f} Hz')
    ax.text(.98, .975, r'$n_{chan}$ : ' +
            f'{np.sum(wn_all<np.max(bins))}/{len(wn_all)}' + '\n'
            f'med : {med:3.1f}',
            transform=ax.transAxes,
            va='top', ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'{timestamp}_hist_all.png'))
    plt.close()

    print(f"Plot saved to : {plot_name}")


if __name__ == "__main__":
    recalc_psd = False

    # Get all the data directories
    datadirs = get_data_dirs()

    if recalc_psd:
        calculate_psd(datadirs)

    keys = np.sort(list(datadirs.keys()))
    for k in keys:
        if datadirs[k] > min_chan:
            plot_data(k.replace('_mask.txt', "_psd.npy"))
