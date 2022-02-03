import numpy as np
import time
import typing
import sodetlib as sdl


from sodetlib.util import get_tracking_kwargs

from pysmurf.client.base.smurf_control import SmurfControl
from sodetlib.det_config import DetConfig


def find_gate_voltage(S, target_Id, vg_min, vg_max, amp_name, max_iter=50):
    """
    Scans through bias voltage for hemt or 50K amplifier to get the correct
    gate voltage for a target current.

    Parameters
    -----------
    S: pysmurf.client.SmurfControl
        PySmurf control object
    target_Id: float
        Target amplifier current
    vg_min: float
        Minimum allowable gate voltage
    vg_max: float
        Maximum allowable gate voltage
    amp_name: str
        Name of amplifier. Must be either "hemt" or "50K'.
    max_iter: int, optional
        Maximum number of iterations to find voltage. Defaults to 30.

    Returns
    --------
    success (bool):
        Returns a boolean signaling whether voltage scan has been successful.
        The set voltages can be read with S.get_amplifier_biases().
    """
    if amp_name not in ['hemt', '50K']:
        raise ValueError("amp_name must be either 'hemt' or '50K'")

    for _ in range(max_iter):
        amp_biases = S.get_amplifier_biases(write_log=True)
        Vg = amp_biases[f"{amp_name}_Vg"]
        Id = amp_biases[f"{amp_name}_Id"]
        delta = target_Id - Id

        # Id should be within 0.5 from target without going over.
        if 0 <= delta < 0.5:
            return True

        if amp_name == 'hemt':
            step = np.sign(delta) * (0.1 if np.abs(delta) > 1.5 else 0.01)
        else:
            step = np.sign(delta) * (0.01 if np.abs(delta) > 1.5 else 0.001)

        Vg_next = Vg + step
        if not (vg_min < Vg_next < vg_max):
            S.log(f"Vg adjustment would go out of range ({vg_min}, {vg_max}). "
                  f"Unable to change {amp_name}_Id to desired value", False)
            return False

        if amp_name == 'hemt':
            S.set_hemt_gate_voltage(Vg_next, override=True)
        else:
            S.set_50k_amp_gate_voltage(Vg_next, override=True)
        time.sleep(0.2)
    S.log(f"Max allowed Vg iterations ({max_iter}) has been reached. "
          f"Unable to get target Id for {amp_name}.", False)
    return False


def setup_amps(S: SmurfControl, cfg: DetConfig, id_hemt=8.0, id_50k=15.0,
               vgmin_hemt=-1, vgmin_50k=-1):
    """

    Initial setup for 50k and hemt amplifiers. Determines gate voltages
    required to reach specified drain currents. Will update detconfig on
    success.

    Args
    -----
        S : (SmurfControl)
            Pysmurf instance
        cfg : (DetConfig)
            DetConfig instance
        id_hemt : (float)
            Target hemt drain current (mA)
        id_50k : (float)
            Target 50k drain current (mA)
        vgmin_hemt : (float)
            Min hemt gate voltage (V)
        vgmin_50k : (float)
            Min 50k gate voltage (V)
    """
    S.pub.publish({'current_operation': 'setup_amps'}, msgtype='session_data')

    S.set_50k_amp_gate_voltage(-0.8)
    S.set_hemt_gate_voltage(-0.8)
    S.C.write_ps_en(3)
    time.sleep(0.3)

    # Data to be passed back to the ocs-pysmurf-controller clients
    summary = {
        'success':     False,
        'amp_50k_Id':  None,
        'amp_hemt_Id': None,
        'amp_50k_Vg':  None,
        'amp_hemt_Vg': None,
    }

    if not find_gate_voltage(S, id_hemt, vgmin_hemt, 0, 'hemt'):
        S.log("Failed determining hemt gate voltage")
        S.pub.publish({'setup_amps_summary': summary}, msgtype='session_data')
        S.C.write_ps_en(0)
        return False, summary

    if not find_gate_voltage(S, id_50k, vgmin_50k, 0, '50K'):
        S.log("Failed determining 50k gate voltage")
        S.pub.publish({'setup_amps_summary': summary}, msgtype='session_data')
        S.C.write_ps_en(0)
        return False, summary

    # Update device cfg
    biases = S.get_amplifier_biases()
    cfg.dev.update_experiment({
        'amp_50k_Id':  id_50k,
        'amp_hemt_Id': id_hemt,
        'amp_50k_Vg':  biases['50K_Vg'],
        'amp_hemt_Vg': biases['hemt_Vg'],
    }, update_file=True)

    summary = {'success': True, **biases}
    S.pub.publish({'setup_amps_summary': summary}, msgtype='session_data')
    return True, summary


def setup_phase_delay(S: SmurfControl, cfg: DetConfig, bands, uc_att=20,
                      dc_att=20):
    """
    Sets uc and dc attens to reasonable values and runs estimate phase delay
    for desired bands.

    Args
    -----
        S : (SmurfControl)
            Pysmurf instance
        cfg : (DetConfig)
            DetConfig instance
        uc_att : (int)
            UC atten to use for phase-delay estimation
        dc_att : (int)
            DC atten to use for phase-delay estimation
    """
    S.pub.publish({'current_operation': 'setup_phase_delay'},
                  msgtype='session_data')

    summary = {
        'bands': [],
        'ref_phase_delay_fine': [],
        'ref_phase_delay': [],
    }
    for b in bands:
        S.set_att_dc(b, dc_att)
        S.set_att_uc(b, uc_att)
        S.estimate_phase_delay(b, make_plot=True, show_plot=False)
        rfd = float(S.get_ref_phase_delay(b))
        rfdf = float(S.get_ref_phase_delay_fine(b))
        summary['bands'].append(int(b))
        summary['ref_phase_delay'].append(rfd)
        summary['ref_phase_delay_fine'].append(rfdf)
        cfg.dev.bands[b].update({
            'ref_phase_delay':      rfd,
            'ref_phase_delay_fine': rfdf,
        })

    cfg.dev.dump(cfg.dev_file, clobber=True)
    S.pub.publish({'setup_phase_delay': summary}, msgtype='session_data')
    return True, summary


def estimate_uc_dc_atten(S, band, tone_power=None, ref_freq=-200,
                         num_subbands=5, resp_range=(0.1, 0.3)):
    att = 15
    step = 7

    if tone_power is None:
        tone_power = S._amplitude_scale[band]

    sb = S.get_closest_subband(ref_freq, band)
    sbs = np.arange(sb, sb + num_subbands)

    nread = 2

    while True:
        S.set_att_uc(band, att)
        S.set_att_dc(band, att)

        _, resp = S.full_band_ampl_sweep(band, sbs, tone_power, nread)
        max_resp = np.max(np.abs(resp))
        S.log(f"att: {att}, max_resp: {max_resp}")

        if resp_range[0] < max_resp < resp_range[1]:
            S.log(f"Estimated atten: {att}")
            return att

        if step == 0:
            S.log(f"Cannot achieve resp in range {resp_range}!")
            return att

        if max_resp < resp_range[0]:
            att -= step
            step = step // 2
        elif max_resp > resp_range[1]:
            att += step
            step = step // 2


def setup_tune(S: SmurfControl, cfg: DetConfig, bands, tone_power=None,
               show_plot=False, amp_cut=0.01, grad_cut=0.01,
               estimate_attens=True):
    """
    Find freq, setup notches, and serial gradient descent and eta scan

    Args
    -----
        S : (SmurfControl)
            Pysmurf instance
        cfg : (DetConfig)
            DetConfig instance
        id_hemt : (float)
            Target hemt drain current (mA)
        id_50k : (float)
            Target 50k drain current (mA)
        vgmin_hemt : (float)
            Min hemt gate voltage (V)
        vgmin_50k : (float)
            Min 50k gate voltage (V)
    """
    bands = np.atleast_1d(bands)

    if tone_power is None:
        # Lets just assume all tone-powers are the same for now
        tone_power = S._amplitude_scale[bands[0]]

    # Temporary : 
    summary = {}

    for band in bands:
        # First try to choose attens that are good for tuning
        if estimate_attens:
            att = estimate_uc_dc_atten(S, band, tone_power=tone_power)
            cfg.dev.update_band(band, {
                'uc_att': att,
                'dc_att': att,
            })
        S.pub.publish({'current_operation': f'find_freq:band{band}'},
                      msgtype='session_data')
        S.find_freq(band, tone_power=tone_power, make_plot=True,
                    save_plot=True, show_plot=show_plot, amp_cut=amp_cut,
                    grad_cut=grad_cut)
        # Probably want to check for number of resonances here and send to ocs

    for band in bands:
        S.pub.publish({'current_operation': f'setup_notches:band{band}'},
                      msgtype='session_data')
        S.setup_notches(band, tone_power=tone_power, new_master_assignment=True)
        # Probably want to check for number of resonances here and send to ocs

    for band in bands:
        S.pub.publish({'current_operation': f'serial_ops:band{band}'},
                      msgtype='session_data')
        S.run_serial_gradient_descent(band)
        S.run_serial_eta_scan(band)

    cfg.dev.update_experiment({'tunefile': S.tune_file}, update_file=True)

    return True, summary


def compute_tracking_quality(S, f, df, sync):
    """
    Computes the tracking quality parameter from tracking_setup results.

    Args
    ------
        S : SmurfControl
            Pysmurf instance
        f : np.ndarray
            Array of the tracked frequency for each channel, as returned by
            tracking_setup
        df : np.ndarray
            Array of the tracked frequency error for each channel, as returned
            by tracking_setup
        sync : np.ndarray
            Array containing tracking sync flags, as returned by tracking_setup
    """
    sync_idxs = S.make_sync_flag(sync)
    seg_size = np.min(np.diff(sync_idxs))
    nstacks = len(sync_idxs) - 1

    fstack = np.zeros((seg_size, len(f[0])))
    for i in range(nstacks):
        fstack += f[sync_idxs[i]:sync_idxs[i]+seg_size] / nstacks


    # calculates quality of estimate wrt real data
    y_real = f[sync_idxs[0]:sync_idxs[-1], :]
    y_est = np.vstack([fstack for _ in range(nstacks)])
    # Force these to be the same len in case all segments are not the same size
    y_est = y_est[:len(y_real)]

    with np.errstate(invalid='ignore'):
        sstot = np.sum((y_real - np.mean(y_real, axis=0))**2, axis=0)
        ssres = np.sum((y_real - y_est)**2, axis=0)
        r2 = 1 - ssres/sstot

    return r2


def setup_tracking_params(S: SmurfControl, cfg: DetConfig, bands,
                          init_fracpp=0.44, nphi0=5, reset_rate_khz=4,
                          lms_gain=0, feedback_gain=2048, show_plots=False):
    """
    Setups up tracking parameters by determining correct frac-pp and lms-freq
    for each band.

    Args
    -----
        S : (SmurfControl)
            Pysmurf instance
        cfg : (DetConfig)
            DetConfig instance
        bands : (np.ndarray, int)
            Band or list of bands to run on
        init_fracpp : (float, optional)
            Initial frac-pp value to use for tracking estimates
        nphi0 : int
            Number of phi0 periods to track on
        reset_rate_khz : float
            Flux ramp reset rate in khz
    """

    bands = np.atleast_1d(bands)

    summary = {
        'success': None,
        'num_good_chans': [None for _ in range(8)]
    }
    for band in bands:
        S.pub.publish(
            {'current_operation': f'setup_tracking_params:band{band}'},
            msgtype='session_data'
        )
        tk = get_tracking_kwargs(
            S, cfg, band, kwargs={
                'lms_freq_hz':         None,
                'show_plot':           False,
                'meas_lms_freq':       True,
                'fraction_full_scale': init_fracpp,
                'reset_rate_khz':      reset_rate_khz,
                'lms_gain': lms_gain,
                'feedback_gain': feedback_gain,
            }
        )
        f, df, sync = S.tracking_setup(band, **tk)
        r2 = compute_tracking_quality(S, f, df, sync)

        # Cut all but good chans to calc fpp / lms-freq
        asa_init = S.get_amplitude_scale_array(band)
        asa_good = asa_init.copy()
        asa_good[r2 < 0.95] = 0
        S.set_amplitude_scale_array(band, asa_good)

        # Calculate trracking parameters
        S.tracking_setup(band, **tk)
        lms_meas = S.lms_freq_hz[band]
        lms_freq = nphi0 * tk['reset_rate_khz'] * 1e3
        frac_pp = tk['fraction_full_scale'] * lms_freq / lms_meas

        # Re-enables all channels and re-run tracking setup with correct params
        S.set_amplitude_scale_array(band, asa_init)
        tk['meas_lms_freq'] = False
        tk['fraction_full_scale'] = frac_pp
        tk['lms_freq_hz'] = lms_freq
        tk['show_plot'] = show_plots
        f, df, sync = S.tracking_setup(band, **tk)
        r2 = compute_tracking_quality(S, f, df, sync)

        ## Lets add some cuts on p2p(f) and p2p(df) here. What are good numbers
        ## for that?
        num_good_chans = np.sum(r2 > 0.95)
        summary['num_good_chans'][band] = int(num_good_chans)
        S.pub.publish(
            {'setup_tracking_params_summary': summary},
            msgtype='session_data'
        )

        # Update det config
        cfg.dev.update_band(band, {
            'frac_pp':            frac_pp,
            'lms_freq_hz':        lms_freq,
            'flux_ramp_rate_khz': reset_rate_khz,
            'lms_gain': lms_gain,
            'feedback_gain': feedback_gain,


        }, update_file=True)

    return True, summary


def uxm_setup(S: SmurfControl, cfg: DetConfig, bands=None, id_hemt=8.0,
              id_50k=15.0, phase_delay_uc=20, phase_delay_dc=20,
              tone_power=None, amp_cut=0.01, grad_cut=0.01, init_fracpp=0.44,
              nphi0=5, reset_rate_khz=4, lms_gain=0, feedback_gain=2048,
              show_plots=False):
    """
    The goal of this function is to do a pysmurf setup completely from scratch,
    meaning no parameters will be pulled from the device cfg.

    The following steps will be run:

        1. setup amps
        2. Estimate phase delay
        3. Setup tune
        4. setup tracking
        5. Measure noise
    """
    if bands is None:
        bands = np.arange(8)
    bands = np.atleast_1d(bands)

    S.all_off()  # Turn off Flux ramp, tones, and biases
    S.set_rtm_arb_waveform_enable(0)
    S.set_filter_disable(0)
    S.set_downsample_factor(20)
    S.set_mode_dc()

    # 1. setup amps
    summary = {'timestamps': []}
    summary['timestamps'].append(('setup_amps', time.time()))
    success, summary['setup_amps'] = setup_amps(
        S, cfg, id_hemt=id_hemt, id_50k=id_50k
    )
    if not success:
        S.log("UXM Setup failed on setup amps step")
        return False, summary

    # 2. Estimate phase delay
    summary['timestamps'].append(('setup_phase_delay', time.time()))
    success, summary['setup_phase_delay'] = setup_phase_delay(
        S, cfg, bands, uc_att=phase_delay_uc, dc_att=phase_delay_dc
    )
    if not success:
        S.log("UXM Setup failed on setup phase delay step")
        return False, summary

    # 3. Find Freq
    summary['timestamps'].append(('setup_tune', time.time()))
    success, summary['setup_tune'] = setup_tune(
        S, cfg, bands, tone_power=tone_power, show_plot=show_plots,
        amp_cut=amp_cut, grad_cut=grad_cut
    )
    if not success:
        S.log("UXM Setup failed on setup tune step")
        return False, summary

    # 4. tracking setup
    summary['timestamps'].append(('setup_tracking_params', time.time()))
    success, summary['setup_tracking_params'] = setup_tracking_params(
        S, cfg, bands, init_fracpp=init_fracpp, nphi0=nphi0,
        reset_rate_khz=reset_rate_khz, lms_gain=lms_gain,
        feedback_gain=feedback_gain, show_plots=show_plots
    )
    if not success:
        S.log("UXM Setup failed on setup tracking step")
        return False, summary

    # 5. Noise Measurement
    summary['timestamps'].append(('noise', time.time()))
    am, summary['noise'] = sdl.noise.take_noise(S, cfg, 30,
                                                show_plot=show_plots,
                                                save_plot=True)
    S.pub.publish({'noise_summary': {
        'band_medians': summary['noise']['noisedict']['band_medians'].tolist()
    }}, msgtype='session_data')

    summary['timestamps'].append(('end', time.time()))

    return True, summary
