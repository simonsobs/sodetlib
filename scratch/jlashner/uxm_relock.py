import time
import numpy as np
import sodetlib as sdl

import typing
if typing.TYPE_CHECKING:
    from pysmurf.client.base.smurf_control import SmurfControl
    from sodetlib.det_config import DetConfig


def publish_ocs_log(S, msg):
    """
    Passes a string to the OCS pysmurf controller to be logged to be passed
    around the OCS network
    """
    S.pub.publish(msg, msgtype='session_log')


def publish_ocs_data(S, data):
    S.pub.publish(data, msgtype='session_data')
    

def reload_amps(S: SmurfControl, cfg: DetConfig):
    summary = {}

    log_to_controller(S, 'setting amp voltage')
    S.set_50k_amp_gate_voltage(cfg.dev.exp['amp_50k_Vg'])
    S.set_hemt_gate_voltage(cfg.dev.exp['amp_hemt_Vg'])
    S.C.write_ps_en(3)
    time.sleep(0.1)
    biases = S.get_amplifier_biases()
    summary['biases'] = biases

    in_range_50k = np.abs(biases['50k_Id'] - exp['amp_50k_Id']) < id_tolerance
    in_range_hemt = np.abs(biases['hemt_Id'] - exp['amp_hemt_Id']) < id_tolerance

    if not (in_range_50k and in_range_hemt):
        S.log("Hemt or 50K Amp drain current not within tolerance")
        S.log(f"Target hemt Id: {exp['amp_hemt_Id']}")
        S.log(f"Target 50K Id: {exp['amp_50k_Id']}")
        S.log(f"tolerance: {id_tolerance}")
        S.log(f"biases: {biases}")

        summary['success'] = False
        publish_ocs_log(S, 'Failed to set amp voltages')
        publish_ocs_data(S, {'amp_summary': summary})
        return False, summary


    summary['success'] = True 
    publish_ocs_log(S, 'Succuessfully to set amp voltages')
    publish_ocs_data(S, {'amp_summary': summary})
    return True, summary


def reload_tune(S: SmurfControl, cfg: DetConfig, bands,
                new_master_assignment=False):
    pass


def relock_tracking_setup(S: SmurfControl, cfg: DetConfig, bands,
                          reset_rate_khz=None, nphi0=None, **kwargs):
    """
    Sets up tracking for smurf. This assumes you already have optimized
    lms_freq and frac-pp for each bands in the device config. This function
    will chose the flux-ramp fraction-full-scale by averaging the optimized
    fractions across the bands you're running on.

    This function also allows you to set reset_rate_khz and nphi0. The
    fraction-full-scale, and lms frequencies of each band will be automatically
    adjusted based on their pre-existing optimized values.
    """

    bands = np.atleast_1d(bands)
    nbands = len(bands)

    # Arrays containing the optimized tracking parameters for each band
    frac_pp0 = np.zeros(nbands)
    lms_freq0 = np.zeros(nbands)  # Hz
    reset_rate_khz0 = None  # This is assumed to be the same for each band

    for i, b in enumerate(bands):
        bcfg = cfg.dev.bands[b]
        frac_pp0[i] = bcfg['frac_pp']
        lms_freq0[i] = bcfg['lms_freq_hz'] 
        if reset_rate_khz0 is None:
            reset_rate_khz0 = bcfg['flux_ramp_rate_khz']

    # Nphi0 used during optimization (assumes this is the same for each band)
    init_nphi0 = np.round(lms_freq0[0] / reset_rate_khz0)

    # Choose frac_pp to be the mean of all running bands.
    # This is the frac-pp at the flux-ramp-rate used for optimization
    fpp0 = np.mean(frac_pp0)

    # Adjust fpp, lms_freq, and flux-ramp-rate depending on desired
    # flux-ramp-rate and nphi0
    fpp, lms_freqs = fpp0, lms_freq0
    if nphi0 is not None:
        fpp *= nphi0 / init_nphi0
        lms_freqs *= fpp / fpp0
    if reset_rate_khz is not None:
        lms_freqs *= reset_rate_khz / reset_rate_khz0
    else:
        reset_rate_khz = reset_rate_khz0

    # Runs tracking setup
    res = {}
    tk = sdl.get_tracking_kwargs(S, cfg, bands[0], kwargs=kwargs)
    tk['reset_rate_khz'] = reset_rate_khz
    tk['fraction_full_scale'] = fpp
    for i, b in enumerate(bands):
        tk.update({'lms_freq_hz': lms_freqs[i]})
        res[b] = S.tracking_setup(b, **tk)

    return res









        
        





def uxm_relock(S: SmurfControl, cfg: DetConfig, bands=None, id_tolerance=0.5,
               new_master_assignment=False):
    """
    Relock steps:

        1. Reset state (all off, disable waveform, etc.)
        2. Set amps and check tolerance 
        3. load tune
        4. setup-notches if specified (new_master_assignment=False)
        5. Serial gradient descent and eta scan 
        6. Run tracking setup
        7. Measure noise
    """
    if bands is None:
        bands = np.arange(8)
    bands = np.atleast_1d(bands)

    summary = {}

    # 1. Reset system to known state
    S.all_off()  # Turn off Flux ramp, tones, and biases
    S.set_rtm_arb_waveform_enable(0)
    S.set_filter_disable(0)
    S.set_downsample_factor(20)
    S.set_mode_dc()
    for band in bands:
        band_cfg = cfg.dev.bands[band]
        S.set_att_uc(band, band_cfg['uc_att'])
        S.set_att_dc(band, band_cfg['dc_att'])
        S.amplitude_scale[band] = band_cfg['tone_power']


    # 2. set amplifiers
    reload_amps(S, cfg)

    # 3. load tune
    reload_tune(S, cfg, bands, new_master_assignment=new_master_assignment)

    # 4. setup-notches if specified (new_master_assignment=False)
    for band in bands:
        S.setup_notches(band, new_master_assignment=new_master_assignment)

    # 5. Serial gradient descent and eta scan 
    for band in bands:
        S.run_serial_gradient_descent(band)
        S.run_serial_eta_scan(band)

    # 6. Run tracking setup
    for band in bands:
        tk = sdl.get_tracking_kwargs(S, cfg)
        S.tracking_setup(band)

    # 7. Measure noise



