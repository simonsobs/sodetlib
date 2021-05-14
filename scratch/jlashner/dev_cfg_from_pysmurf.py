from sodetlib import det_config
import numpy as np


def dev_cfg_from_pysmurf(S, save_file=None, clobber=True):
    dev = det_config.DeviceConfig()

    # Experiment setup:
    # Experiment updates:
    amp_biases = S.get_amplifier_biases()
    if hasattr(S, 'tune_file'):
        tunefile = S.tune_file
    else:
        tunefile = None
    dev.exp.update({
        'amp_50k_Id': amp_biases['50K_Id'],
        'amp_50k_Vg': amp_biases['50K_Vg'],
        'amp_hemt_Id': amp_biases['hemt_Id'],
        'amp_hemt_Vg': amp_biases['hemt_Vg'],
        'tunefile': tunefile
    })

    # Right now not getting any bias group info
    for band in S._bands:
        tone_powers = S.get_amplitude_scale_array(band)
        drive = np.median(tone_powers[tone_powers != 0])

        feedback_start_frac = S._feedback_to_feedback_frac(band, S.get_feedback_start(band))
        feedback_end_frac = S._feedback_to_feedback_frac(band, S.get_feedback_end(band))

        flux_ramp_rate_khz = S.get_flux_ramp_freq()
        lms_freq_hz = S.get_lms_freq_hz(band)
        nphi0 = np.round(lms_freq_hz / flux_ramp_rate_khz / 1e3)

        dev.bands[band].update({
            'uc_att': S.get_att_uc(band),
            'dc_att': S.get_att_dc(band),
            'drive': drive,
            'feedback_start_frac': feedback_start_frac,
            'feedback_end_frac': feedback_end_frac,
            'lms_gain': S.get_lms_gain(band),
            'frac_pp': S.get_fraction_full_scale(),
            'flux_ramp_rate_khz': flux_ramp_rate_khz,
            'lms_freq_hz': lms_freq_hz,
            'nphi0': nphi0
        })

    if save_file is not None:
        dev.dump(save_file, clobber=True)
    return dev


if __name__ == '__main__':
    cfg = det_config.DetConfig()
    cfg.parse_args()
    S = cfg.get_smurf_control()
    S.load_tune()
    dev = dev_cfg_from_pysmurf(S, save_file='test.yaml')


