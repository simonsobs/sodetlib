import numpy as np

def play_bias_steps_dc(S, cfg, duration, num_steps=5, step_voltage=0.05,
                       bias_groups=None, do_enable=True):
    """
    Plays bias steps on a group of bias groups stepping with only one DAC
    """
    if bias_groups is None:
        bias_groups = np.arange(12)
    bias_groups = np.atleast_1d(bias_groups)

    dac_volt_array_low = S.get_rtm_slow_dac_volt_array()
    dac_volt_array_high = dac_volt_array_low.copy()

    dac_enable_array = S.get_rtm_slow_dac_enable_array()

    bias_order, dac_positives, dac_negatives = S.bias_group_to_pair.T

    for bg in bias_groups:
        bg_idx = np.ravel(np.where(bias_order == bg))
        dac_positive = dac_positives[bg_idx][0] - 1
        dac_volt_array_high[dac_positive] += step_voltage

    for _ in range(num_steps):
        S.set_rtm_slow_dac_volt_array(dac_volt_array_high)
        time.sleep(duration)
        S.set_rtm_slow_dac_volt_array(dac_volt_array_low)
        time.sleep(duration)

    return

            self.set_rtm_slow_dac_volt_array(dac_volt_array, **kwargs)

