Bias Step Rebias
================================

For various reasons, it may be beneficial to bias or rebias detectors into their
transition using bias-step functions instead of IVs, since
- bias steps can be much faster than IVs
- bias steps don't require overbiasing

Yuhan Wang has developed the bias-step rebiasing function to find a suitable
bias-point using just bias steps, which is described in greater detail in `his
spie proceeding <https://arxiv.org/abs/2107.13504>`_.

The general control flow is as follows:

    1. Take initial bias steps bsa_0
        will rebias if there are any bias lines where R0 is all nan
        If >10% of dets are have Rfrac < 0.1, marks bg as overbias_needed
        If >50% of dets have Rfrac > 0.9, marks bg as normal (and drop from normal needed)
    2. Overbias overbias_needed bgs
        runs overbias_dets on bgs that need overbiasing, and waits
        For biaslines that have been overbiased, set the DC bias voltage to testbed_100mK_bias_voltage
        Run bias steps
            (no bias retry here)
        Check normal condition (above) and add updated bias lines to bg_detectors_normal array if they meet it
    3. Drop from normal
        For each normal bg, step voltage by 0.5(V_norm - V_SC) determined from IV
        Run bias steps
            update bsa_0 so it's always the last bias step ran
            These bias steps don't have a retry
        Check normal condition. If any bgs are still normal, repeat.
        After exiting loop, the resulting DC biases are initial_dc_biases
    4. Find new bjas voltage and take bias steps:
        vspread is diff between median normal and median SC voltages based on IV
        New bias voltage is init +/- 0.15*vspread where +/- is determined by
        whether or not the median Rfrac is > or < than 0.5
        Take bias steps (bsa_1) and retry if needed
    5. Determine new bias point for each channel:
        v_estimate = (v0 (target - Rfrac_1) + v1 (Rfrac_0 - target)) / (Rfrac_0 - Rfrac_1)
        vnew = median(v_estimate[bg]), or if all are nan, vnew = (v1 - v0) / 2
    6. Set bias to vnew and retake bias steps bsa_2 (with retry if needed)
        BG is successful if abs(Rfrac - target) < 0.05
    7. Fine tuning unsuccessful bias groups
        Step 0.15 vspread in the appropriate direction
        Take bias steps (with retry)
        v_estimate2 = (v0 (target - Rfrac_1) + v1 (Rfrac_0 - target)) / (Rfrac_0 - Rfrac_1)
            with v0= prev estimate and v1 = prev estimate + delta
        Apply v_estimate2, retake bias steps.
    8. Return final bias step and bias voltages
