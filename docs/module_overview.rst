Package Overview
=================

Right now just a list of functions, in the future an explanation of how the
module is organized and what you can expect to be where.

sodetlib
---------

* analysis/
    * det_analysis.py
        * CHANS_PER_BAND -- param, duplicate 2x
        * sine
        * fit_sine
        * analyze_biasgroup_data
        * predict_bg_index
        * plot_tickle_summary
        * analyze_tickle_data -- Action
        * load_from_dat
        * load_from_g3
        * load_from_sid
        * analyze_iv_info
        * analyze_iv_and_save -- Action
        * iv_channel_plots
        * iv_summary_plots
        * make_bias_group_map -- Action
        * bias_points_from_rfrac -- Action
    * squid_fit.py
        * plot_squid_curves
        * serial_corr
        * autocorr
        * estimate_fit_parameters
        * model
* smurf_funcs/
    * check_state.py
        * CHANS_PER_BAND -- param, duplicate 2x
        * NBANDS -- param
        * NCHANS -- param
        * ChannelState -- Class
            * set_channel_wls
            * save
            * from_file
            * desc
        * compare_state_with_base
        * get_base_state
        * set_base_state -- Action
        * get_channel_state -- Action
        * plot_state_noise
    * det_ops.py
        * get_current_mode
        * take_iv -- Action
        * take_tickle -- Action
        * bias_detectors_from_sc -- Action
    * optimize_params.py
        * optimize_tacking
        * lowpass_fit
        * identify_best_chan
        * analyze_noise_psd -- Action
        * optimize_bias -- Action
        * plot_optimize_attens -- Action
        * optimize_attens -- Action
    * smurf_ops.py
        * take_squid_open_loop -- Action
        * find_and_tune_freq 
        * tracking_quality -- Action
        * get_session_files
        * load_session -- almost duplicate of det_analysis.load_from_sid
        * take_g3_data -- Action
        * stream_g3_on -- Action
        * stream_g3_off -- Action
        * apply_dev_cfg -- Action
        * cryo_amp_check -- Action
        * get_wls_from_am -- duplicate
        * plot_band_noise 
        * loopband_test -- Action
        * plot_loopback_results
    * bias_steps.py
        * play_bias_steps_dc
        * BiasStepAnalysis -- Class
            * save
            * load
            * run_analysis
            * _load_am
            * _find_bias_edges
            * _create_bg_map
            * _get_step_response
            * _compute_R0_I0_Pj
            * _compute_dc_params
            * _fit_tau_effs
            * plot_step_fit
        * take_bias_steps -- Action 
* det_config.py
    * YamlReps -- Class
    * DeviceConfig -- Class
        * from_dict
        * from_yaml
        * dump
        * update_band
        * update_bias_group
        * update_experiment
        * apply_to_pysmurf_instance
    * make_parser
    * DetConfig
        * parse_args
        * load_config_files
        * dump_configs
        * get_smurf_control
* util.py
    * rtm_bit_to_volt -- param
    * CHANS_PER_BAND -- param, duplicate 2x
    * TermColors -- Class
    * cprint
    * make_filename
    * get_tracking_kwargs
    * get_psd
    * SectionTimer -- Class
        * start_section
        * stop
        * reset
        * summary
    * dev_cfg_from_pysmurf
    * get_wls_from_am -- duplicate
    * invert_mask
    * get_r2
    * Registers -- Class

