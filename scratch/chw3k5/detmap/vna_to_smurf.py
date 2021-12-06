def emulate_smurf_bands(shift_mhz=0.0, smurf_bands=None):
    # Auto-set the bands to the default
    if smurf_bands is None:
        smurf_bands = [band for band in range(8)]
    # define the upper and lower bands from the smurf bands
    lower_bands = [band for band in smurf_bands if band < 4]
    available_lower_bands = set(lower_bands)
    upper_bands = [band for band in smurf_bands if 3 < band]
    available_upper_bands = set(upper_bands)

    real_band_bounds_mhz = {}
    all_data_band_bounds_mhz = {}
    all_data_lower_band_bounds_mhz = {}
    all_data_upper_band_bounds_mhz = {}
    for smurf_band in smurf_bands:
        # simulate the real bounds
        lower_bound_mhz = 4000.0 + (smurf_band * 500.0) + shift_mhz
        upper_bound_mhz = lower_bound_mhz + 500.0
        real_band_bounds_mhz[smurf_band] = (lower_bound_mhz, upper_bound_mhz)
        # simulate bounds that will not cut out any data
        # by default all bands are like this
        all_data_lower_bound_mhz = lower_bound_mhz
        all_data_upper_bound_mhz = upper_bound_mhz
        if smurf_band == smurf_bands[0]:
            # set the first band to go to zero
            all_data_lower_bound_mhz = 0.0
        elif smurf_band == smurf_bands[-1]:
            # set the last band to go up to infinity
            all_data_upper_bound_mhz = float('inf')
        all_data_band_bounds_mhz[smurf_band] = (all_data_lower_bound_mhz, all_data_upper_bound_mhz)

        # set bounds just for the lower bands i.e. bands
        if smurf_band in available_lower_bands:
            all_data_lower_band_lower_bound_mhz = lower_bound_mhz
            all_data_lower_band_upper_bound_mhz = upper_bound_mhz
            if smurf_band == lower_bands[0]:
                # set the first band to go to zero
                all_data_lower_band_lower_bound_mhz = 0.0
            elif smurf_band == lower_bands[-1]:
                # set the last band to go up to infinity
                all_data_lower_band_upper_bound_mhz = float('inf')
            all_data_lower_band_bounds_mhz[smurf_band] = (all_data_lower_band_lower_bound_mhz,
                                                          all_data_lower_band_upper_bound_mhz)

        # set bounds just for the lower bands i.e. bands
        if smurf_band in available_upper_bands:
            all_data_upper_band_lower_bound_mhz = lower_bound_mhz
            all_data_upper_band_upper_bound_mhz = upper_bound_mhz
            if smurf_band == upper_bands[0]:
                # set the first band to go to zero
                all_data_upper_band_lower_bound_mhz = 0.0
            elif smurf_band == upper_bands[-1]:
                # set the last band to go up to infinity
                all_data_upper_band_upper_bound_mhz = float('inf')
            all_data_upper_band_bounds_mhz[smurf_band] = (all_data_upper_band_lower_bound_mhz,
                                                          all_data_upper_band_upper_bound_mhz)

    return real_band_bounds_mhz, all_data_band_bounds_mhz, all_data_lower_band_bounds_mhz, \
               all_data_upper_band_bounds_mhz