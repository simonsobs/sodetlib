
def loopback_health_check(S, cfg, bands=None, uc_attens=None, dc_attens=None):
    if bands is None:
        bands =S._bands

    if uc_attens is None:
        uc_attens = [0, 1, 2, 4, 8, 16, 31]
    else:
        uc_attens = np.atleast_1d(uc_attens)

    if dc_attens is None:
        dc_attens = [0, 1, 2, 4, 8, 16, 31]
    else:
        dc_attens = np.atleast_1d(dc_attens)

    uc_resps = [[] for _ in bands]
    dc_resps = [[] for _ in bands]

    # Run UC sweep at dc atten = 0
    for b in bands:
        S.set_att_dc(b, 0)

    print("Runnin UC Sweep:")
    for att in tqdm(uc_attens):
        for b in bands:
            S.set_att_uc(b, att)
        time.sleep(0.1)
        for b in bands:





uc_resps, dc_resps = [[] for _ in bands], [[] for _ in bands]

for b in bands:
    S.set_att_dc(b, 0)

for att in attens:
    for b in bands:
        S.set_att_uc(b, att)
    time.sleep(0.1)
    for b in bands:
        uc_resps[b].append(
            S.full_band_resp(band=b, make_plot=False, save_plot=False, show_plot=False,
                     save_data=True, n_scan=n_scan_per_band,
                     correct_att=False)
        )

for b in bands:
    S.set_att_uc(b, 0)

for att in attens:
    for b in bands:
        S.set_att_dc(b, att)
    time.sleep(0.1)
    for b in bands:
        dc_resps[b].append(
            S.full_band_resp(band=b, make_plot=False, save_plot=False, show_plot=False,
                     save_data=True, n_scan=n_scan_per_band,
                     correct_att=False)
        )

