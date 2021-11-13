import os

verbose = True
pton_mode = True

do_amplifier_check = False
do_full_band_response = False
do_ufm_optimize = True


true_set = {'y', 'yes', 't', 'true', 'g', 'good'}

base_scratch_dir = '/sodetlib/scratch'
# base_scratch_dir = '/Users/cwheeler/PycharmProjects/sodetlib/scratch'
yuhan_dir = os.path.join(base_scratch_dir, 'yuhanw')
daniel_dir = os.path.join(base_scratch_dir, 'daniel')
caleb_dir = os.path.join(base_scratch_dir, 'chw3k5')
argparse_files_dir = os.path.join(caleb_dir, 'checklist', 'scripts')

yuhan_dir_caleb_branch = os.path.join(caleb_dir, 'checklist/unversioned-yuhan')
daniel_dir_caleb_branch = os.path.join(caleb_dir, 'checklist/unversioned-daniel')

versioned_files = {'full_band_response': os.path.join(yuhan_dir, 'full_band_response.py'),
                   'ufm_biasstep_loop': os.path.join(yuhan_dir, 'ufm_biasstep_loop.py')}


out_of_date_files_yuhan = {'ufm_optimize_quick_normal': os.path.join(yuhan_dir_caleb_branch, 'ufm_optimize_quick_normal.py'),
                           'ufm_optimize_quick': os.path.join(yuhan_dir_caleb_branch, 'ufm_optimize_quick.py'),
                           'noise_stack_by_band_new': os.path.join(yuhan_dir_caleb_branch, 'noise_stack_by_band_new.py'),
                           'ufm_noise_in_transition': os.path.join(yuhan_dir_caleb_branch, 'ufm_noise_in_transition.py'),
                           'uxm_bath_iv_noise_biasstep': os.path.join(yuhan_dir_caleb_branch, 'uxm_bath_iv_noise_biasstep.py'),
                           }

out_of_date_files_daniel = {'uxm_setup': os.path.join(daniel_dir_caleb_branch, 'uxm_setup.py'),
                            'uxm_relock': os.path.join(daniel_dir_caleb_branch, 'uxm_relock.py')}

unversioned_files_daniel = {'tes_yield': os.path.join(daniel_dir_caleb_branch, 'tes_yield.py'),
                            'uxm_bath_iv_noise': os.path.join(daniel_dir_caleb_branch, 'uxm_bath_iv_noise.py'),
                            }
all_files = {}
all_files.update(versioned_files)
all_files.update(out_of_date_files_yuhan)
all_files.update(out_of_date_files_daniel)
all_files.update(unversioned_files_daniel)


"""
Amplifier check
"""
if do_amplifier_check:
    input("The cold amplifier check is not currently written, press enter to continue")


"""
Full band response
"""
if do_full_band_response:
    if verbose:
        print('Full Band Response - Starting')
    finished_full_band_response = False
    while not finished_full_band_response:
        if pton_mode:
            exec(open(all_files['full_band_response']).read())
        else:
            full_file_path = os.path.join(argparse_files_dir, 'full_band_response.py')
            os.system(f"python3 {full_file_path} 0 1 2 3 4 5 6 7 --slot {3} --verbose --n-scan-per-band {5} " +
                      f"--wait-bwt-bands-sec {5}")
        full_band_response_good = input("Is the full band response good? [yes/no]: ")
        if full_band_response_good.lower() in true_set:
            finished_full_band_response = True
        else:
            raise RuntimeError("A rough sweep using a VNA is recommended as diagnostic tool.")
    else:
        if verbose:
            print('  Full Band Response - Finished\n')


"""
UFM optimize
"""
if do_ufm_optimize:
    if verbose:
        print('UFM optimize - Starting')
    finished_full_band_response = False
    while not finished_full_band_response:
        if pton_mode:
            exec(open(all_files['ufm_optimize_quick_normal']).read())
        else:
            full_file_path = os.path.join(argparse_files_dir, 'ufm_optimize_quick_normal.py')
            os.system(f"python3 {full_file_path} ")
    else:
        if verbose:
            print('  UFM optimize - Finished\n')
