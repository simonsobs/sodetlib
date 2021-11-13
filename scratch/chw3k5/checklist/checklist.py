import os

verbose = True
bands_full_band_response = [f'{n}' for n in range(8)]
band_num_normal_opt = 0
band_num_opt = 4
slot_num = 3
stream_time = 20.0
fmin = 5.0
fmax = 50.0
fs = 200.0
nperseg = 2 ** 16
detrend = 'constant'


pton_mode = False
print_help = True
print_os_strings = True

do_amplifier_check = False
do_full_band_response = True
do_ufm_optimize = True


true_set = {'y', 'yes', 't', 'true', 'g', 'good'}

base_scratch_dir = '/sodetlib/scratch'
# base_scratch_dir = '/Users/cwheeler/PycharmProjects/sodetlib/scratch'
yuhan_dir = os.path.join(base_scratch_dir, 'yuhanw')
daniel_dir = os.path.join(base_scratch_dir, 'daniel')
caleb_dir = os.path.join(base_scratch_dir, 'chw3k5')
argparse_files_dir = os.path.join(caleb_dir, 'checklist', 'scripts')
# argparse_files_dir = 'scripts'

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


def commanding_mode_selector(file_basename, ocs_arg):
    if pton_mode:
        exec(open(all_files[python_file_basename]).read())
    else:
        full_path = os.path.join(argparse_files_dir, f'{file_basename}.py')
        python_cmd_str = f"python3 {full_path}"
        # print a the helps statements, skips the sending actual commands.
        if print_help:
            print(f"\nPrinting the help page for {file_basename}.py\n")
            os.system(f"{python_cmd_str} -h")
        # assemble the arguments
        for arg_part in ocs_arg:
            python_cmd_str += f' {arg_part}'
        # add the verbosity argument
        if verbose:
            ocs_arg.append('--verbose')
        else:
            ocs_arg.append('--no-verbose')
        # optionally print the the command or send the assembled python command
        if print_help or print_os_strings:
            print(f'\nCMD string: {python_cmd_str}\n')
        else:
            os.system(python_cmd_str)
    return


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
        python_file_basename = 'full_band_response'
        ocs_arg_list = bands_full_band_response
        ocs_arg_list.extend([f'--slot', f'{slot_num}',
                             f'--n-scan-per-band', f'{5}',
                             f'--wait-bwt-bands-sec', f'{5}'])
        commanding_mode_selector(file_basename=python_file_basename, ocs_arg=ocs_arg_list)

        # os.system(f"python3 {full_file_path} 0 1 2 3 4 5 6 7 --slot {slot_num} --verbose --n-scan-per-band {5} " +
        #           f"--wait-bwt-bands-sec {5}")
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
    python_file_basename = 'ufm_optimize_quick_normal'
    ocs_arg_list = [f'{band_num_normal_opt}',
                    f'--slot', f'{slot_num}',
                    f'--stream-time', f'{stream_time}',
                    f'--fmin', f'{fmin}',
                    f'--fmax', f'{fmax}',
                    f'--fs', f'{fs}',
                    f'--nperseg', f'{nperseg}',
                    f'--detrend', f'{detrend}']
    if verbose and not pton_mode:
        print('UFM optimize - Starting')
        print('Optimizing TES biases in the normal stage\n' +
              f'takes median noise from {fmin}Hz to {fmax}Hz\n' +
              'different noise levels here are based on phase 2 \n' +
              'noise target and noise model after considering\n' +
              'johnson noise at 100mK')
    commanding_mode_selector(file_basename=python_file_basename, ocs_arg=ocs_arg_list)

    # biased superconducting
    python_file_basename = 'ufm_optimize_quick_normal'
    ocs_arg_list = [f'{band_num_opt}',
                    f'--slot', f'{slot_num}',
                    f'--stream-time', f'{stream_time}',
                    f'--fmin', f'{fmin}',
                    f'--fmax', f'{fmax}',
                    f'--fs', f'{fs}',
                    f'--nperseg', f'{nperseg}',
                    f'--detrend', f'{detrend}']
    if verbose and not pton_mode:
        print('Optimizing TES biases in the superconducting stage\n' +
              f'takes median noise from {fmin}Hz to {fmax}Hz\n' +
              'different noise levels here are based on phase 2 \n' +
              'noise target and noise model after considering\n' +
              'johnson noise at 100mK')
    commanding_mode_selector(file_basename=python_file_basename, ocs_arg=ocs_arg_list)

    if verbose:
        print('  UFM optimize - Finished\n')