import os
import warnings

array_name_allowed_first_letters = {'s', 'c', 'm', 'u'}
array_name_allowed_second_letters = {'v'}
array_name_dis_allowed_ints = {1, 2, 3, 10}
copper_versions_to_rename_to_u = {8}


# get the absolute path fo the detector mapping code
dir_this_file = os.path.dirname(os.path.realpath(__file__))
abs_path_sodetlib, _ = dir_this_file.rsplit("sodetlib", 1)
abs_path_detmap = os.path.join(abs_path_sodetlib, "sodetlib", "detmap")

# get the absolute dir path for the configuration files
abs_path_metadata_files_default = os.path.join(abs_path_detmap, 'meta')

# set the default metadate file names and file paths.
metadata_waferfile_default = "copper_map_corrected.csv"
metadata_designfile_default = "umux_32_map.pkl"
mux_pos_to_mux_band_file_suffix = 'mux_pos_num_to_mux_band_num.csv'
metadata_mux_pos_to_mux_band_file_default = mux_pos_to_mux_band_file_suffix


def is_int(test_str):
    try:
        int(test_str)
        return True
    except ValueError:
        return False


def get_metadata_files_for_array(array_name: str):
    # Do a lot of parsing and error checking to make sure the array name is understood
    array_name = array_name.lower()
    first_letter = array_name[0:1]
    if first_letter not in array_name_allowed_first_letters:
        raise ValueError(f"The first letter '{first_letter}' of the array name '{array_name}' is not one of the " +
                         f"allowed letters: {array_name_allowed_first_letters}")

    second_character = array_name[1:2]
    if is_int(test_str=second_character):
        # if the second charter is a number the rest of array name should be an integer.
        num_str = array_name[1:]
    else:
        # if the second charter is a letter then it must be an allowed letter
        if second_character not in array_name_allowed_second_letters:
            raise ValueError(f"The second letter '{second_character}' of the array name '{array_name}' is not one of " +
                             f"the allowed letters: {array_name_allowed_second_letters}")
        num_str = array_name[2:]
    try:
        version_num = int(num_str)
    except ValueError:
        raise ValueError(f"The last part of the array name '{array_name}' is expected to be an integer, " +
                         f"got '{num_str}' instead")
    if version_num in array_name_dis_allowed_ints:
        raise KeyError(f"Array version numer '{version_num}' is in the array name '{array_name}', is one of the " +
                       f"retired and therefor disallowed, version numbers, namely: {array_name_dis_allowed_ints}.")
    # Map to the appropriate metadata.
    if first_letter == 's':
        waferfile = 'UFM_Si_corrected.csv'
    else:
        waferfile = 'copper_map_corrected.csv'
    if first_letter == 'c':
        if first_letter in copper_versions_to_rename_to_u:
            first_letter = 'u'
        else:
            first_letter = 'm'
    mux_pos_to_mux_band_file = f'{first_letter.upper()}v{version_num}_{mux_pos_to_mux_band_file_suffix}'
    return metadata_designfile_default, waferfile, mux_pos_to_mux_band_file


def check_existence(dirname, basename):
    path = os.path.join(dirname, basename)
    if not os.path.exists(dirname):
        raise FileNotFoundError(f'The the directory {dirname} for metadata file {basename} does not exist. ' +
                                f'Check the path inputs, and try again')
    if not os.path.exists(path):
        raise FileNotFoundError(f'The the metadata file {basename} is not located in the direcotry {dirname}. ' +
                                f'Check the path inputs, and try again')
    return path


def get_metadata_files(array_name: str = None, abs_path_metadata_files: str = None, verbose=False):
    if array_name is None:
        metadata_designfile = metadata_designfile_default
        waferfile = metadata_waferfile_default
        mux_pos_to_mux_band_file = metadata_mux_pos_to_mux_band_file_default
        if array_name is not None:
            warnings.warn("Warning! no array name was specified, using the default meta data configurations!")
            verbose = True
    else:
        metadata_designfile, waferfile, mux_pos_to_mux_band_file = get_metadata_files_for_array(array_name=array_name)
    if abs_path_metadata_files is None:
        abs_path_metadata_files = abs_path_metadata_files_default
    designfile_path = check_existence(dirname=abs_path_metadata_files, basename=metadata_designfile)
    waferfile_path = check_existence(dirname=abs_path_metadata_files, basename=waferfile)
    mux_pos_to_mux_band_file_path = check_existence(dirname=abs_path_metadata_files, basename=mux_pos_to_mux_band_file)
    if verbose:
        if array_name is None:
            array_name = 'default'
        print(f"Metadata files for the '{array_name}' array localed in the directory: {abs_path_metadata_files}")
        print(f"  Resonator Design         : {metadata_designfile}")
        print(f"  Detector Layout          : {waferfile}")
        print(f"  mux-band to mux-position : {mux_pos_to_mux_band_file}")
    return waferfile_path, designfile_path, mux_pos_to_mux_band_file_path


if __name__ == '__main__':
    a_waferfile_path, a_designfile_path, a_mux_pos_to_mux_band_file_path = \
        get_metadata_files(array_name='Cv4', verbose=True)
