import os
designfile = "umux_32_map.pkl"
mux_pos_to_mux_band_file_suffix = 'mux_pos_num_to_mux_band_num.csv'

array_name_allowed_first_letters = {'s', 'c', 'm', 'u'}
array_name_allowed_second_letters = {'v'}
array_name_dis_allowed_ints = {1, 2, 3, 10}
copper_versions_to_rename_to_u = {8}


def is_int(test_str):
    try:
        an_int = int(test_str)
        return True
    except ValueError:
        return False


def get_metadata_filenames(array_name: str):
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
    return designfile, waferfile, mux_pos_to_mux_band_file
