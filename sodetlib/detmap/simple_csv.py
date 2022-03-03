import os
from operator import itemgetter
from sodetlib.detmap.meta_select import abs_path_detmap
from sodetlib.detmap.example.download_example_data import sample_data_init

default_null_strings = {'', 'None', 'none', 'null', 'NaN', 'nan'}
default_true_strings = {'Y', 'y', 'True', 'true'}
default_false_strings = {'N', 'n', 'False', 'false'}


def make_int(test_num_str):
    """
    Parameters
    ----------
    test_num_str:
        str, required. A string to test to see it it can be cast into and integer.

    Returns int, if the string can be cast into and integer, otherwise it returns the original string.
    -------
    """
    try:
        return int(test_num_str)
    except ValueError:
        return test_num_str


def make_num(test_datum):

    # tests to se if this string is an int
    test_datum_maybe_int = make_int(test_num_str=test_datum)
    if isinstance(test_datum_maybe_int, int):
        return test_datum_maybe_int
    else:
        # either a float or a sting.
        try:
            return float(test_datum_maybe_int)
        except ValueError:
            return test_datum_maybe_int


def format_datum(test_datum, null_strs=None, true_strs=None, false_strs=None):
    # strip off spaces and newline charters
    test_datum_stripped = test_datum.strip()
    # Null handling
    if null_strs is None:
        null_strs = default_null_strings
    if not isinstance(null_strs, set):
        null_strs = set(null_strs)
    # True handling
    if true_strs is None:
        true_strs = default_true_strings
    if not isinstance(true_strs, set):
        true_strs = set(true_strs)
    # False handling
    if false_strs is None:
        false_strs = default_false_strings
    if not isinstance(false_strs, set):
        false_strs = set(false_strs)
    # see if this an expected null, true, or false value, else move on to number testing
    if test_datum_stripped in null_strs:
        return None
    elif test_datum_stripped in true_strs:
        return True
    elif test_datum_stripped in false_strs:
        return False
    else:
        return make_num(test_datum=test_datum_stripped)


def line_format(raw_line, delimiter=','):
    return [format_datum(datum_raw) for datum_raw in raw_line.split(delimiter)]


def read_csv(path, header=None):
    with open(path, 'r') as f:
        if header is None:
            # if header is not specified, the first line on then file is expected to be the header.
            raw_header = f.readline()
            header = [raw_column_name.strip().lower() for raw_column_name in raw_header.split(',')]
        data_by_column = {column_name: [] for column_name in header}
        data_by_row = []
        for raw_line in f.readlines():
            line_data = line_format(raw_line=raw_line)
            line_dict = {}
            for column_name, datum_converted in zip(header, line_data):
                data_by_column[column_name].append(datum_converted)
                line_dict[column_name] = datum_converted
            data_by_row.append(line_dict)
    return data_by_column, data_by_row


def find_data_path(simons1_path):
    if os.path.exists(simons1_path):
        return simons1_path
    simons1_dir_path, tune_filename = simons1_path.rsplit('/', 1)
    sample_data_dir_path = os.path.join(abs_path_detmap, 'sample_data')
    tune_file_sample_data_path = os.path.join(sample_data_dir_path, tune_filename)
    if not os.path.exists(sample_data_dir_path):
        sample_data_init(del_dir=False)
    if os.path.exists(tune_file_sample_data_path):
        return tune_file_sample_data_path
    tune_data_dir_path = os.path.join(abs_path_detmap, 'tunes')
    tune_file_tune_data_path = os.path.join(tune_data_dir_path, tune_filename)
    if not os.path.exists(tune_data_dir_path):
        sample_data_init(del_dir=False, zip_file_id='1BugpuMsoKtlaxqagIt2d0uaQQhWcMb_V', folder_name='tunes')
    if os.path.exists(tune_file_tune_data_path):
        return tune_file_tune_data_path
    else:
        raise FileNotFoundError(f'cannot fine the tune file {tune_filename} ' +
                                f'at the following paths {simons1_path}\n{tune_file_sample_data_path}' +
                                f'\n{tune_file_tune_data_path}')


if __name__ == '__main__':
    data_by_column, data_by_row = read_csv(path=os.path.join('sample_data', 'coldloadramp_example.csv'))
