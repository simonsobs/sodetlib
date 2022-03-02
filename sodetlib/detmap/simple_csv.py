import os
from operator import itemgetter

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


def manifest_parse(path):
    _data_by_column, manifest_data_by_row = read_csv(path=path)
    manifest_data = {}
    for single_row in manifest_data_by_row:
        source = single_row['source'].lower()
        if source not in manifest_data.keys():
            manifest_data[source] = []
        single_row['parent_dir_path'], single_row['tune_filename'] = single_row['simons1_path'].rsplit('/', 1)
        manifest_data[source].append(single_row)
    manifest_ordered = {}
    for source in sorted(manifest_data.keys()):
        manifest_this_source = manifest_data[source]
        manifest_ordered[source] = sorted(manifest_this_source, key=itemgetter('array_name'))
    return manifest_ordered


if __name__ == '__main__':
    data_by_column, data_by_row = read_csv(path=os.path.join('sample_data', 'coldloadramp_example.csv'))
    manifest = manifest_parse(path=os.path.join('example', 'manifest.csv'))
