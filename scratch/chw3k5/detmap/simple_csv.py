import os

# working_dir = os.path.dirname(os.path.abspath(__file__))


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


def make_num(test_datum, null_str=None):
    # see if this is an empty string, or and expected null value
    if test_datum == '' or test_datum == null_str:
        return None
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


def line_format(raw_line):
    line_list = []
    for datum_raw in raw_line.split(','):
        datum_stripped = datum_raw.strip()
        if datum_stripped == '':
            datum = None
        else:
            datum = make_num(datum_stripped)
        line_list.append(datum)
    return line_list


def read_csv(path, header=None):
    with open(path, 'r') as f:
        if header is None:
            header = line_format(f.readline())
        data_by_column = {column_name: [] for column_name in header}
        data_by_row = []
        for raw_line in f.readlines():
            line_data = line_format(raw_line=raw_line)
            line_dict = {}
            for column_name, datum in zip(header, line_data):
                datum_converted = make_num(test_datum=datum)
                data_by_column[column_name].append(datum_converted)
                line_dict[column_name] = datum_converted
            data_by_row.append(line_dict)
    return data_by_column, data_by_row


if __name__ == '__main__':
    data_by_column, data_by_row = read_csv(path=os.path.join('sample_data', 'coldloadramp_example.csv'))
