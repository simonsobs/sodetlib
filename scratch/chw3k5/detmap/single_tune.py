from typing import NamedTuple, Optional, Union


class TuneDatum(NamedTuple):
    """
    For the a single Tune datum (unique smurf_band and resonator_index)
    """
    smurf_band: int
    res_index: int
    freq_mhz: float
    is_north: Optional[bool] = None
    is_highband: Optional[bool] = None
    smurf_channel: Optional[int] = None
    smurf_subband: Optional[int] = None
    bond_pad: Optional[int] = None
    mux_band: Optional[int] = None
    mux_channel: Optional[int] = None
    mux_subband: Optional[str] = None
    mux_layout_position: Optional[int] = None
    design_freq_mhz: Optional[float] = None
    bias_line: Optional[int] = None
    pol: Optional[str] = None
    bandpass: Optional[Union[int, str]] = None
    det_row: Optional[int] = None
    det_col: Optional[int] = None
    rhomb: Optional[str] = None
    is_optical: Optional[bool] = None
    det_x: Optional[float] = None
    det_y: Optional[float] = None

    def __str__(self):
        output_str = ''
        for column in list(self._fields):
            output_str += f'{self.__getattribute__(column)},'
        # the last comma is not needed
        final_output = output_str[:-1]
        return final_output

    def __iter__(self):
        for field_key in self._fields:
            yield self.__getattribute__(field_key)

    def dict(self):
        return {field_key: self.__getattribute__(field_key) for field_key in self._fields}

    def dict_without_none(self):
        return {field_key: self.__getattribute__(field_key) for field_key in self._fields
                if self.__getattribute__(field_key) is not None}


tune_data_column_names = list(TuneDatum._fields)
tune_data_header = f'{tune_data_column_names[0]}'
for column_name in tune_data_column_names[1:]:
    tune_data_header += f',{column_name}'
