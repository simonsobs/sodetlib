import glob
import pandas as pd

from detector_map import smurf_chan_to_realized_freq, all_smurf_chan_to_realized_freq, vna_freq_to_muxpad, \
    smurf_to_mux, mux_band_to_mux_posn, get_pad_to_wafer, smurf_to_detector

# SMuRF channel assignments
chan_assign = glob.glob("Sv5/*channel_assignment*.txt")
chan_assign.sort()
N_chan = chan_assign[4:]
S_chan = chan_assign[:4]
highband = 'S'  # Which side is 6-8 GHz
dark_bias_lines = []  # If certain sides are covered

N_vna_map = "Sv5/N_vna_map.csv"
S_vna_map = "Sv5/S_vna_map.csv"
posnfile = "Sv5/band2posn.csv"
design_file = "../metadata/umux_32_map.pkl"
waferfile = "metadata/UFM_Si_corrected.csv"

# South side VNA match
S_smurf2freq = all_smurf_chan_to_realized_freq(S_chan)
S_vna2pad = vna_freq_to_muxpad(S_vna_map, design_file)
S_smurf2mux = smurf_to_mux(S_smurf2freq, S_vna2pad, threshold=0.01)

# Repeat for north side VNA.
N_smurf2freq = all_smurf_chan_to_realized_freq(N_chan)
N_vna2pad = vna_freq_to_muxpad(N_vna_map, design_file)
N_smurf2mux = smurf_to_mux(N_smurf2freq, N_vna2pad, threshold=0.01)

# Put things together
smurf2mux = pd.concat([S_smurf2mux, N_smurf2mux], axis=0)

# All Smurf frequency to mux bondpad assignments
smurf2padloc = mux_band_to_mux_posn(smurf2mux, posnfile, highband='S')
# smurf2padloc


wafer_info = get_pad_to_wafer(waferfile, dark_bias_lines=dark_bias_lines)  # Load information about silicon assembly
smurf2det = smurf_to_detector(smurf2padloc, wafer_info)

smurf2det.to_csv("MD14_Sv5_smurf2det.csv", index=False)

"""
Copper Assembly (change design_file)
"""

# SMuRF channel assignments
chan_assign = glob.glob("Cv4/run1/*channel_assignment*.txt")
chan_assign.sort()
N_chan = chan_assign[4:]
S_chan = chan_assign[:4]
highband = 'N'
dark_bias_lines = []  # If certain sides are covered

N_vna_map = "Cv4/run1/N_vna_map.csv"
S_vna_map = "Cv4/run1/S_vna_map.csv"
posnfile = "Cv4/band2posn.csv"
design_file = "../metadata/umux_32_map.pkl"
waferfile = "metadata/copper_map_corrected.csv"

# South side VNA match
S_smurf2freq = all_smurf_chan_to_realized_freq(S_chan)
S_vna2pad = vna_freq_to_muxpad(S_vna_map, design_file)
S_smurf2mux = smurf_to_mux(S_smurf2freq, S_vna2pad, threshold=0.01)

# Repeat for north side VNA.
N_smurf2freq = all_smurf_chan_to_realized_freq(N_chan)
N_vna2pad = vna_freq_to_muxpad(N_vna_map, design_file)
N_smurf2mux = smurf_to_mux(N_smurf2freq, N_vna2pad, threshold=0.01)

smurf2mux = pd.concat([S_smurf2mux, N_smurf2mux], axis=0)

# All Smurf frequency to mux bondpad assignments
smurf2padloc = mux_band_to_mux_posn(smurf2mux, posnfile, highband='N')
# smurf2padloc

wafer_info = get_pad_to_wafer(waferfile, dark_bias_lines=dark_bias_lines)  # Load information about copper assembly
smurf2det = smurf_to_detector(smurf2padloc, wafer_info)  # Complete map
# smurf2det

smurf2det.to_csv("MD16_Cv4_smurf2det.csv", index=False)
# N_smurf2freq
