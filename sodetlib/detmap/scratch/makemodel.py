import os
import getpass
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
from multiprocessing.pool import Pool
from sodetlib.detmap.layout_data import get_layout_data
from sodetlib.detmap.channel_assignment import OperateTuneData, get_mux_band_to_mux_pos_dict
from sodetlib.detmap.meta_select import get_metadata_files

waferfile_path_default, designfile_path_default, mux_pos_to_mux_band_file_path_default = \
    get_metadata_files(verbose=True)
# Debug mode
debug_mode = True
# multiprocessing
# the 'assumption' of max threads is that we are cpu limited in processing,
# so we use should not use more than a computer's available threads
max_threads = int(os.cpu_count())
# Try to strike a balance between the best performance and computer usability during processing
balanced_threads = max(max_threads - 2, 2)
# Use onl half of the available threads for processing
half_threads = int(np.round(os.cpu_count() * 0.5))
current_user = getpass.getuser()
if debug_mode:
    # this will do standard linear processing.
    multiprocessing_threads = None
elif current_user == "chw3k5":
    multiprocessing_threads = balanced_threads  # Caleb's other computers
elif current_user in "cwheeler":
    multiprocessing_threads = max_threads  # Mac Pro 8-core intel core i9 processor 16 threads
else:
    multiprocessing_threads = balanced_threads


def generate_model(design_tune_data, spread, offset_mhz, sigma_mhz, lost_resonators_per_band_range):
    for band in design_tune_data.keys():
        des_this_band = design_tune_data[band]
        # apply a linear offset to all bands of the array
        des_trans = spread * des_this_band + offset_mhz
        # add normal random noise to each resonator individually
        des_with_wiggles = random.normal(loc=0.0, scale=sigma_mhz, size=len(des_trans)) + des_trans
        # record the mapping of simulated to design data to use as an answer key
        sim_to_des = {sim: des for sim, des in zip(des_with_wiggles, des_this_band)}
        # randomly remove a (random) number of resonator from each band
        res_num_remove = random.randint(low=lost_resonators_per_band_range[0],
                                        high=lost_resonators_per_band_range[1] + 1)
        random.shuffle(des_with_wiggles)
        res_lost = np.array(sorted(des_with_wiggles[:res_num_remove]))
        res_found = np.array(sorted(des_with_wiggles[res_num_remove:]))
        yield band, res_lost, res_found, sim_to_des


def get_model(design_tune_data, spread, offset_mhz, sigma_mhz, lost_resonators_per_band_range):
    simulated = {band: (res_lost, res_found, simulated_to_design)
                 for band, res_lost, res_found, simulated_to_design
                 in iter(generate_model(design_tune_data=design_tune_data, spread=spread,
                                        offset_mhz=offset_mhz, sigma_mhz=sigma_mhz,
                                        lost_resonators_per_band_range=lost_resonators_per_band_range))}
    # split the lost and found resonators, as well as the simulated_to_design data
    simulated_lost = {band: simulated[band][0] for band in simulated.keys()}
    simulated_found = {band: simulated[band][1] for band in simulated.keys()}
    simulated_to_design = {band: simulated[band][2] for band in simulated.keys()}
    return simulated_lost, simulated_found, simulated_to_design


def extract_strategy_mapping(tune_data_model, simulated_to_design):
    for tune_datum in list(tune_data_model.tune_data_with_design_data) + \
                      list(tune_data_model.tune_data_without_design_data):
        freq_simulated = tune_datum.freq_mhz
        freq_mapped = tune_datum.design_freq_mhz
        band_num = tune_datum.smurf_band
        simulated_to_design_this_band = simulated_to_design[band_num]
        if simulated_to_design_this_band[freq_simulated] == freq_mapped:
            yield True, band_num, freq_simulated
        else:
            yield False, band_num, freq_simulated


def compare_strategy(tune_data_model, simulated_to_design):
    mapped_correctly = {band_num: set() for band_num in simulated_to_design.keys()}
    mapped_incorrectly = {band_num: set() for band_num in simulated_to_design.keys()}
    [mapped_correctly[band_num].add(freq_simulated)
     if is_correctly_mapped else mapped_incorrectly[band_num].add(freq_simulated)
     for is_correctly_mapped, band_num, freq_simulated
     in iter(extract_strategy_mapping(tune_data_model, simulated_to_design))]
    return mapped_correctly, mapped_incorrectly


def full_model_sim(design_tune_data_arrays, sigma_mhz, spread, offset_mhz, lost_resonators_per_band_range,
                   design_tune_data, wafer_data, mux_band_to_mux_pos_dict, mapping_strategy):
    simulated_lost, simulated_found, simulated_to_design \
        = get_model(design_tune_data=design_tune_data_arrays, spread=spread, offset_mhz=offset_mhz,
                    sigma_mhz=sigma_mhz, lost_resonators_per_band_range=lost_resonators_per_band_range)
    tune_data_model = OperateTuneData(north_is_highband=True)
    tune_data_model.from_simulated(simulated_lost=simulated_lost, simulated_found=simulated_found,
                                   simulated_to_design=simulated_to_design)

    tune_data_model.map_design_data(design_tune_data, layout_position_path=mux_band_to_mux_pos_dict,
                                    mapping_strategy=mapping_strategy)
    tune_data_model.map_layout_data(layout_data=wafer_data)
    mapped_correctly, mapped_incorrectly = compare_strategy(tune_data_model=tune_data_model,
                                                            simulated_to_design=simulated_to_design)
    fraction_mapped_correctly = {band_num: float(len(mapped_correctly[band_num])) / float(len(simulated_found[band_num]))
                                 for band_num in mapped_correctly.keys()}
    results_summary = {band_num: {'sigma_mhz': sigma_mhz, 'spread': spread, 'offset_mhz': offset_mhz,
                                  'num_lost': len(simulated_lost[band_num]),
                                  'fraction_mapped_correctly': fraction_mapped_correctly[band_num]}
                       for band_num in mapped_correctly.keys()}
    # for band_num in results_summary.keys():
    #     if results_summary[band_num]['fraction_mapped_correctly'] == 0.0 and results_summary[band_num]['num_lost'] == 0:
    #         print(f'band_num: {band_num}')
    #         tune_data_model.plot_with_layout(show_plot=True, save_plot=False)
    #         break
    return results_summary


def full_model_sim_arg_to_kwargs(kwargs):
    return full_model_sim(**kwargs)


class TuneDataModelEngine:
    def __init__(self, design_file_path=None, wafer_path=None, layout_position_path=None):
        if design_file_path is None:
            self.design_file_path = designfile_path_default
        else:
            self.design_file_path = design_file_path
        if wafer_path is None:
            self.wafer_path = waferfile_path_default
        else:
            self.wafer_path = wafer_path
        if layout_position_path is None:
            self.layout_position_path = mux_pos_to_mux_band_file_path_default
        else:
            self.layout_position_path = layout_position_path

        self.design_data = OperateTuneData(design_file_path=self.design_file_path)
        design_data_by_band_raw = self.design_data.tune_data_side_band_res_index[None]
        self.design_tune_data_arrays = {band: np.array([design_data_by_band_raw[band][res_index].freq_mhz
                                                       for res_index in sorted(design_data_by_band_raw[band].keys())])
                                        for band in design_data_by_band_raw.keys()}
        self.wafer = get_layout_data(self.wafer_path)
        self.mux_band_to_mux_pos_dict = get_mux_band_to_mux_pos_dict(self.layout_position_path)

        self.results = None
        self.sigma_mhz_range = None
        self.spread_range = None
        self.offset_range = None
        self.lost_resonators_per_band_range = None

    def simulate(self, number_of_models: int,
                 sigma_mhz_range=(0.0001, 3.1),
                 spread_range=(0.995, 1.01), offset_range=(-30, 30),
                 lost_resonators_per_band_range=(0, 20)):
        self.sigma_mhz_range = sigma_mhz_range
        self.spread_range = spread_range
        self.offset_range = offset_range
        self.lost_resonators_per_band_range = lost_resonators_per_band_range
        model_kwargs = [{'design_tune_data_arrays': self.design_tune_data_arrays,
                         'sigma_mhz': random.uniform(low=self.sigma_mhz_range[0],
                                                     high=self.sigma_mhz_range[1]),
                         'spread': random.uniform(low=self.spread_range[0],
                                                  high=self.spread_range[1]),
                         'offset_mhz': random.uniform(low=self.offset_range[0],
                                                      high=self.offset_range[1]),
                         'lost_resonators_per_band_range': self.lost_resonators_per_band_range,
                         'design_tune_data': self.design_data,
                         'wafer_data': self.wafer,
                         'mux_band_to_mux_pos_dict': self.mux_band_to_mux_pos_dict,
                         'mapping_strategy': 'map_by_freq'
                         } for i in range(number_of_models)]
        if multiprocessing_threads is None:
            self.results = [full_model_sim_arg_to_kwargs(kwargs=kwargs) for kwargs in model_kwargs]
        else:
            with Pool(multiprocessing_threads) as p:
                self.results = p.map(full_model_sim_arg_to_kwargs, model_kwargs)
        print("Simulation complete")

    def plot(self):
        num_lost = []
        fraction_mapped_correctly = []
        sigma_mhz = []

        for a_result in self.results:
            for band in a_result.keys():
                results_this_band = a_result[band]
                num_lost.append(results_this_band['num_lost'])
                fraction_mapped_correctly.append(results_this_band['fraction_mapped_correctly'])
                sigma_mhz.append(results_this_band['sigma_mhz'])
        # initialize the plot
        left = 0.05
        bottom = 0.08
        right = 0.99
        top = 0.99
        x_total = right - left
        y_total = top - bottom

        y_between_plots = 0.1

        y_scatter = y_total * 0.7

        y_colorbar = y_total - y_scatter - y_between_plots

        coord_scatter = [left, top - y_scatter, x_total, y_scatter]
        coord_colorbar = [left, bottom, x_total, y_colorbar]
        fig = plt.figure(figsize=(16, 6))
        ax_scatter = fig.add_axes(coord_scatter, frameon=False)
        ax_colorbar = fig.add_axes(coord_colorbar, frameon=False)
        ax_colorbar.tick_params(axis='x',  # changes apply to the x-axis
                                which='both',  # both major and minor ticks are affected
                                bottom=False,  # ticks along the bottom edge are off
                                top=False,  # ticks along the top edge are off
                                labelbottom=False)
        ax_colorbar.tick_params(axis='y',  # changes apply to the x-axis
                                which='both',  # both major and minor ticks are affected
                                left=False,  # ticks along the bottom edge are off
                                right=False,  # ticks along the top edge are off
                                labelleft=False)

        # set the color scale
        norm = colors.Normalize(vmin=self.sigma_mhz_range[0], vmax=self.sigma_mhz_range[1])
        cmap = plt.get_cmap('gist_rainbow_r')
        scalar_map = cm.ScalarMappable(norm=norm, cmap=cmap)
        color_vals = [scalar_map.to_rgba(a_sigma_mhz) for a_sigma_mhz in sigma_mhz]
        fig.colorbar(scalar_map, ax=ax_colorbar, orientation='horizontal', fraction=1.0)

        ax_scatter.scatter(fraction_mapped_correctly, num_lost, ls='None', marker='o', alpha=0.4, c=color_vals,
                           vmin=self.sigma_mhz_range[0], vmax=self.sigma_mhz_range[1])
        ax_scatter.set_xlabel('Fraction Mapped Correctly')
        ax_scatter.set_ylabel('Number of Resonator lost at Random')
        fig.suptitle('Sigma of Normal per resonator error (kHz)', x=0.5, y=0.0, ha='center', va='bottom',
                     fontsize=11)
        plt.show(block=True)


if __name__ == '__main__':
    tune_data_model_engine = TuneDataModelEngine()
    tune_data_model_engine.simulate(number_of_models=100)
    tune_data_model_engine.plot()
