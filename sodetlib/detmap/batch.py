import os
import sys
from operator import itemgetter
from sodetlib.detmap.meta_select import abs_path_detmap
from sodetlib.detmap.simple_csv import read_csv, find_data_path
from sodetlib.detmap.makemap import MapMaker


class Manifest:
    def __init__(self, path, output_dir=None, overwrite=True, plot_pdf=False):
        # settings
        self.manifest_path = path
        if output_dir is None:
            self.output_dir = os.path.join(abs_path_detmap, 'output')
        else:
            self.output_dir = output_dir
        self.overwrite = overwrite
        self.plot_pdf = plot_pdf

        #  data variables populated in other methods
        self.manifest_data = None
        self.per_array_map_makers = {}

        # auto read-in
        self.read()
        self.process()

    def __iter__(self):
        for source in sorted(self.manifest_data.keys()):
            manifest_this_source = self.manifest_data[source]
            for single_manifest_row in sorted(manifest_this_source, key=itemgetter('array_name')):
                yield single_manifest_row

    def read(self):
        _data_by_column, manifest_data_by_row = read_csv(path=self.manifest_path)
        self.manifest_data = {}
        for single_row in manifest_data_by_row:
            source = single_row['source'].lower()
            if source not in self.manifest_data.keys():
                self.manifest_data[source] = []

            single_row['local_path'] = find_data_path(single_row['simons1_path'])
            if sys.platform == 'win32':
                single_row['local_dir_path'], single_row['tune_filename'] = single_row['local_path'].rsplit('\\', 1)
            else:
                single_row['local_dir_path'], single_row['tune_filename'] = single_row['local_path'].rsplit('/', 1)
            self.manifest_data[source].append(single_row)

    def process(self):
        for single_manifest_row in list(self):
            array_name = single_manifest_row['array_name']
            north_is_highband = single_manifest_row['north_is_highband']
            map_maker_key = (array_name, north_is_highband)
            if map_maker_key not in self.per_array_map_makers.keys():
                self.per_array_map_makers[map_maker_key] = MapMaker(output_parent_dir=self.output_dir,
                                                                    north_is_highband=north_is_highband,
                                                                    array_name=array_name,
                                                                    overwrite_plot=self.overwrite,
                                                                    overwrite_csv_output=self.overwrite,
                                                                    plot_pdf=self.plot_pdf)
            tunefile = single_manifest_row['local_path']
            single_manifest_row['operate_tune_data'] = \
                self.per_array_map_makers[map_maker_key].make_map_smurf(tunefile=tunefile)
