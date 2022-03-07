"""
A Developmental Environment for Testing New features.
"""
import os
from getpass import getuser
from sodetlib.detmap.meta_select import abs_path_detmap
from sodetlib.detmap.example.download_example_data import sample_data_init
from sodetlib.detmap.makemap import MapMaker
from sodetlib.detmap.batch import Manifest
if getuser() in {'chw3k5', 'cwheeler'}:
    # this only happens on Caleb's computers
    import matplotlib as mpl
    # an interactive backend to render the plots, allows for zooming/panning and other interactions
    mpl.use(backend='TkAgg')


if __name__ == '__main__':
    all_manifest = os.path.join(abs_path_detmap, 'example', 'manifest.csv')
    test_manifest = os.path.join(abs_path_detmap, 'example', 'dev_manifest.csv')
    manifest = Manifest(path=all_manifest, plot_pdf=False)
