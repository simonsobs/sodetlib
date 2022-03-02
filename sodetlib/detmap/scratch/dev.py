"""
A Developmental Environment for Testing New features.
"""
import os
from getpass import getuser
from sodetlib.detmap.simple_csv import manifest_parse
from sodetlib.detmap.meta_select import abs_path_detmap
from sodetlib.detmap.example.download_example_data import sample_data_init
from sodetlib.detmap.makemap import MapMaker
if getuser() in {'chw3k5', 'cwheeler'}:
    # this only happens on Caleb's computers
    import matplotlib as mpl
    # an interactive backend to render the plots, allows for zooming/panning and other interactions
    mpl.use(backend='TkAgg')


sample_data_init(del_dir=False)
manifest = manifest_parse(path=os.path.join(abs_path_detmap, 'example', 'manifest.csv'))
sample_data_init(del_dir=True, zip_file_id='1BugpuMsoKtlaxqagIt2d0uaQQhWcMb_V', folder_name='tunes')
