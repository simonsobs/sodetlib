"""
A Developmental Environment for Testing New features.
"""
from getpass import getuser
from sodetlib.detmap.example.download_example_data import sample_data_init
from sodetlib.detmap.makemap import MapMaker
if getuser() in {'chw3k5', 'cwheeler'}:
    # this only happens on Caleb's computers
    import matplotlib as mpl
    # an interactive backend to render the plots, allows for zooming/panning and other interactions
    mpl.use(backend='TkAgg')


sample_data_init(del_dir=False)
