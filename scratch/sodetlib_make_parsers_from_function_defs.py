import re

'''
Rita Sonka
3/27/2020
For Simons Observatory
'''


def make_parsers(argstr, docstr):
    """ given strings of function arguments (argstr) and proper pandas
    docstrings for those arguments (docstr), prints the argstr with all the
    default values replaced by the names of their argument [for passing to a
    main or other function], and then prints all the "parser.add_argument("
    lines that it can manage from them.
    Can't currently handle array arguments. You do that with nargs.
    (See documentation: https://docs.python.org/3/library/argparse.html)
    See examples below this function definition.
    Please check your results--that's still much faster than typing by hand!

    Parameters
    ----------
    argstr : str
        A string copy of the function's argument list. Arguments must be
        separated by ', '. You must follow the last argument with ', ' for it
        to catch all arguments. There should be no spaces around equal sign in
        any optional arguments.
    docstr : str
        A string copy of the function's parameters in its proper pandas
        docstring, like the one you are reading right now. Right now requires
        exactly 4 spaces before the start of an argument line, exactly 8 spaces
        before the start of a 'help' line.
    """
    # Remember to type ', ' after end of last argument in argstr!
    # Remember docstr must follwo the pandas layout guidelines!
    mandatory = []
    args = argstr.replace("\n"," ").split(", ")
    for arg in args:
        result = re.search("\w+",arg)
        if result:
            mandatory.append(result.group(0))
    default = {}
    result = re.search(r"(\w+?)(=(\S+?)),([ |\n])", argstr)
    while result:
        #print(result.group(0))
        default[result.group(1)]=result.group(3)
        argstr = argstr.replace(result.group(0),result.group(1)+"++++"+
                                result.group(1)+","+result.group(4))
        #print(argstr)
        result = re.search(r"(\w+?)(=(\S+?)),([ |\n])", argstr)
    print(argstr.replace("++++","=")) # Might as well get that, too
    # Now, get type and help
    type = {}
    help = {}
    result = re.search("    (\w*) : (\S*)((\n        .*)+)", docstr)
    while result:
        type[result.group(1)]=result.group(2)
        if result.group(2) not in ["int","str","float","bool"]:
            print("WARNING! arg %s of type %s" % (result.group(1), result.group(2)))
        help[result.group(1)]=' '.join(result.group(3).split())
        docstr=docstr.replace(result.group(0),'')
        result = re.search("    (\w*) : (\S*)((\n        .*)+)", docstr)

    #print(default)
    #print(type)
    for arg in mandatory:
        print('    parser.add_argument("--' + arg.replace("_","-") + '", type=' +
            type[arg] + ', required=True' + ',\n        help="'+
            help[arg] +'")')
    for arg in default:
        print('    parser.add_argument("--' + arg.replace("_","-") + '", type=' +
            type[arg] + ', default=' + default[arg] + ',\n        help="'+
            help[arg] +'")')


# Example, using the function find_freq in pysmurf/client/tune/smurf_tune.py

find_freq_args = """band, subband=np.arange(13,115), drive_power=None,
            n_read=2, make_plot=False, save_plot=True, plotname_append='',
            window=50, rolling_med=True, make_subband_plot=False,
            show_plot=False, grad_cut=.05, amp_cut=.25, pad=2, min_gap=2, """

# a useful string for getting below docstring from find_freq's docstring
# (though beware original find_freq didn't document all its arguments, and
# didn't note which ones were really arrays originally!):
# print(string.replace(" :",":").replace("): ", "\n            ").replace(" ("," : "))
find_freq_docstring = """    band : int
        band to find tuned frequencies on. In range [0,7].
    subband : [int]
        An int array for the subbands
    drive_power : int
        The drive amplitude.  If none given, takes from cfg.
    n_read : int
        The number sweeps to do per subband
    make_plot : bool
        make the plot frequency sweep. Default False.
    save_plot : bool
        save the plot. Default True.
    plotname_append : str
        Appended to the default plot filename. Default ''.
    rolling_med : bool
        Whether to iterate on a rolling median or just the median of the whole
        sample.
    window : int
        The width of the rolling median window
    make_subband_plot : bool
        Make subband plots? Default False.
    show_plot : bool
        Show plots as they are made? Default False.
    pad : int
        number of samples to pad on either side of a resonance search
        window
    min_gap : int
        minimum number of samples between resonances
    grad_cut : float
        The value of the gradient of phase to look for
        resonances. Default is .05
    amp_cut : float
        The fractional distance from the median value to decide
        whether there is a resonance. Default is .25."""



make_parsers(find_freq_args,find_freq_docstring)
# The above prints the following string to terminal when the file is run:
"""
band, subband=subband, drive_power=drive_power,
            n_read=n_read, make_plot=make_plot, save_plot=save_plot, plotname_append=plotname_append,
            window=window, rolling_med=rolling_med, make_subband_plot=make_subband_plot,
            show_plot=show_plot, grad_cut=grad_cut, amp_cut=amp_cut, pad=pad, min_gap=min_gap,
WARNING! arg subband of type [int]
    parser.add_argument("--band", type=int, required=True,
        help="band to find tuned frequencies on. In range [0,7].")
    parser.add_argument("--subband", type=[int], required=True,
        help="An int array for the subbands")
    parser.add_argument("--drive-power", type=int, required=True,
        help="The drive amplitude. If none given, takes from cfg.")
    parser.add_argument("--n-read", type=int, required=True,
        help="The number sweeps to do per subband")
    parser.add_argument("--make-plot", type=bool, required=True,
        help="make the plot frequency sweep. Default False.")
    parser.add_argument("--save-plot", type=bool, required=True,
        help="save the plot. Default True.")
    parser.add_argument("--plotname-append", type=str, required=True,
        help="Appended to the default plot filename. Default ''.")
    parser.add_argument("--window", type=int, required=True,
        help="The width of the rolling median window")
    parser.add_argument("--rolling-med", type=bool, required=True,
        help="Whether to iterate on a rolling median or just the median of the whole sample.")
    parser.add_argument("--make-subband-plot", type=bool, required=True,
        help="Make subband plots? Default False.")
    parser.add_argument("--show-plot", type=bool, required=True,
        help="Show plots as they are made? Default False.")
    parser.add_argument("--grad-cut", type=float, required=True,
        help="The value of the gradient of phase to look for resonances. Default is .05")
    parser.add_argument("--amp-cut", type=float, required=True,
        help="The fractional distance from the median value to decide whether there is a resonance. Default is .25.")
    parser.add_argument("--pad", type=int, required=True,
        help="number of samples to pad on either side of a resonance search window")
    parser.add_argument("--min-gap", type=int, required=True,
        help="minimum number of samples between resonances")
    parser.add_argument("--subband", type=[int], default=np.arange(13,115),
        help="An int array for the subbands")
    parser.add_argument("--drive-power", type=int, default=None,
        help="The drive amplitude. If none given, takes from cfg.")
    parser.add_argument("--n-read", type=int, default=2,
        help="The number sweeps to do per subband")
    parser.add_argument("--make-plot", type=bool, default=False,
        help="make the plot frequency sweep. Default False.")
    parser.add_argument("--save-plot", type=bool, default=True,
        help="save the plot. Default True.")
    parser.add_argument("--plotname-append", type=str, default='',
        help="Appended to the default plot filename. Default ''.")
    parser.add_argument("--window", type=int, default=50,
        help="The width of the rolling median window")
    parser.add_argument("--rolling-med", type=bool, default=True,
        help="Whether to iterate on a rolling median or just the median of the whole sample.")
    parser.add_argument("--make-subband-plot", type=bool, default=False,
        help="Make subband plots? Default False.")
    parser.add_argument("--show-plot", type=bool, default=False,
        help="Show plots as they are made? Default False.")
    parser.add_argument("--grad-cut", type=float, default=.05,
        help="The value of the gradient of phase to look for resonances. Default is .05")
    parser.add_argument("--amp-cut", type=float, default=.25,
        help="The fractional distance from the median value to decide whether there is a resonance. Default is .25.")
    parser.add_argument("--pad", type=int, default=2,
        help="number of samples to pad on either side of a resonance search window")
    parser.add_argument("--min-gap", type=int, default=2,
        help="minimum number of samples between resonances")
"""


# Other examples I did.
docstring_missed_args = """delta_freq=None,
                      lock_max_derivative=False, """
docstring_missed_docstring = """
    delta_freq : float
        The frequency offset at which to measure
        the complex transmission to compute the eta parameters.
        Passed to eta_estimator.  Units are MHz.  If none supplied
        as an argument, takes value in config file.
    lock_max_derivative : bool
        I'm not sure what this is, but it was in setup_notches when I wrote the
        OCS script (though with no documentation.) Defaults to False."""

docstring_missed_args = """ make_subband_plot=False,
show_plot=False, """
docstring_missed_docstring = """    make_subband_plot : bool
        Make subband plots? Default False.
    show_plot : bool
        Show plots as they are made? Default False.
        """

run_serial_gradient_descent_args = """sync_group=True,
                                    timeout=240, """
run_serial_gradient_descent_docstring = """
    sync_group : bool
        Whether to use the sync group to monitor
        the PV. Defauult is True.
    timeout : float
        The maximum amount of time to wait for the PV."""

setup_notches_args = """ resonance=None,
            sweep_width=.3, df_sweep=.002, min_offset=0.1,
            delta_freq=None, new_master_assignment=False,
            lock_max_derivative=False, """

setup_notches_docstring = """    resonance : floatarray
        A 2 dimensional array with resonance
        frequencies and the subband they are in. If given, this will take
        precedent over the one in self.freq_resp.
    sweep_width : float
        The range to scan around the input resonance in
        units of MHz. Default .3
    df_sweep : float
        The sweep step size in MHz. Default .005
    min_offset : float
        Minimum distance in MHz between two resonators for assigning channels.
    delta_freq : float
        The frequency offset at which to measure
        the complex transmission to compute the eta parameters.
        Passed to eta_estimator.  Units are MHz.  If none supplied
        as an argument, takes value in config file.
    new_master_assignment : bool
        Whether to create a new master assignment
        file. This file defines the mapping between resonator frequency
        and channel number."""





def echo_optional_args(argstr):
    result = re.search(r" (.+?)(=(.+?)),[ |\n]", argstr)
    #print(result.group(0))
    #print(result.group(1))
    #print(result.group(2))
    #print(result.group(3))
    while result:
        argstr = argstr.replace(result.group(2),"++++"+result.group(1))
        result = re.search(r" (\w+?)(=(.+?)),[ |\n]", argstr)
        #print(argstr)
    print(argstr.replace("++++","="))
    #print(re.sub(r' (.+?)=(.+?),[ |\n]', r'\g<2>\g<1>',argstr))
