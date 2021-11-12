"""
Tools and classes for loading SMuRF controllers.
"""

from sodetlib.det_config import DetConfig


class LoadS:
    """
    For obtaining and initializing and abstract number of SMuRF controllers (identified by slot number)
    for testing, commonly this is denoted as python objected denoted as an uppercase 'S'.

    For a single S, and cfg use:

        slot_num = 1
        load_s = LoadS(slot_nums={slot_num}, verbose=verbose)
        cfg = load_s.S_dict[slot_num]
        S = load_s.S_dict[slot_num]

    For multiple S and cfg values use:

        slot_nums = [2, 0, 4]  # just showing order is not important.
        load_s = LoadS(slot_nums=slot_nums, verbose=verbose)
        for slot_num in sorted(load_s.slot_nums):
            cfg = load_s.S_dict[slot_num]
            S = load_s.S_dict[slot_num]
    """
    def __init__(self, slot_nums=None, verbose=True):
        """
        Proved with a slot numbers, an instance of this class will automatically load and configure
        each SMuRF slot. The smurf slots are accessible through the instance variables self.cfg_dict,
        self.S_dict, and self.logs_dict. As the names of these variables suggest, each variable is a
        dictionary with the values of the slot numbers as keys to S, cfg, log, for a single SMuRF slot.

        :param slot_nums: single value or iterable of slot number (or numbers) to load and control.
        :param verbose: bool - Toggles print() statements for the actions within the methods methods of this
                               class
        """
        # record the information used across the methods in this class
        self.verbose = verbose
        if slot_nums is None:
            self.slot_nums = set()
        else:
            self.slot_nums = {int(slot_num) for slot_num in slot_nums}

        # initialize the controller storage variables
        self.cfg_dict = {}
        self.S_dict = {}
        self.logs_dict = {}

        # load the controllers here if
        for slot_num in sorted(self.slot_nums):
            self.load_single_slot(slot_num=slot_num)

    def load_single_slot(self, slot_num):
        """
        This loads the controller for a single SMuRF slot:

        This method populates the instance variables of self.cfg_dict, self.S_dict, and self.logs_dict
        using 'slot_num' as the key.

        :param slot_num: int - The SMuRF slot number.
        """
        slot_num = int(slot_num)
        # this may already by in self.slot_nums, but we add it here to capture slot_nums specified outside of __init__
        self.slot_nums.add(slot_num)
        if self.verbose:
            print(f'Getting SMuRF control for slot: {slot_num}')
        # get control
        self.cfg_dict[slot_num] = DetConfig()
        # load config files
        self.cfg_dict[slot_num].load_config_files(slot=slot_num)
        if self.verbose:
            print(DetConfig)
        # get the controller with the configuration
        self.S_dict[slot_num] = self.cfg_dict[slot_num].get_smurf_control(make_logfile=True)
        if self.verbose:
            print(f'  log file found at {self.S_dict[slot_num].logfile}')
            print(f'  plotting directory at {self.S_dict[slot_num].plotdir}')
        # automatically configure the controllers after acquiring them
        self.configure_single_slot(slot_num=slot_num)

    def configure_single_slot(self, slot_num, set_downsample_factor=20):
        """

        :param slot_num: int - The SMuRF slot number.
        :param set_downsample_factor: float - The SMuRF downsampling factor.
        """
        self.S_dict[slot_num].all_off()
        self.S_dict[slot_num].set_rtm_arb_waveform_enable(0)
        self.S_dict[slot_num].set_filter_disable(0)
        self.S_dict[slot_num].set_downsample_factor(set_downsample_factor)
        self.S_dict[slot_num].set_mode_dc()


if  __name__ == "__main__":
    pass