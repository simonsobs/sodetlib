"""
Sudo code from a meeting with Caleb and Daniel about how a potential channel class would gather information.
"""
channel_meta_data = some_function_to_read_static_data()


class Channel:
    def __init__(self, band, slot, channel_num, frequecy):
        self.band = band
        self.slot = slot
        self.channel_num = channel_num
        self.frequecy = frequecy

        self.bais = self.bias_method()
        self.dector_x_pos, self.dector_y_pos = channel_meta_data[self.bais, channel_num]

    def bias_method(self):
        return self.band


channel_set = [Channel(**kwargs) for kwargs in channel_kwargs]