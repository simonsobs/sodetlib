from sotodlib.io import load_smurf
from sodetlib.util import Registers
from sodetlib.operations import uxm_setup
import os
import time

try:
    from pysmurf.client.util.pub import set_action
except:
    set_action = lambda : (lambda f : f)


def get_session_files(stream_id, session_id, idx=None,
                      base_dir='/data/so/timestreams'):
    """
    Gets a list of all files on the system corresponding to a given streaming
    session.

    Args
    ----
    stream_id : str
        stream_id for the stream you wish to load. Often this will be in
        cfg.stream_id
    session_id : int
        Session id corresonding with the stream session you wish to load.
        This is what is returned by the stream-taking functions.
    idx : int, list(int), optional
        Index of file you wish to load. Long streams are chunked into 10-minute
        files, and this parameter can used to isolate a smaller part of a long
        stream.
    base_dir : str
        Base directory where timestreams are stored. Defaults to
        /data/so/timestreams.
    """
    subdir = os.path.join(base_dir, str(session_id)[:5], stream_id)
    files = sorted([
        os.path.join(subdir, f) for f in os.listdir(subdir)
        if str(session_id) in f
    ])

    if idx is None:
        return files
    elif isinstance(idx, int):
        return files[idx]
    else:  # list of indexes
        return [files[i] for i in idx]

def load_session_status(stream_id, session_id, base_dir='/data/so/timestreams'):
    """
    Gets SmurfStatus object for a given stream.

    Args
    ----
    stream_id : str
        stream_id for the stream you wish to load. Often this will be in
        cfg.stream_id
    session_id : int
        Session id corresonding with the stream session you wish to load.
        This is what is returned by the stream-taking functions.
    base_dir : str
        Base directory where timestreams are stored. Defaults to
        /data/so/timestreams.
    """
    files = get_session_files(stream_id, session_id, base_dir=base_dir)
    if len(files) == 0:
        raise FileNotFoundError(
            f"Could not find files for {(stream_id, session_id)}"
        )
    return load_smurf.SmurfStatus.from_file(files[0])


def load_session(stream_id, session_id, idx=None,
                 base_dir='/data/so/timestreams', show_pb=False, **kwargs):
    """
    Loads a stream-session into an axis manager. Any additional keyword
    arguments will be passed to the ``load_smurf.load_file`` function.

    Args
    ----
    stream_id : str
        stream_id for the stream you wish to load. Often this will be in
        cfg.stream_id
    session_id : int
        Session id corresonding with the stream session you wish to load.
        This is what is returned by the stream-taking functions.
    idx : int, list(int), optional
        Index of file you wish to load. Long streams are chunked into 10-minute
        files, and this parameter can used to isolate a smaller part of a long
        stream.
    base_dir : str
        Base directory where timestreams are stored. Defaults to
        /data/so/timestreams.
    """
    files = get_session_files(stream_id, session_id, idx=idx,
                              base_dir=base_dir)
    if len(files) == 0:
        raise FileNotFoundError(
            f"Could not find files for {(stream_id, session_id)}"
        )
    am =  load_smurf.load_file(files, show_pb=show_pb, **kwargs)

    if 'ch_info' not in am._fields:
        am.wrap('ch_info', am.det_info.smurf)

    return am


@set_action()
def take_g3_data(S, dur, **stream_kw):
    """
    Takes data for some duration

    Args
    ----
    S : SmurfControl
        Pysmurf control object
    dur : float
        Duration to take data over

    Returns
    -------
    session_id : int
        Id used to read back stream data
    """
    stream_g3_on(S, **stream_kw)
    time.sleep(dur)
    return stream_g3_off(S, emulator=stream_kw.get('emulator', False))


@set_action()
def stream_g3_on(S, make_freq_mask=False, emulator=False, tag=None,
                 channel_mask=None, filter_wait_time=2, make_datfile=False,
                 downsample_factor=None, downsample_mode=None,
                 filter_disable=False, filter_order=None, filter_cutoff=None,
                 stream_type=None, subtype='stream', enable_compression=None):
    """
    Starts the G3 data-stream. Returns the session-id corresponding with the
    data stream.

    Args
    ----
    S : S
        Pysmurf control object
    make_freq_mask : bool, optional
        Tell pysmurf to write and register the current freq mask
    emulator : bool
        If True, will enable the emulator source data-generator. Defaults to
        False.
    tag : optional(string)
        If set, this will attach the tag to the g3 stream
    channel_mask : optional(list)
        List of channels to stream. If None, this will stream all channels with
        a probe tone
    filter_wait_time : float
        Seconds to wait after resetting the filter before writing data to disk
    make_datfile bool
        If True, will create a datfile
    downsample_factor : optional(int)
        Downsample factor. If None, this will be pulled from the device
        cfg.
    downsample_mode : optional(string)
        Downsample mode, can either be 'internal' or 'external'. If not set,
        this will be pulled from the device cfg.
    filter_disable : bool
        If true, will disable the downsample filter before streaming.
    filter_order : int
        Order of the downsample filter. Read from device config by default.
    filter_cutoff : float
        The cutoff frequency in Hz for the downsample filter.
        Read from device config by default.
    stream_type : optional(string)
        Type of stream. This should either be "obs" or "oper". If None,
        it will be determined based off of the subtype. ('stream', 'cmb', and
        'cal', will automatically be set to 'obs' type)
    subtype : optional(string)
        Stream subtype. This defaults to 'stream'.
    enable_compression : optional(bool)
        If true, will tell the smurf-streamer to compress data.

    Return
    -------
    session_id : int
        Id used to read back streamed data
    """
    cfg = S._sodetlib_cfg
    if stream_type is None:
        if subtype in ['stream', 'cmb', 'cal']:
            stream_type = 'obs'
        else:
            stream_type = 'oper'

    if enable_compression is None:
        enable_compression = cfg.dev.exp.get('enable_compression', True)

    tags = f'{stream_type},{subtype}'
    if tag is not None:
        tags += ',' + tag

    reg = Registers(S)

    reg.pysmurf_action.set(S.pub._action)
    reg.pysmurf_action_timestamp.set(S.pub._action_ts)
    reg.stream_tag.set(tags)

    reg.enable_compression.set(enable_compression)

    if downsample_mode is None:
        downsample_mode = cfg.dev.exp['downsample_mode']
    if downsample_factor is None:
        downsample_factor = cfg.dev.exp['downsample_factor']
    if filter_order is None:
        filter_order = cfg.dev.exp["downsample_filter"]["order"]
    if filter_cutoff is None:
        filter_cutoff = cfg.dev.exp["downsample_filter"]["cutoff_freq"]

    S.set_downsample_mode(downsample_mode)
    S.set_downsample_factor(downsample_factor)
    S.set_downsample_filter(filter_order, filter_cutoff)
    S.set_filter_disable(int(filter_disable))

    # ensure fixed tones are on
    uxm_setup.turn_on_fixed_tones(S, cfg)

    S.stream_data_on(make_freq_mask=make_freq_mask, channel_mask=channel_mask,
                     filter_wait_time=filter_wait_time, make_datafile=make_datfile)

    if emulator:
        reg.source_enable.set(1)
        S.set_stream_enable(1)

    reg.open_g3stream.set(1)

    # Sometimes it takes a bit for data to propogate through to the
    # streamer
    for _ in range(10):
        sess_id = reg.g3_session_id.get()
        if sess_id != 0:
            break
        time.sleep(0.5)
    S.log(f"Session id: {sess_id}")
    return sess_id


@set_action()
def stream_g3_off(S, emulator=False):
    """
    Stops the G3 data-stream. Returns the session-id corresponding with the
    data stream.

    Args
    ----
    S : S
        Pysmurf control object
    emulator : bool
        If True, will enable the emulator source data-generator. Defaults to
        False.

    Return
    -------
    session_id : int
        Id used to read back streamed data
    """
    reg = Registers(S)
    sess_id = reg.g3_session_id.get()
    if sess_id == 0:
        S.log("Session-id returned 0! Will try to obtain from file path")
        fpath = reg.g3_filepath.get()
        try:
            sess_id = int(os.path.basename(fpath).split('_')[0])
        except Exception:
            S.log("Could not extract session id from filepath! Setting to 0")
            sess_id = 0

    if emulator:
        reg.source_enable.set(0)

    S.set_stream_enable(0)
    S.stream_data_off()

    reg.open_g3stream.set(0)
    reg.pysmurf_action.set('')
    reg.pysmurf_action_timestamp.set(0)
    reg.stream_tag.set('')

    # Waits until file is closed out before returning
    S.log("Waiting for g3 file to close out")
    while reg.g3_session_id.get() != 0:
        time.sleep(0.5)

    return sess_id

