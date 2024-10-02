"""
Script for deleting old data on the smurf-server.

Configuration parameters can be set via a config file:
```
python3 clear_old_data.py --config-file clear_data_cfg.yaml
```
where the entries in the config file map onto the Config class below.

Specific configuration settings can be set directly from the command line, such as:
```
python3 clear_old_data.py --dry
```
"""
from dataclasses import dataclass
import os
import datetime
from typing import List, Optional, Dict, Any
from enum import Enum, auto
import yaml
import shutil
import argparse
import logging
from copy import deepcopy

logger = logging.getLogger()

_nameToLogLevel = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARN': logging.WARNING,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL,
}

@dataclass
class Config:
    """
    Configuration object to control the behavior of the data deletion script.

    Args
    -----
    dry: bool
        If true, will do a dry-run of the data-deletion without deleting any
        files. Logs will be printed for all of the files that would be deleted.
    verbose: bool
        If true, logs will be more verbose.
    delete_smurf_data_after_days: int
        Days after which smurf data will be deleted.
    delete_timestream_data_after_days: int
        Days after which timestream data will be deleted.
    delete_core_dumps_after_days: int
        Days after which core-dumps will be deleted.
    delete_logs_after_days: int
        Days after which log directories will be deleted.
    """
    dry: bool = False
    delete_smurf_data_after_days: int = 31
    delete_timestream_data_after_days: int = 365 // 2
    delete_core_dumps_after_days: int = 365
    delete_logs_after_days: int = 365 * 5
    log_level: int = logging.INFO
    log_file: Optional[str] = '/data/logs/clear_data.log'

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        _data = deepcopy(data)
        if 'log_level' in data:
            if isinstance(data['log_level'], str):
                _data['log_level'] = _nameToLogLevel[data['log_level'].upper()]
        return cls(**_data)


    @classmethod
    def from_yaml(cls, path) -> "Config":
        """
        Creates a Config object based on a yaml file. Key names in the file must
        match config dataclass fields.
        """
        with open(path, 'r') as f:
            return cls.from_dict(yaml.safe_load(f))

    @classmethod
    def from_args(cls, args_list: Optional[List[str]]=None) -> "Config":
        parser = argparse.ArgumentParser()
        parser.add_argument('--config-file', type=str, default=None)
        parser.add_argument('--dry', action='store_true')
        args = parser.parse_args(args_list)

        if args.config_file:
            cfg = cls.from_yaml(args.config_file)
        else:
            cfg = cls()

        if args.dry:
            cfg.dry = args.dry
        return cfg

def setup_logger(cfg: Config) -> None:
    logger.setLevel(cfg.log_level)
    if len(logger.handlers) > 0:
        logger.error("Logger has already been configured! Doing nothing")
        return

    formatter = logging.Formatter('%(asctime)s - %(levelname)s:  %(message)s')

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(cfg.log_level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if cfg.log_file is not None:
        file_handler = logging.FileHandler(cfg.log_file)
        file_handler.setLevel(cfg.log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


class FileType(Enum):
    SmurfData = auto()
    TimestreamData = auto()
    LogData = auto()
    CoreDump = auto()

@dataclass
class FileInfo:
    path: str
    dt: datetime.datetime
    file_type: FileType


def create_smurf_date_dir(cfg: Config, path: str) -> Optional[FileInfo]:
    dirname = os.path.split(path)[1]
    try:
        year = int(dirname[:4])
        month = int(dirname[4:6])
        day = int(dirname[6:])
        dt = datetime.datetime(year=year, month=month, day=day)
    except Exception:
        logger.debug(f"Could not parse datetime: {dirname}")
        return None
    return FileInfo(path=path, dt=dt, file_type=FileType.SmurfData)


def scan_smurf_data(cfg: Config) -> List[FileInfo]:
    date_dirs: List[FileInfo] = []
    base_dir = '/data/smurf_data'
    now = datetime.datetime.now()
    max_time_delta = datetime.timedelta(days=cfg.delete_smurf_data_after_days)
    for d in os.listdir(base_dir):
        result = create_smurf_date_dir(cfg, os.path.join(base_dir, d))
        if result is not None:
            if now - result.dt > max_time_delta:
                date_dirs.append(result)

    return sorted(date_dirs, key=lambda f:f.dt)


def scan_timestream_dirs(cfg: Config) -> List[FileInfo]:
    timestream_dirs: List[FileInfo] = []
    base_dir = '/data/so/timestreams'
    now = datetime.datetime.now()
    max_time_delta = datetime.timedelta(days=cfg.delete_timestream_data_after_days)
    for d in os.listdir(base_dir): # d is 5-digit ctime code
        path = os.path.join(base_dir, d)
        try:
            # Give one day buffer for timezone effects, etc.
            timestamp = (int(d) + 1)* 1e5
            dt = datetime.datetime.fromtimestamp(timestamp)
        except ValueError:
            logger.debug(f"Could not parse datetime: {path}")
            continue
        file = FileInfo(
            path=path,
            dt=dt,
            file_type=FileType.TimestreamData
        )
        if now - file.dt > max_time_delta:
            timestream_dirs.append(file)
    return sorted(timestream_dirs, key=lambda f:f.dt)

def scan_core_dumps(cfg: Config) -> List[FileInfo]:
    core_dump_dir = '/data/cores'
    files: List[FileInfo] = []
    now = datetime.datetime.now()
    max_time_delta = datetime.timedelta(
        days=cfg.delete_core_dumps_after_days
    )
    for f in os.listdir(core_dump_dir):
        path = os.path.join(core_dump_dir, f)
        ts = os.path.getctime(path)
        dt = datetime.datetime.fromtimestamp(ts)
        if now - dt > max_time_delta:
            files.append(FileInfo(
                path=path, dt=dt, file_type=FileType.CoreDump
            ))
    return files

def scan_log_dirs(cfg: Config) -> List[FileInfo]:
    log_dir = '/data/logs'
    files: List[FileInfo] = []
    now = datetime.datetime.now()
    max_time_delta = datetime.timedelta(
        days=cfg.delete_logs_after_days
    )
    for f in os.listdir(log_dir):
        path = os.path.join(log_dir, f)
        ts = os.path.getmtime(path)
        dt = datetime.datetime.fromtimestamp(ts)
        if now - dt > max_time_delta:
            files.append(FileInfo(
                path=path, dt=dt, file_type=FileType.CoreDump
            ))
    return files


def remove_file(cfg: Config, file: FileInfo):
    if cfg.dry:
        if os.path.isdir(file.path):
            logger.info(f"dry mode: rm -rf {file.path}")
        else:
            logger.info(f"rm {file.path}")
    else:
        if os.path.isdir(file.path):
            shutil.rmtree(file.path)
        else:
            os.remove(file.path)


def main(cfg: Config) -> None:
    setup_logger(cfg)
    logger.info('-'*80)

    logger.info(cfg)
    now = datetime.datetime.now()
    files_to_delete: List[FileInfo] = []
    files_to_delete += scan_smurf_data(cfg)
    files_to_delete += scan_timestream_dirs(cfg)
    files_to_delete += scan_core_dumps(cfg)
    files_to_delete += scan_log_dirs(cfg)

    logger.info(f"{len(files_to_delete)} files to delete:")
    for f in files_to_delete:
        days_old = (now - f.dt).days
        logger.info(f' - {f.path} ({days_old} days old)')

    if len(files_to_delete) == 0:
        logger.info("No files to delete")
        return

    resp = input(f"Proceed with deletion (dry={cfg.dry})? [y/n]  ")
    if resp.strip().lower() != 'y':
        logger.info("Not proceed with deletion")
        return

    logger.info("Deleting files")
    for f in files_to_delete:
        remove_file(cfg, f)

if __name__ == '__main__':
    cfg = Config.from_args()
    main(cfg)

