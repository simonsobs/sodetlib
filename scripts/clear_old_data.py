from dataclasses import dataclass
import os
import datetime
from typing import List, Optional
from enum import Enum, auto
import yaml
import shutil
import argparse


@dataclass
class Config:
    dry: bool = True
    delete_smurf_data_after_days: int = 31
    delete_timestream_data_after_days: int = 365 // 2
    delete_core_dumps_after_days: int = 365
    delete_logs_after_days: int = 365
    verbose: bool = True

    @classmethod
    def from_yaml(cls, path) -> "Config":
        with open(path, 'r') as f:
            return cls(**yaml.safe_load(f))

    @classmethod
    def from_args(cls, args_list: Optional[List[str]]=None) -> "Config":
        parser = argparse.ArgumentParser()
        parser.add_argument('--dry', action='store_true')
        parser.add_argument('--verbose', action='store_true')
        args = parser.parse_args(args_list)
        return cls(
            verbose=args.verbose,
            dry=args.dry
        )


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
        if cfg.verbose:
            print(f"Could not parse datetime: {dirname}")
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
            if cfg.verbose:
                print(f"Could not parse datetime: {path}")
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
            print(f"dry mode: rm -rf {file.path}")
        else:
            print(f"rm {file.path}")
    else:
        if os.path.isdir(file.path):
            shutil.rmtree(file.path)
        else:
            os.remove(file.path)


def main(cfg: Config) -> None:
    now = datetime.datetime.now()
    files_to_delete: List[FileInfo] = []
    files_to_delete += scan_smurf_data(cfg)
    files_to_delete += scan_timestream_dirs(cfg)
    files_to_delete += scan_core_dumps(cfg)
    files_to_delete += scan_log_dirs(cfg)

    print(f"{len(files_to_delete)} files to delete:")
    for f in files_to_delete:
        days_old = (now - f.dt).days
        print(f' - {f.path} ({days_old} days old)')

    if len(files_to_delete) == 0:
        print("No files to delete")
        return

    resp = input("Proceed with deletion? [y/n]  ")
    if resp.strip().lower() != 'y':
        print("Not proceed with deletion")
        return

    print("Deleting files")
    for f in files_to_delete:
        remove_file(cfg, f)

if __name__ == '__main__':
    cfg = Config.from_args()
    main(cfg)

