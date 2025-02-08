#!/usr/bin/env python
"""\
This script will maintain a local copy of your uploads to vimeo.com using the official PyVimeo library.
"""

import sys, os, os.path as op, re, threading, functools, datetime, runpy, time
import urllib.request, urllib.parse, json, shutil
from tqdm import tqdm
from contextlib import contextmanager
import dateutil.parser

import ssl
import certifi
# 建立 SSL 上下文，使用 certifi 提供的根憑證
ssl_context = ssl.create_default_context(cafile=certifi.where())

# 使用官方 PyVimeo
from vimeo import VimeoClient

# 以下 t4 模組若無法取得，請自行實作簡易版本：
# 請建立 t4/title_to_id.py 與 t4/typography.py 參考示例：
# --- t4/title_to_id.py ---
# import re
# def safe_filename(name):
#     return re.sub(r'[\\/:"*?<>|]+', "", name)
#
# --- t4/typography.py ---
# def pretty_bytes(num):
#     for unit in ['B','KB','MB','GB','TB']:
#         if num < 1024.0:
#             return f"{num:.1f}{unit}"
#         num /= 1024.0
#     return f"{num:.1f}PB"

from t4.title_to_id import safe_filename
from t4.typography import pretty_bytes

# 若環境中出現 collections.MutableMapping 問題（Python 3.10+），則補上：
import collections
import collections.abc
if not hasattr(collections, 'MutableMapping'):
    collections.MutableMapping = collections.abc.MutableMapping

class DownloadLocked(Exception):
    pass

class NoOriginalDownloadError(Exception):
    pass

def parse_timestamp(s):
    """Return a datetime object for vimeo timestamp `s`."""
    return dateutil.parser.parse(s)

def original_ctime(metadata):
    """Lookup the original download’s create_time field, parse it, and return as timestamp."""
    for d in metadata["download"]:
        if d["quality"] == "source":
            return parse_timestamp(d["created_time"]).timestamp()
    else:
        video_id = metadata.get("uri", "unknown").split("/")[-1]
        raise NoOriginalDownloadError(
            "No original download available for %s “%s”" % (video_id, metadata["name"]))

def report_progress_on(item, done, extra=None, metainfo=None):
    # 進度回報，可自行擴充
    pass

def report(*things, end="\n"):
    print(*things, end=end, file=sys.stderr)

class ArchiveDir(object):
    def __init__(self, backup, dirname):
        self.backup = backup
        self.dirname = dirname
        match = re.match(r"(\d+).*", dirname)
        if match is None:
            raise IOError("Not an archive dir: " + dirname)
        else:
            self.vimeo_id = int(match.group(1))
        self.abspath = op.join(backup.local_root, dirname)

    @classmethod
    def download(cls, backup, metadata_json):
        metadata = json.loads(metadata_json)
        uri = metadata["uri"]
        rest, vimeo_id = uri.rsplit("/", 1)
        vimeo_id = int(vimeo_id)

        original_download = None
        for d in metadata["download"]:
            if d["quality"] == "source":
                original_download = d
                break
        else:
            raise NoOriginalDownloadError(
                "No original download available for %i “%s”" % (vimeo_id, metadata["name"]))

        url = original_download["link"]
        size = original_download["size"]
        # 印出影片資訊與分隔線
        print("\n-----")
        print(f"下載影片 ID: {vimeo_id}  Title: {metadata['name']}")
        print(f"檔案大小: {pretty_bytes(size)}")
        print(f"下載網址: {url}")

        fields = urllib.parse.urlparse(url)
        path = urllib.parse.unquote(fields.path)
        parts = path.split("/")
        filename = safe_filename(parts[-1])
        fn, ext = op.splitext(filename)
        title = safe_filename(metadata["name"])
        filename = f"{title}{ext}"
        dirname = f"{vimeo_id} {title}"

        tmppath = op.join(backup.local_root, "tmp." + dirname)
        os.mkdir(tmppath)

        try:
            with open(op.join(tmppath, "metadata.json"), "w", encoding="utf-8") as fp:
                fp.write(metadata_json)

            outpath = op.join(tmppath, filename)
            print(f"開始下載 {filename} ...")
            start_time = time.time()
            with urllib.request.urlopen(url, context=ssl_context) as infp, open(outpath, "wb") as outfp:
                blocksize = 102400
                # 使用 tqdm 進度條，設定更新間隔為 1 秒
                with tqdm(total=size, unit='B', unit_scale=True, desc=filename, mininterval=1) as pbar:
                    while True:
                        bytes_chunk = infp.read(blocksize)
                        if not bytes_chunk:
                            break
                        outfp.write(bytes_chunk)
                        pbar.update(len(bytes_chunk))
            end_time = time.time()
            elapsed = end_time - start_time
            avg_speed = size / elapsed if elapsed > 0 else 0
            print(f"下載完成 {filename}: 用時 {elapsed:.1f} 秒，平均速度 {pretty_bytes(avg_speed)}/s")
            if os.path.getsize(outpath) != size:
                try:
                    os.unlink(outpath)
                except IOError:
                    pass
                raise IOError("下載失敗: " + url)
        except (Exception, KeyboardInterrupt) as e:
            shutil.rmtree(tmppath, ignore_errors=True)
            raise

        target_path = op.join(backup.local_root, dirname)
        if op.exists(target_path):
            archive_path = op.join(backup.local_root, "Archive")
            if not op.exists(archive_path):
                os.mkdir(archive_path)
            idx = 1
            while True:
                backup_path = op.join(archive_path, f"{dirname}.{idx}")
                if op.exists(backup_path):
                    idx += 1
                else:
                    break
            os.rename(target_path, backup_path)
        os.rename(tmppath, target_path)
        timestamp = original_ctime(metadata)
        os.utime(target_path, (timestamp, timestamp))
        return ArchiveDir(backup, dirname)

    def abspath_of(self, filename):
        return op.join(self.abspath, filename)

    @functools.cached_property
    def media_file_path(self):
        for fn in os.listdir(self.abspath):
            abspath = op.join(self.abspath, fn)
            if fn != "metadata.json" and op.isfile(abspath):
                return abspath
        raise IOError("No media file downloaded for " + repr(self.dirname))

    @functools.cached_property
    def mtime(self):
        return op.getmtime(self.abspath)

    @functools.cached_property
    def metadata(self):
        path = self.abspath_of("metadata.json")
        with open(path, encoding="utf-8") as fp:
            return json.load(fp)

    def delete(self):
        shutil.rmtree(self.abspath)
        self.backup.invalidate_archive_dir(self.vimeo_id)

class VimeoBackup(object):
    def __init__(self, local_root, vimeo_credentials):
        self.local_root = local_root
        self.vimeo_credentials = vimeo_credentials
        self._vimeo_connection = None
        self._connection_reset_timer = None
        self._archive_dirs = {}
        for dirname in os.listdir(self.local_root):
            full_path = op.join(self.local_root, dirname)
            if op.isdir(full_path):
                try:
                    ad = ArchiveDir(self, dirname)
                except IOError:
                    pass
                else:
                    self._archive_dirs[ad.vimeo_id] = ad

    @classmethod
    def from_config_file(cls, config_file_path):
        config = runpy.run_path(config_file_path)
        return VimeoBackup(config["local_root"], config)

    def invalidate_archive_dir(self, vimeo_id):
        try:
            del self._archive_dirs[vimeo_id]
        except KeyError:
            pass

    @property
    def vimeo_connection(self):
        if self._connection_reset_timer is not None:
            self._connection_reset_timer.cancel()
        if self._vimeo_connection is None:
            self._vimeo_connection = VimeoClient(
                token=self.vimeo_credentials["access_token"],
                key=self.vimeo_credentials["client_id"],
                secret=self.vimeo_credentials["client_secret"])
        def reset_connection():
            self._connection_reset_timer = None
            self._vimeo_connection = None
        self._connection_reset_timer = threading.Timer(180, reset_connection)
        return self._vimeo_connection

    def download_metadata_json(self, vimeo_id: int):
        # 印出分隔線與換行
        print("\n\n----------------------------------------")
        print(f"Retrieving metadata for {vimeo_id}")
        response = self.vimeo_connection.get("/videos/%i" % vimeo_id)
        return response.text

    def _download(self, vimeo_id, metadata_json):
        with self.lock_download(vimeo_id):
            try:
                ad = ArchiveDir.download(self, metadata_json)
                self._archive_dirs[vimeo_id] = ad
            except NoOriginalDownloadError as e:
                print(e, file=sys.stderr)

    def _archive_dir(self, vimeo_id):
        if vimeo_id not in self._archive_dirs:
            self._download(vimeo_id, self.download_metadata_json(vimeo_id))
        return self._archive_dirs[vimeo_id]

    def ensure_current(self, vimeo_id: int, metadata_json=None):
        if not metadata_json:
            metadata_json = self.download_metadata_json(vimeo_id)
        metadata = json.loads(metadata_json)
        ad = self._archive_dirs.get(vimeo_id, None)
        local_mtime = ad.mtime if ad else 0
        remote_mtime = original_ctime(metadata)
        if remote_mtime > local_mtime:
            self._download(vimeo_id, metadata_json)

    def media_file_path(self, vimeo_id: int):
        return self._archive_dir(vimeo_id).media_file_path

    def download_lockfile_path(self, vimeo_id):
        return op.join(self.local_root, "%i.download-lock" % vimeo_id)

    @contextmanager
    def lock_download(self, vimeo_id, wait=0):
        called = time.time()
        path = self.download_lockfile_path(vimeo_id)
        while op.exists(path):
            if wait == 0 or time.time() - called >= wait:
                with open(path) as fp:
                    pid = int(fp.read())
                    raise DownloadLocked("Download for %i locked by pid %i" % (vimeo_id, pid))
            report("Lock exists for %i, waiting..." % vimeo_id)
            time.sleep(5)
        with open(path, "w") as fp:
            print(os.getpid(), file=fp)
        try:
            yield called
        finally:
            self.unlock_download(vimeo_id)

    def unlock_download(self, vimeo_id):
        path = self.download_lockfile_path(vimeo_id)
        if op.exists(path):
            os.unlink(path)

    def sync(self):
        def videos():
            next_uri = "/me/videos?direction=desc&sort=modified_time"
            while next_uri is not None:
                report("Retrieving", next_uri)
                response = self.vimeo_connection.get(next_uri)
                returned = response.json()
                next_uri = returned["paging"]["next"]
                for video in returned["data"]:
                    yield video
        for video in videos():
            uri = video["uri"]
            parts = uri.split("/")
            vimeo_id = int(parts[-1])
            if vimeo_id:
                self.ensure_current(vimeo_id, json.dumps(video))

if __name__ == "__main__":
    def _report_progress_on(item, done, extra=None, metainfo=None):
        if extra:
            extra = "(%s)" % extra
        print("\r" + str(item), "%.1f%%" % (done * 100.0,), extra, end="", file=sys.stderr)

    def _report(*things, end="\n"):
        print(*things, end=end, file=sys.stderr)

    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-v", dest="verbose", action="store_true", default=False)
    parser.add_argument("--progress", dest="progress", action="store_true", default=False)
    parser.add_argument("-c", dest="config_file", type=argparse.FileType("r"), required=True)
    subparsers = parser.add_subparsers(title="Commands", dest="command")
    download_parser = subparsers.add_parser("download", help="Download media for a specific Video by id.")
    download_parser.add_argument("vimeo_ids", type=int, nargs="+")
    sync_parser = subparsers.add_parser("sync", help="Sync local data with vimeo.com.")
    check_parser = subparsers.add_parser("check", help="Check mtime for a specific Video by id.")
    check_parser.add_argument("vimeo_ids", type=int, nargs="+")
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        parser.exit("Please specify a command.")
    if args.config_file is None:
        default_path = op.join(os.getenv("HOME"), ".vimeo_backup")
        if op.exists(default_path):
            config_file_path = default_path
        else:
            parser.exit("Can’t find default config file " + default_path)
    else:
        config_file_path = args.config_file.name
    if args.verbose:
        report = _report
    if args.progress:
        report_progress_on = _report_progress_on
    backup = VimeoBackup.from_config_file(config_file_path)
    if args.command == "download":
        for vimeo_id in args.vimeo_ids:
            try:
                local_path = backup.media_file_path(vimeo_id)
                print("Downloaded media file at:", local_path)
            except DownloadLocked:
                pass
    elif args.command == "sync":
        backup.sync()
    elif args.command == "check":
        for vimeo_id in args.vimeo_ids:
            try:
                backup.ensure_current(vimeo_id)
            except DownloadLocked:
                pass
