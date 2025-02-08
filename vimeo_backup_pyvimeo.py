#!/usr/bin/env python
"""\
This script maintains a local copy of your uploads to vimeo.com using the official PyVimeo library.
此程式使用官方 PyVimeo 庫來備份你在 vimeo.com 上的上傳內容。
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

# 使用官方 PyVimeo 庫
from vimeo import VimeoClient

# 若無 t4 模組，請自行建立以下兩個檔案：
# t4/title_to_id.py:
#   import re
#   def safe_filename(name):
#       return re.sub(r'[\\/:"*?<>|]+', "", name)
#
# t4/typography.py:
#   def pretty_bytes(num):
#       for unit in ['B','KB','MB','GB','TB']:
#           if num < 1024.0:
#               return f"{num:.1f}{unit}"
#           num /= 1024.0
#       return f"{num:.1f}PB"
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
    """Parse a Vimeo timestamp string and return a datetime object.
    解析 Vimeo 時間字串並回傳 datetime 物件。"""
    return dateutil.parser.parse(s)

def original_ctime(metadata):
    """Get original download create_time timestamp from metadata.
    從 metadata 中取得原始下載的建立時間 timestamp。"""
    for d in metadata["download"]:
        if d["quality"] == "source":
            return parse_timestamp(d["created_time"]).timestamp()
    else:
        video_id = metadata.get("uri", "unknown").split("/")[-1]
        raise NoOriginalDownloadError(
            "No original download available for %s “%s” (無法取得影片 %s 的原始下載)" % (video_id, metadata["name"], video_id))

def report_progress_on(item, done, extra=None, metainfo=None):
    # 此函式可擴充以顯示更詳細的進度，目前留空。
    pass

def report(*things, end="\n"):
    print(*things, end=end, file=sys.stderr)

class ArchiveDir(object):
    def __init__(self, backup, dirname):
        """Initialize ArchiveDir instance.
        初始化 ArchiveDir 實例。"""
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
        """Download a video and create an archive directory.
        下載影片並建立存檔資料夾。"""
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
                "No original download available for %i “%s” (無法取得影片 %i 的原始下載)" % (vimeo_id, metadata["name"], vimeo_id))

        url = original_download["link"]
        size = original_download["size"]

        # 輸出分隔線與影片基本資訊 (中英文雙語)
        print("\n\n----------------------------------------")
        print(f"下載影片 ID (Downloading video ID): {vimeo_id}  Title: {metadata['name']}")
        print(f"檔案大小 (File size): {pretty_bytes(size)}")
        print(f"下載網址 (Download URL): {url}")

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
            print(f"開始下載 (Start downloading) {filename} ...")
            start_time = time.time()
            with urllib.request.urlopen(url, context=ssl_context) as infp, open(outpath, "wb") as outfp:
                blocksize = 102400
                # 使用 tqdm 進度條，更新間隔 1 秒
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
            # 在下載完成 log 前加換行
            print("")
            print(f"下載完成 (Download completed) {filename}: 用時 {elapsed:.1f} 秒，平均速度 {pretty_bytes(avg_speed)}/s (Elapsed {elapsed:.1f} sec, Average speed {pretty_bytes(avg_speed)}/s)")
            if os.path.getsize(outpath) != size:
                try:
                    os.unlink(outpath)
                except IOError:
                    pass
                raise IOError("下載失敗 (Download failed): " + url)
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
        print("\n\n----------------------------------------")
        print(f"Retrieving metadata for {vimeo_id} (取得影片 {vimeo_id} 的 metadata)")
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
        return op.join(self.local_root, f"{vimeo_id}.download-lock")

    @contextmanager
    def lock_download(self, vimeo_id, wait=0):
        called = time.time()
        path = self.download_lockfile_path(vimeo_id)
        while op.exists(path):
            if wait == 0 or time.time() - called >= wait:
                with open(path) as fp:
                    pid = int(fp.read())
                    raise DownloadLocked(f"Download for {vimeo_id} locked by pid {pid} (下載鎖定：影片 {vimeo_id} 被 PID {pid} 鎖定)")
            report(f"Lock exists for {vimeo_id}, waiting... (等待中：影片 {vimeo_id} 的鎖定存在)")
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

    def download_video(self, vimeo_id: int, retry=3, sleep_interval=2):
        """Download a video with retry mechanism and sleep between attempts.
        使用重試機制下載影片，下載完成後休息指定秒數。"""
        attempt = 0
        while attempt < retry:
            try:
                local_path = self.media_file_path(vimeo_id)
                return local_path
            except Exception as e:
                attempt += 1
                print(f"Error downloading video {vimeo_id}: {e} - 嘗試第 {attempt}/{retry} 次重試 (Error downloading video {vimeo_id}: {e} - retry attempt {attempt}/{retry})", file=sys.stderr)
                time.sleep(sleep_interval)
        raise Exception(f"Failed to download video {vimeo_id} after {retry} attempts (影片 {vimeo_id} 下載失敗，重試 {retry} 次後仍失敗)")

    def sync_concurrent(self, concurrency=3, sleep_interval=2, retry=3):
        """Sync all videos concurrently with specified concurrency.
        使用指定併發數同步所有影片。"""
        # 取得 /me/videos 列表
        videos_list = []
        next_uri = "/me/videos?direction=desc&sort=modified_time"
        while next_uri is not None:
            print(f"Retrieving {next_uri} (取得 {next_uri})")
            response = self.vimeo_connection.get(next_uri)
            returned = response.json()
            next_uri = returned["paging"]["next"]
            for video in returned["data"]:
                videos_list.append(video)
        # 使用 ThreadPoolExecutor 進行併發下載
        from concurrent.futures import ThreadPoolExecutor, as_completed
        results = {}
        max_workers = min(concurrency, 10)
        print(f"開始併發下載影片 (Start concurrent downloads) with {max_workers} threads")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_vid = {executor.submit(self.download_video, int(video["uri"].rsplit("/", 1)[-1]), retry, sleep_interval): video for video in videos_list}
            for future in as_completed(future_to_vid):
                video = future_to_vid[future]
                vimeo_id = int(video["uri"].rsplit("/", 1)[-1])
                try:
                    local_path = future.result()
                    results[vimeo_id] = local_path
                    # 在印出下載成功前加入換行
                    print(f"\nDownloaded video {vimeo_id} at: {local_path} (影片 {vimeo_id} 下載成功)")
                except Exception as e:
                    print(f"影片 {vimeo_id} 下載失敗 (Failed downloading video {vimeo_id}): {e}", file=sys.stderr)
        return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-v", dest="verbose", action="store_true", default=False, help="顯示詳細訊息 / Verbose output")
    parser.add_argument("--progress", dest="progress", action="store_true", default=False, help="顯示進度條 / Show progress bar")
    parser.add_argument("-c", dest="config_file", type=argparse.FileType("r"), required=True, help="設定檔 / Config file")
    parser.add_argument("--concurrency", type=int, default=3, help="併發下載數量 (最大 10) / Number of concurrent downloads (max 10)")
    parser.add_argument("--sleep", type=float, default=2, help="每支影片下載後休息秒數 / Sleep seconds between downloads (default 2)")
    parser.add_argument("--retry", type=int, default=3, help="重試次數 / Number of retry attempts (default 3)")
    subparsers = parser.add_subparsers(title="Commands", dest="command")
    download_parser = subparsers.add_parser("download", help="Download media for specific Video(s) by id. / 根據影片 ID 下載影片")
    download_parser.add_argument("vimeo_ids", type=int, nargs="+", help="Vimeo 影片 ID(s)")
    sync_parser = subparsers.add_parser("sync", help="Sync local data with vimeo.com. / 同步 Vimeo 帳戶內所有影片")
    check_parser = subparsers.add_parser("check", help="Check mtime for specific Video(s) by id. / 檢查指定影片的修改時間")
    check_parser.add_argument("vimeo_ids", type=int, nargs="+", help="Vimeo 影片 ID(s)")
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        parser.exit("請指定一個命令 (Please specify a command.)")
    if args.config_file is None:
        default_path = op.join(os.getenv("HOME"), ".vimeo_backup")
        if op.exists(default_path):
            config_file_path = default_path
        else:
            parser.exit("找不到設定檔，無法繼續。 (Can't find default config file " + default_path + ")")
    else:
        config_file_path = args.config_file.name
    if args.verbose:
        report = lambda *x, **y: print(*x, **y, file=sys.stderr)
    backup = VimeoBackup.from_config_file(config_file_path)
    if args.command == "download":
        # 使用 ThreadPoolExecutor 併發下載指定影片
        from concurrent.futures import ThreadPoolExecutor, as_completed
        max_workers = min(args.concurrency, 10)
        print(f"開始併發下載影片 (Start concurrent downloads) with {max_workers} threads")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_vid = {executor.submit(backup.download_video, v_id, args.retry, args.sleep): v_id for v_id in args.vimeo_ids}
            for future in as_completed(future_to_vid):
                v_id = future_to_vid[future]
                try:
                    local_path = future.result()
                    print(f"\nDownloaded video {v_id} at: {local_path}")
                except Exception as e:
                    print(f"影片 {v_id} 下載失敗 (Failed downloading video {v_id}): {e}", file=sys.stderr)
    elif args.command == "sync":
        backup.sync_concurrent(concurrency=args.concurrency, sleep_interval=args.sleep, retry=args.retry)
    elif args.command == "check":
        for v_id in args.vimeo_ids:
            try:
                backup.ensure_current(v_id)
            except DownloadLocked:
                pass
