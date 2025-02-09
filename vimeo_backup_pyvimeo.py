#!/usr/bin/env python
"""\
This script maintains a local copy of your uploads to vimeo.com using the official PyVimeo library,
and supports syncing videos according to folder hierarchy.
此程式使用官方 PyVimeo 庫來備份你在 vimeo.com 上的影片，並支援依據資料夾層次同步下載影片.
It also cleans up stale temporary directories and lock files (with a 5-minute threshold)
so that if interrupted, re-running the command will resume properly.
此外，它會自動清除超過 5 分鐘（300 秒）的臨時目錄與鎖定檔，確保中斷後重新執行時能正確續傳.
"""

import sys, os, os.path as op, re, threading, functools, datetime, runpy, time, json, shutil
import urllib.request, urllib.parse
from tqdm import tqdm
from contextlib import contextmanager
import dateutil.parser

import ssl
import certifi
# Create SSL context using certifi certificates
ssl_context = ssl.create_default_context(cafile=certifi.where())

from vimeo import VimeoClient  # PyVimeo library

# --- t4/title_to_id.py ---
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

import collections, collections.abc
if not hasattr(collections, 'MutableMapping'):
    collections.MutableMapping = collections.abc.MutableMapping

# ----------------------------
# Helper Function: folder_complete
# Checks if folder is complete: metadata.json exists and the media file size
# matches metadata["expected_size"] (within 1% difference).
# ----------------------------
def folder_complete(folder_path):
    meta_path = op.join(folder_path, "metadata.json")
    if not op.exists(meta_path):
        return False
    try:
        with open(meta_path, "r", encoding="utf-8") as fp:
            meta = json.load(fp)
    except Exception as e:
        print(f"Error reading metadata.json in {folder_path}: {e}", file=sys.stderr)
        return False

    expected = meta.get("expected_size")
    if expected is None:
        return False

    for fn in os.listdir(folder_path):
        if fn != "metadata.json" and op.isfile(op.join(folder_path, fn)):
            actual = op.getsize(op.join(folder_path, fn))
            diff_ratio = abs(actual - expected) / expected
            print(f"[DEBUG] Folder {folder_path}: expected_size={expected}, actual_size={actual}, diff_ratio={diff_ratio:.3f}",
                  file=sys.stderr)
            if diff_ratio < 0.01:
                return True
    return False

# ----------------------------
# Exception Classes
# ----------------------------
class DownloadLocked(Exception):
    pass

class NoOriginalDownloadError(Exception):
    pass

# ----------------------------
# Utility Functions
# ----------------------------
def parse_timestamp(s):
    return dateutil.parser.parse(s)

def original_ctime(metadata):
    """Get original download create_time timestamp from metadata."""
    for d in metadata["download"]:
        if d["quality"] == "source":
            return parse_timestamp(d["created_time"]).timestamp()
    else:
        video_id = metadata.get("uri", "unknown").split("/")[-1]
        raise NoOriginalDownloadError(
            f"No original download available for {video_id} “{metadata['name']}”"
        )

def report_progress_on(item, done, extra=None, metainfo=None):
    pass

def report(*things, end="\n"):
    print(*things, end=end, file=sys.stderr)

# ----------------------------
# ArchiveDir Class
# ----------------------------
class ArchiveDir(object):
    def __init__(self, backup, dirname, abspath):
        self.backup = backup
        self.dirname = dirname
        self.abspath = abspath
        match = re.match(r"(\d+).*", dirname)
        if match is None:
            raise IOError("Not an archive dir: " + dirname)
        else:
            self.vimeo_id = int(match.group(1))

    @classmethod
    def download(cls, backup, metadata_json, base_dir=None):
        """Download a video and create an archive directory in base_dir."""
        metadata = json.loads(metadata_json)
        uri = metadata["uri"]
        _, vimeo_id = uri.rsplit("/", 1)
        vimeo_id = int(vimeo_id)

        original_download = None
        for d in metadata["download"]:
            if d["quality"] == "source":
                original_download = d
                break
        else:
            raise NoOriginalDownloadError(
                f"No original download available for {vimeo_id} “{metadata['name']}”"
            )

        url = original_download["link"]
        size = original_download["size"]
        metadata["expected_size"] = size  # Store expected size for verification

        print("\n\n----------------------------------------")
        print(f"下載影片 ID (Downloading video ID): {vimeo_id}  Title: {metadata['name']}")
        print(f"檔案大小 (File size): {pretty_bytes(size)}")
        print(f"下載網址 (Download URL): {url}")

        base = base_dir if base_dir is not None else backup.base_dir
        fields = urllib.parse.urlparse(url)
        path = urllib.parse.unquote(fields.path)
        parts = path.split("/")
        filename = safe_filename(parts[-1])
        fn, ext = op.splitext(filename)
        title = safe_filename(metadata["name"])
        filename = f"{title}{ext}"

        dirname = f"{vimeo_id} {title}"
        tmppath = op.join(base, "tmp." + dirname)
        if op.exists(tmppath):
            print(f"Found leftover tmp folder {tmppath}, removing...", file=sys.stderr)
            shutil.rmtree(tmppath, ignore_errors=True)
        os.mkdir(tmppath)

        try:
            with open(op.join(tmppath, "metadata.json"), "w", encoding="utf-8") as fp:
                json.dump(metadata, fp, indent=4, ensure_ascii=False)

            outpath = op.join(tmppath, filename)
            print(f"Start downloading {filename} ...")
            start_time = time.time()
            with urllib.request.urlopen(url, context=ssl_context) as infp, open(outpath, "wb") as outfp:
                blocksize = 102400
                with tqdm(total=size, unit='B', unit_scale=True, desc=filename, mininterval=1) as pbar:
                    while True:
                        chunk = infp.read(blocksize)
                        if not chunk:
                            break
                        outfp.write(chunk)
                        pbar.update(len(chunk))

            end_time = time.time()
            elapsed = end_time - start_time
            avg_speed = size / elapsed if elapsed > 0 else 0
            print("")
            print(f"Download completed {filename}: elapsed {elapsed:.1f}s, avg speed {pretty_bytes(avg_speed)}/s")

            if os.path.getsize(outpath) != size:
                try:
                    os.unlink(outpath)
                except IOError:
                    pass
                shutil.rmtree(tmppath, ignore_errors=True)
                raise IOError("Download failed: " + url)

        except (Exception, KeyboardInterrupt) as e:
            shutil.rmtree(tmppath, ignore_errors=True)
            raise

        target_path = op.join(base, dirname)
        if op.exists(target_path) and folder_complete(target_path):
            print(f"Folder {target_path} exists and is complete, skip download.")
            shutil.rmtree(tmppath, ignore_errors=True)
            return ArchiveDir(backup, dirname, target_path)
        else:
            if op.exists(target_path):
                shutil.rmtree(target_path, ignore_errors=True)

        # Move tmp folder to final
        os.rename(tmppath, target_path)
        timestamp = original_ctime(metadata)
        os.utime(target_path, (timestamp, timestamp))
        return ArchiveDir(backup, dirname, target_path)

    def abspath_of(self, filename):
        return op.join(self.abspath, filename)

    @functools.cached_property
    def media_file_path(self):
        for fn in os.listdir(self.abspath):
            abspath = op.join(self.abspath, fn)
            if fn != "metadata.json" and op.isfile(abspath):
                return abspath
        raise IOError("No media file downloaded in folder: " + repr(self.dirname))

    @functools.cached_property
    def mtime(self):
        return op.getmtime(self.abspath)

    @functools.cached_property
    def metadata(self):
        path = self.abspath_of("metadata.json")
        with open(path, "r", encoding="utf-8") as fp:
            return json.load(fp)

    def delete(self):
        shutil.rmtree(self.abspath)
        self.backup.invalidate_archive_dir(self.vimeo_id)

# ----------------------------
# VimeoBackup Class
# ----------------------------
class VimeoBackup(object):
    def __init__(self, local_root, vimeo_credentials):
        self.local_root = local_root
        self.base_dir = local_root
        self.vimeo_credentials = vimeo_credentials
        self._vimeo_connection = None
        self._connection_reset_timer = None
        self._archive_dirs = {}

    @classmethod
    def from_config_file(cls, config_file_path):
        """
        Loads the config file (like a .rc or .py) via runpy, and returns VimeoBackup instance.
        透過 runpy 讀取設定檔 (Python 格式)，然後建立並回傳 VimeoBackup 實例。
        """
        config = runpy.run_path(config_file_path)
        return cls(config["local_root"], config)

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
                secret=self.vimeo_credentials["client_secret"]
            )

        def reset_connection():
            self._connection_reset_timer = None
            self._vimeo_connection = None

        self._connection_reset_timer = threading.Timer(180, reset_connection)
        return self._vimeo_connection

    def cleanup_stale_files(self, stale_threshold=300):
        now = time.time()
        for item in os.listdir(self.base_dir):
            full_path = op.join(self.base_dir, item)
            # Clean up stale tmp folders
            if item.startswith("tmp.") and op.isdir(full_path):
                mtime = op.getmtime(full_path)
                if now - mtime > stale_threshold:
                    print(f"Removing stale temporary directory: {full_path}", file=sys.stderr)
                    shutil.rmtree(full_path, ignore_errors=True)
            # Clean up stale download-lock
            if item.endswith(".download-lock") and op.isfile(full_path):
                mtime = op.getmtime(full_path)
                if now - mtime > stale_threshold:
                    print(f"Removing stale lock file: {full_path}", file=sys.stderr)
                    os.unlink(full_path)

    def download_metadata_json(self, vimeo_id: int):
        print("\n\n----------------------------------------")
        print(f"Retrieving metadata for {vimeo_id}")
        response = self.vimeo_connection.get(f"/videos/{vimeo_id}")
        return response.text

    def _download(self, vimeo_id, metadata_json, base_dir=None):
        with self.lock_download(vimeo_id):
            try:
                ad = ArchiveDir.download(self, metadata_json, base_dir=base_dir)
                self._archive_dirs[vimeo_id] = ad
            except NoOriginalDownloadError as e:
                print(e, file=sys.stderr)

    def _search_archive_dir_in(self, vimeo_id, base_dir):
        """
        Recursively search for a folder named like '{vimeo_id} ...' under base_dir,
        and check if it's complete. Return ArchiveDir if found.
        遞迴搜尋 base_dir 下以 '{vimeo_id} ' 開頭的資料夾，檢查完整度後回傳 ArchiveDir。
        """
        for root, dirs, _ in os.walk(base_dir):
            for d in dirs:
                if d.startswith(f"{vimeo_id} "):
                    candidate_path = op.join(root, d)
                    try:
                        if folder_complete(candidate_path):
                            with open(op.join(candidate_path, "metadata.json"), "r", encoding="utf-8") as fp:
                                meta = json.load(fp)
                            if meta.get("uri", "").endswith(f"/{vimeo_id}"):
                                return ArchiveDir(self, d, candidate_path)
                    except Exception as e:
                        print(f"[DEBUG] Error checking folder {candidate_path}: {e}", file=sys.stderr)
        return None

    def _archive_dir(self, vimeo_id, base_dir=None):
        base = base_dir if base_dir is not None else self.base_dir

        # Check cache
        if vimeo_id in self._archive_dirs:
            ad = self._archive_dirs[vimeo_id]
            # Ensure the cached folder is still valid and inside base_dir
            if ad.abspath.startswith(op.abspath(base)) and op.exists(ad.abspath) and folder_complete(ad.abspath):
                print(f"[DEBUG] Found cached complete folder {ad.abspath} for video {vimeo_id}", file=sys.stderr)
                return ad
            else:
                print(f"[DEBUG] Cached folder {ad.abspath} not found or incomplete for video {vimeo_id}", file=sys.stderr)
                self.invalidate_archive_dir(vimeo_id)

        # Search on disk
        found = self._search_archive_dir_in(vimeo_id, base)
        if found:
            print(f"[DEBUG] Found matching folder {found.abspath} for video {vimeo_id}", file=sys.stderr)
            self._archive_dirs[vimeo_id] = found
            return found

        # If not found, download it
        print(f"[DEBUG] Folder for video {vimeo_id} not found or incomplete, proceeding to download", file=sys.stderr)
        self._download(vimeo_id, self.download_metadata_json(vimeo_id), base_dir=base)

        # Search again after download
        found_after = self._search_archive_dir_in(vimeo_id, base)
        if found_after:
            self._archive_dirs[vimeo_id] = found_after
            return found_after

        raise Exception(f"Folder for video {vimeo_id} not found or incomplete.")

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
        return op.join(self.base_dir, f"{vimeo_id}.download-lock")

    @contextmanager
    def lock_download(self, vimeo_id, wait=0):
        called = time.time()
        path = self.download_lockfile_path(vimeo_id)
        stale_threshold = 300
        while op.exists(path):
            mtime = op.getmtime(path)
            if time.time() - mtime > stale_threshold:
                print(f"Lock file {path} is stale. Removing it.", file=sys.stderr)
                os.unlink(path)
                break
            if wait == 0 or time.time() - called >= wait:
                with open(path) as fp:
                    pid = int(fp.read())
                raise DownloadLocked(f"Download for {vimeo_id} locked by pid {pid}")
            report(f"Lock exists for {vimeo_id}, waiting...")
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

    def download_video(self, vimeo_id: int, retry=1, sleep_interval=2):
        """Download a video with a retry mechanism."""
        attempt = 0
        while attempt < retry:
            try:
                local_path = self.media_file_path(vimeo_id)
                return local_path
            except Exception as e:
                attempt += 1
                print(f"Error downloading video {vimeo_id}: {e} - retry {attempt}/{retry}", file=sys.stderr)
                time.sleep(sleep_interval)
        raise Exception(f"Failed to download video {vimeo_id} after {retry} attempts")

    def _download_in_folder(self, vimeo_id, base_dir):
        metadata_json = self.download_metadata_json(vimeo_id)
        with self.lock_download(vimeo_id):
            ad = ArchiveDir.download(self, metadata_json, base_dir=base_dir)
            self._archive_dirs[vimeo_id] = ad
        return ad.media_file_path

    def download_video_in_folder(self, vimeo_id: int, base_dir, retry=1, sleep_interval=2):
        """Download a video into specified base_dir with a retry mechanism."""
        try:
            ad = self._archive_dir(vimeo_id, base_dir=base_dir)
            print(f"[DEBUG] 目標資料夾存在：{ad.abspath}")
            return ad.media_file_path
        except Exception as ex:
            print(f"資料夾對應影片 {vimeo_id} 不存在或不完整，將進行下載: {ex}", file=sys.stderr)

        attempt = 0
        while attempt < retry:
            try:
                local_path = self._download_in_folder(vimeo_id, base_dir)
                return local_path
            except Exception as e:
                attempt += 1
                print(f"Error downloading video {vimeo_id} in folder: {e} - retry {attempt}/{retry}", file=sys.stderr)
                for item in os.listdir(base_dir):
                    if item.startswith(f"tmp.{vimeo_id} "):
                        full_path = op.join(base_dir, item)
                        print(f"Removing leftover tmp folder: {full_path}", file=sys.stderr)
                        shutil.rmtree(full_path, ignore_errors=True)
                time.sleep(sleep_interval)
        raise Exception(f"Failed to download video {vimeo_id} in folder after {retry} attempts")

    def get_folder_tree(self):
        """
        Retrieves folder tree from Vimeo API and build a recursive structure.
        This reads /me/projects and for each folder, calls /me/projects/{folder_id}/folders to get children.
        """
        folders = []
        next_uri = "/me/projects"
        while next_uri is not None:
            print(f"Retrieving folders from {next_uri}")
            response = self.vimeo_connection.get(next_uri)
            data = response.json()
            folders.extend(data["data"])
            next_uri = data["paging"].get("next")

        def build_tree(folder):
            folder_id = folder["uri"].split("/")[-1]
            resp = self.vimeo_connection.get(f"/me/projects/{folder_id}/folders")
            children = resp.json().get("data", [])
            folder["children"] = [build_tree(child) for child in children]
            return folder

        tree = [build_tree(f) for f in folders]
        return tree

    def sync_with_folders(self, concurrency=3, sleep_interval=2, retry=1, skip_download=False):
        """
        Sync videos based on folder hierarchy.
        For each folder, create corresponding local folder, then either skip-download
        or actually download videos into that folder (with concurrency).
        """
        self.cleanup_stale_files()
        folder_tree = self.get_folder_tree()

        def process_folder(node, parent_path=""):
            local_folder = op.join(self.base_dir, parent_path, safe_filename(node["name"]))
            os.makedirs(local_folder, exist_ok=True)
            print(f"\n\n同步資料夾 (Sync folder): {local_folder}")

            folder_id = node["uri"].split("/")[-1]
            response = self.vimeo_connection.get(f"/me/projects/{folder_id}/videos")
            videos = response.json().get("data", [])

            if skip_download:
                summary_path = op.join(local_folder, "videos_summary.json")
                with open(summary_path, "w", encoding="utf-8") as fp:
                    json.dump(videos, fp, indent=4, ensure_ascii=False)
                print(f"\n[SKIP DOWNLOAD] Video list saved at: {summary_path}")
            else:
                from concurrent.futures import ThreadPoolExecutor, as_completed
                max_workers = min(concurrency, 10)
                futures = {}
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    for video in videos:
                        v_id = int(video["uri"].rsplit("/", 1)[-1])
                        fut = executor.submit(self.download_video_in_folder, v_id, local_folder, retry, sleep_interval)
                        futures[fut] = v_id

                    for future in as_completed(futures):
                        v_id = futures[future]
                        try:
                            local_path = future.result()
                            print(f"\nDownloaded video {v_id} in folder at: {local_path}")
                        except Exception as e:
                            print(f"影片 {v_id} 下載失敗 in folder: {e}", file=sys.stderr)

            for child in node.get("children", []):
                process_folder(child, parent_path=op.join(parent_path, safe_filename(node["name"])))

        for node in folder_tree:
            process_folder(node)

    def sync_concurrent(self, concurrency=3, sleep_interval=2, retry=1):
        """
        Sync all videos (flat) ignoring folder structure.
        Calls /me/videos?direction=desc&sort=modified_time and downloads them concurrently.
        """
        videos_list = []
        next_uri = "/me/videos?direction=desc&sort=modified_time"
        while next_uri is not None:
            print(f"Retrieving {next_uri}")
            response = self.vimeo_connection.get(next_uri)
            returned = response.json()
            next_uri = returned["paging"].get("next")
            for video in returned["data"]:
                videos_list.append(video)

        from concurrent.futures import ThreadPoolExecutor, as_completed
        results = {}
        max_workers = min(concurrency, 10)
        print(f"開始併發下載影片 with {max_workers} threads")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_vid = {
                executor.submit(self.download_video, int(video["uri"].rsplit("/", 1)[-1]), retry, sleep_interval): video
                for video in videos_list
            }
            for future in as_completed(future_to_vid):
                video = future_to_vid[future]
                vimeo_id = int(video["uri"].rsplit("/", 1)[-1])
                try:
                    local_path = future.result()
                    results[vimeo_id] = local_path
                    print(f"\nDownloaded video {vimeo_id} at: {local_path}")
                except Exception as e:
                    print(f"影片 {vimeo_id} 下載失敗: {e}", file=sys.stderr)
        return results

    def show_folder_structure(self):
        """
        Retrieve folder structure from Vimeo API and display + save summary.
        """
        folders = []
        next_uri = "/me/projects"
        while next_uri is not None:
            print(f"Retrieving folders from {next_uri}")
            response = self.vimeo_connection.get(next_uri)
            data = response.json()
            folders.extend(data["data"])
            next_uri = data["paging"].get("next")

        folder_summary = []
        for folder in folders:
            folder_id = folder["uri"].split("/")[-1]
            name = folder.get("name", "")
            videos_total = folder.get("metadata", {}).get("connections", {}).get("videos", {}).get("total", 0)
            subfolders_total = folder.get("metadata", {}).get("connections", {}).get("folders", {}).get("total", 0)
            folder_summary.append({
                "folder_id": folder_id,
                "name": name,
                "videos_total": videos_total,
                "subfolders_total": subfolders_total
            })

        print("\n\nFolder Structure:")
        print(json.dumps(folder_summary, indent=4, ensure_ascii=False))
        output_path = op.join(self.base_dir, "folder_structure.json")
        with open(output_path, "w", encoding="utf-8") as fp:
            json.dump(folder_summary, fp, indent=4, ensure_ascii=False)
        print(f"\nFolder structure saved to {output_path}")
        return folder_summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-v", dest="verbose", action="store_true", default=False,
                        help="Verbose output")
    parser.add_argument("--progress", dest="progress", action="store_true", default=False,
                        help="Show progress bar")
    parser.add_argument("-c", dest="config_file", type=argparse.FileType("r"), required=True,
                        help="Config file (Python script format)")
    parser.add_argument("--concurrency", type=int, default=3,
                        help="Number of concurrent downloads (max 10)")
    parser.add_argument("--sleep", type=float, default=2,
                        help="Seconds to sleep between downloads")
    parser.add_argument("--retry", type=int, default=1,
                        help="Number of retry attempts")
    subparsers = parser.add_subparsers(title="Commands", dest="command")

    download_parser = subparsers.add_parser("download",
                                            help="Download media by specified video id(s)")
    download_parser.add_argument("vimeo_ids", type=int, nargs="+", help="Vimeo video ID(s)")

    sync_parser = subparsers.add_parser("sync", help="Sync all videos (flat)")

    check_parser = subparsers.add_parser("check", help="Check mtime for video id(s)")
    check_parser.add_argument("vimeo_ids", type=int, nargs="+", help="Vimeo video ID(s)")

    folder_parser = subparsers.add_parser("show_folder_structure",
                                          help="Show Vimeo folder structure")

    folders_sync_parser = subparsers.add_parser("sync_folders",
                                                help="Sync videos based on folder hierarchy")
    folders_sync_parser.add_argument("--concurrency", type=int, default=3,
                                     help="Concurrent downloads (max 10)")
    folders_sync_parser.add_argument("--sleep", type=float, default=2,
                                     help="Seconds to sleep between each download (default 2)")
    folders_sync_parser.add_argument("--retry", type=int, default=1,
                                     help="Number of retry attempts (default 1)")
    folders_sync_parser.add_argument("--skip-download", action="store_true", default=False,
                                     help="Only create folder structure without downloading videos")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        parser.exit("Please specify a command.")

    # Handle config file path
    if args.config_file is None:
        default_path = op.join(os.getenv("HOME"), ".vimeo_backup")
        if op.exists(default_path):
            config_file_path = default_path
        else:
            parser.exit("No config file found.")
    else:
        config_file_path = args.config_file.name

    if args.verbose:
        report = lambda *x, **y: print(*x, **y, file=sys.stderr)

    # Instantiate VimeoBackup from config file
    backup = VimeoBackup.from_config_file(config_file_path)

    # Clean up stale files
    backup.cleanup_stale_files()

    if args.command == "download":
        from concurrent.futures import ThreadPoolExecutor, as_completed
        max_workers = min(args.concurrency, 10)
        print(f"開始併發下載影片 with {max_workers} threads")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_vid = {
                executor.submit(backup.download_video, v_id, args.retry, args.sleep): v_id
                for v_id in args.vimeo_ids
            }
            for future in as_completed(future_to_vid):
                v_id = future_to_vid[future]
                try:
                    local_path = future.result()
                    print(f"\nDownloaded video {v_id} at: {local_path}")
                except Exception as e:
                    print(f"影片 {v_id} 下載失敗: {e}", file=sys.stderr)

    elif args.command == "sync":
        backup.sync_concurrent(concurrency=args.concurrency,
                               sleep_interval=args.sleep,
                               retry=args.retry)

    elif args.command == "check":
        for v_id in args.vimeo_ids:
            try:
                backup.ensure_current(v_id)
            except DownloadLocked as dl:
                print(dl, file=sys.stderr)

    elif args.command == "show_folder_structure":
        backup.show_folder_structure()

    elif args.command == "sync_folders":
        backup.sync_with_folders(
            concurrency=args.concurrency,
            sleep_interval=args.sleep,
            retry=args.retry,
            skip_download=args.skip_download
        )
