#!/usr/bin/env python
"""\
This script will maintain a local copy of your uploads to vimeo.com. 
"""

import sys, os, os.path as op, re, threading, functools, datetime, runpy
import urllib.request, urllib.parse, json, shutil, time

from contextlib import contextmanager

import vimeo

from t4.title_to_id import safe_filename
from t4.typography import pretty_bytes

class DownloadLocked(Exception):
    pass

class NoOriginalDownloadError(Exception):
    pass

def parse_timestamp(s):
    """
    Return a datetime object for vimeo timestamp `s`.
    """
    return datetime.datetime.fromisoformat(s.split("+")[0])

def original_ctime(metadata):
    """
    Lookup the original download’s create_time field,
    parse it, and return as (Unix-)timestamp.
    """
    for d in metadata["download"]:
        if d["quality"] == "source":
            return parse_timestamp(d["created_time"]).timestamp()
    else:
        raise NoOriginalDownloadError(
            "No original download available for %i “%s”" % (
                vimeo_id, metadata["name"]))

def report_progress_on(item, done, extra=None, metainfo=None):
    pass

def report(*things, end="\n"):
    pass


# After a vimeo connection has net been used for this duration of time,
# a new one will be made.
vimeo_connection_timeout = 180

dirname_re = re.compile(r"(\d+).*")
class ArchiveDir(object):
    def __init__(self, backup, dirname):
        self.backup = backup
        self.dirname = dirname

        match = dirname_re.match(dirname)
        if match is None:
            raise IOError("Not an archive dir: " + dirname)
        else:
            self.vimeo_id = int(match.group(1))
        
        self.abspath = op.join(backup.local_root, dirname)           
        
    @classmethod
    def download(ArchiveDir, backup, metadata_json):
        """
        Create an archive dir in `backup` of the video whose `metadata`
        was download from vimeo.
        * Create a target directory
        * store the metadata
        * download the original video file
        * set the correct mtime of the directory
        * return an ArchiveDir instance for it
        """
        metadata = json.loads(metadata_json)
        uri = metadata["uri"]
        rest, vimeo_id = uri.rsplit("/", 1)
        vimeo_id = int(vimeo_id)

        # Download the original file.
        original_download = None
        for d in metadata["download"]:
            if d["quality"] == "source":
                original_download = d
                break
        else:
            raise NoOriginalDownloadError(
                "No original download available for %i “%s”" % (
                    vimeo_id, metadata["name"]))

        url = original_download["link"]

        fields = urllib.parse.urlparse(url)
        path = urllib.parse.unquote(fields.path)
        parts = path.split("/")
        filename = safe_filename(parts[-1])
        fn, ext = op.splitext(filename)
        
        name = metadata["name"]
        filename = safe_filename(name)
        
        dirname = "%i %s" % ( vimeo_id, filename, )
        filename += ext
        # Create the temporary output directory.
        tmppath = op.join(backup.local_root, "tmp." + dirname)
        os.mkdir(tmppath)

        try:
            # Write the metadata
            with open(op.join(tmppath, "metadata.json"), "w") as fp:
                fp.write(metadata_json)

            size = original_download["size"]

            msg = f"Loading “{filename}”"
            report(msg, end="")
            outpath = op.join(tmppath, filename)
            with urllib.request.urlopen(url) as infp, \
                 open(outpath, "wb") as outfp:
                bytes_read = 0
                blocksize = 102400
                while True:
                    bytes = infp.read(blocksize)
                    if len(bytes) == 0:                        
                        break
                    else:
                        outfp.write(bytes)
                        bytes_read += len(bytes)
                        report_progress_on(msg,
                                           float(bytes_read) / float(size),
                                           pretty_bytes(bytes_read),
                                           metainfo={"vimeo_id": vimeo_id})
            report(" done.")
          
            # Check, if we got it all
            if os.path.getsize(outpath) != size:
                try:
                    os.unlink(outpath)
                except IOError:
                    pass

                raise IOError("Failed to download " + url)
        except (Exception, KeyboardInterrupt) as e:
            # On failure, remove the output directory.
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

            # Move the old version of the video to the archive.
            os.rename(target_path, backup_path)

        # Move the version just downloaded to its proper location.
        os.rename(tmppath, target_path)

        # Set our directory’s mtime.
        timestamp = original_ctime(metadata)
        
        # This is in UTC and as a definition, that’s what we want,
        # no metter what our operating system thinks about it.
        os.utime(target_path, (timestamp, timestamp,))

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
        """
        Seconds since the epoch.
        """        
        return op.getmtime(self.abspath)
        
    @functools.cached_property
    def metadata(self):
        path = self.abspath_of("metadata.json")
        with open(path) as fp:
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
            if op.isdir(op.join(self.local_root, dirname)):
                try:
                    ad = ArchiveDir(self, dirname)
                except IOError:
                    pass
                else:
                    self._archive_dirs[ad.vimeo_id] = ad

    @classmethod
    def from_config_file(VimeoBackup, config_file_path):
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
            self._vimeo_connection = vimeo.VimeoClient(
                token=self.vimeo_credentials["access_token"],
                key=self.vimeo_credentials["client_identifyer"],
                secret=self.vimeo_credentials["client_secret"])

        def reset_connection():
            self._connection_reset_timer = None
            self._vimeo_connection = None            
        self._connection_reset_timer = threading.Timer(
            vimeo_connection_timeout, reset_connection)
            
        return self._vimeo_connection

    def download_metadata_json(self, vimeo_id:int):
        report("Retrieving metadata for", vimeo_id)
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
    
    def ensure_current(self, vimeo_id:int, metadata_json=None):
        """
        Download a video’s metadata and compare the modified_time to the
        local file. If needed, this will start a media download.
        """
        if not metadata_json:
            metadata_json = self.download_metadata_json(vimeo_id)
        metadata = json.loads(metadata_json)

        ad = self._archive_dirs.get(vimeo_id, None)
        
        if ad:
            local_mtime = ad.mtime
        else:
            local_mtime = 0

        remote_mtime = original_ctime(metadata)


        if remote_mtime > local_mtime:
            self._download(vimeo_id, metadata_json)

    def media_file_path(self, vimeo_id:int):
        """
        Return the absolute local path for a media file. If not available,
        it will be downloaded. This will not check, if a more recent upload
        is available on vimeo. This may be a long running function call.
        """
        self._archive_dir(vimeo_id).media_file_path

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

                    if wait > 0:
                        for_ = " (waited at least %i sec.)" % wait
                    else:
                        for_ = ""
                        
                    raise DownloadLocked(
                        "Download for %i locked by pid %i%s" % (
                            vimeo_id, pid, for_))

            report("Lock exisists for %i, waiting..." % vimeo_id)
            time.sleep(5)

        # Write my pid to the lock file.
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
        """
        Print progress how much of `item` is `done`, with 0 <= done <= 1.0.
        """
        if extra:
            extra = "(%s)" % extra

        print("\r" + str(item), "%.1f%%" % ( done*100.0, ), extra,
              end="", file=sys.stderr)

    def _report(*things, end="\n"):
        print(*things, end=end, file=sys.stderr)

    import sys, argparse
    
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-v", dest="verbose", action="store_true",
                        default=False)
    parser.add_argument("--progress",
                        dest="progress", action="store_true", default=False)
    parser.add_argument("-c", dest="config_file", type=argparse.FileType("r"),
                        required=True)

    subparsers = parser.add_subparsers(title="Commands", dest="command")

    download_parser = subparsers.add_parser(
        "download", help="Download media for a specific Video by id.")
    download_parser.add_argument("vimeo_ids", type=int, nargs="+")

    sync_parser = subparsers.add_parser(
        "sync", help="Sync local data with vimeo.com.")
    
    check_parser = subparsers.add_parser(
        "check", help="Check mtime for a specific Video by id.")
    check_parser.add_argument("vimeo_ids", type=int, nargs="+")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        print(file=sys.stderr)
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


    
