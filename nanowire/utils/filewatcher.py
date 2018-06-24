import os
import psutil
import subprocess
import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
import tables as tb
from nanowire.utils.utils import wait_until_released
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler

try:
    from Queue import Queue
except ImportError:
    from queue import Queue


def run_hdf5_watcher(watch_dirs, master, recursive=True):
    if not watch_dirs:
        watch_dirs = (os.getcwd(), )
    hands = (HDF5FileMerger(master, remove=True), )
    blocking_watcher(watch_dirs, hands, recursive=recursive)


def blocking_watcher(watch_dirs, handlers, recursive=True):
    """
    Creates a :py:class:`watchdog.observers.Observer` that watches a list of
    directories for any modifications, emitting events to the specified
    handlers when modifications ocurr. This function runs an infinite, blocking
    loop after scheduling all the handlers and starting the filesystem
    observer. Interruptible with any exception, including KeyboardInterrupt

    Parameters
    ----------
    watch_dirs : tuple, list
        A tuple or list of directories to watch.
    handlers : tuple, list
        A tuple or list of handlers that will receive events when any of
        the directories in watch_dirs is modified
    recursive : bool
        Whether or not the entire directory tree beneath each specified
        directory will be watched, or just the contents immediately below
        that directory. Default: True
    """
    log = logging.getLogger(__name__)
    for d in watch_dirs:
        if not os.path.isdir(d):
            raise ValueError("Path {} is not a directory".format(d))
    observer = Observer()
    for d in watch_dirs:
        for h in handlers:
            observer.schedule(h, d, recursive=recursive)
    observer.start()
    try:
        while True:
            print('Observing dirs ...')
            time.sleep(5)
    except KeyboardInterrupt:
        log.info('Stopping observer')
        for h in handlers:
            try:
                getattr(h, 'stop')()
                print('Stopped handler {}'.format(h))
            except AttributeError:
                pass
        observer.stop()
        observer.join()
        return
    except:
        log.error("Error")
        for h in handlers:
            try:
                getattr(h, 'stop')()
                print('Stopped handler {}'.format(h))
            except AttributeError:
                pass
        observer.stop()
        observer.join()
        raise


def threaded_watcher(watch_dirs, handlers, start=True, recursive=True):
    """
    Creates a :py:class:`watchdog.observers.Observer` that watches a list of
    directories for any modifications, emitting events to the specified
    handlers when modifications ocurr. This function is non-blocking, and just
    returns the Observer object and a list of the ObservedWatch objects created

    Parameters
    ----------
    watch_dirs : tuple, list
        A tuple or list of directories to watch.
    handlers : tuple, list
        A tuple or list of handlers that will receive events when any of
        the directories in watch_dirs is modified
    recursive : bool, optional
        Whether or not the entire directory tree beneath each specified
        directory will be watched, or just the contents immediately below
        that directory. Default: True
    start : bool, optional
        Start the observer when calling this function. If False, return the
        observer without starting it. Default: True

    Returns
    -------
    :py:class:`watchdog.observers.Observer`, list
        An Observer instance and a list of
        :py:class:`watchdog.observers.api.ObservedWatch` objects
    """
    for d in watch_dirs:
        if not os.path.isdir(d):
            raise ValueError("Path {} is not a directory".format(d))
    watches = []
    observer = Observer()
    for d in watch_dirs:
        for h in handlers:
            watch = observer.schedule(h, d, recursive=recursive)
            watches.append(watch)
    for d in watch_dirs:
        if not os.path.isdir(d):
            raise ValueError("Path {} is not a directory".format(d))
    for d in watch_dirs:
        for h in handlers:
            observer.schedule(h, d, recursive=recursive)
    if start:
        observer.start()
    return observer, watches


class HDF5FileMerger(PatternMatchingEventHandler):
    """
    An :py:class:`watchdog.events.PatternMatchingEventHandler` that receives
    events when files matching *.hdf or *.hdf5 are created and merges
    them into the provided master HDF5 file using ptrepack after the creating
    process is done writing to them

    Parameters
    ----------

    master_file : str
        Path to the master HDF5 file that all discovered HDF5 files will be
        merged into
    create : bool, optional
        Create the master HDF5 file if it does not already exist.
        Default: True
    remove : bool, optional
        Remove any discovered files after merging them. Default: False
    *args, **kwargs
       Passed on to the parent class unmodified
    """

    patterns = ['*.hdf', '*.hdf5']

    def __init__(self, master_file, create=True, remove=False,
                 max_workers=4, *args, **kwargs):
        master_file = os.path.abspath(master_file)
        if not os.path.isfile(master_file) and create:
            hdf = tb.open_file(master_file, 'w')
            hdf.close()
        elif not os.path.isfile(master_file):
            raise ValueError('Master file {} is not a regular '
                             'file'.format(master_file))
        self.master = master_file
        self.merged = []
        self.remove = remove
        self.log = logging.getLogger(__name__)
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        # self.q = Queue()
        super().__init__(*args, **kwargs, ignore_patterns=[master_file])


    def on_created(self, event):
        """
        event.event_type
            'modified' | 'created' | 'moved' | 'deleted'
        event.is_directory
            True | False
        event.src_path
            path/to/observed/file
        """
        print('Created {}'.format(event.src_path))
        if event.is_directory:
            return None
        print('Submitting {}'.format(event.src_path))
        self.executor.submit(self.merge, event.src_path)
        # self.merge(event.src_path)

    def merge(self, path):
        wait_until_released(path)
        self.log.info('Merging file {} into {}'.format(path, self.master))
        print('Merging file {} into {}'.format(path, self.master))
        args = ('ptrepack', '--overwrite-nodes', path, self.master)
        with self.lock:
            print('Lock aquired')
            try:
                subprocess.run(args, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, check=True)
                self.log.info('Completed merge!')
            except subprocess.CalledProcessError as proc:
                msg = "Raised CalledProcessError"
                msg += '\nCMD: {}'.format(proc.cmd)
                msg += '\nEXITCODE: {}'.format(proc.returncode)
                msg += '\nSTDOUT: {}'.format(proc.stdout.decode())
                msg += '\nSTDERR: {}'.format(proc.stderr.decode())
                self.log.error(msg)
                raise
        print('Lock released!')
        self.merged.append(path)
        if self.remove:
            print('Removing path {}'.format(path))
            os.remove(path)

    def stop(self):
        self.executor.shutdown()
