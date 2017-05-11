import sys
import os
import hashlib
import logging
import tempfile as tmp

from collections import OrderedDict
from contextlib import contextmanager


@contextmanager
def tempfile(suffix='', dir=None, npz=True):
    """ Context for temporary file.

    Will find a free temporary filename upon entering
    and will try to delete the file on leaving, even in case of an exception.

    Parameters
    ----------
    suffix : string
        optional file suffix
    dir : string
        optional directory to save temporary file in
    """

    tf = tmp.NamedTemporaryFile(delete=False, suffix=suffix, dir=dir)
    tf.file.close()
    try:
        yield tf.name
    finally:
        try:
            if npz:
                os.remove(tf.name)
                os.remove(tf.name+'.npz')
            else:
                os.remove(tf.name)
        except OSError as e:
            if e.errno == 2:
                pass
            else:
                raise

@contextmanager
def open_atomic(filepath, npz=True):
    """Get a temporary file path in the same directory as filepath. The temp
    file is used as a placeholder for filepath to make atomic write operations
    possible. The file will not be moved to destination in case of an exception.

    Parameters
    ----------
    filepath : string
        the actual filepath we wish to write to
    """
    with tempfile(npz=npz, dir=os.path.dirname(os.path.abspath(filepath))) as tmppath:
        yield tmppath
        if npz:
            os.rename(tmppath+'.npz', filepath+'.npz')
        else:
            os.rename(tmppath, filepath)

class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        # create a flush method so things can be flushed when
        # the system wants to. Not sure if simply 'printing'
        # sys.stderr is the correct way to do it, but it seemed
        # to work properly for me.
        self.level(sys.stderr)


def configure_logger(level='info', name=None, console=False, logfile=None,
                     propagate=True):
    """Creates a logger providing some arguments to make it more configurable.
       name: Name of logger to be created. Defaults to the root logger
       level: The log level of the logger, defaults to INFO
       console: Add a stream handler to send messages to the console. Generally
                only necessary for the root logger.
       logfile: If specified, will create a simple file handler and send
                messages to the specified file. The parent dirs to the location will
                be created automatically if they don't already exist. """
    # Get numeric level safely
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % level)
    # Set formatting
    formatter = logging.Formatter('%(asctime)s [%(module)s:%(name)s:%(levelname)s] - %(message)s',datefmt='%m/%d/%Y %I:%M:%S %p')
    # Create logger
    if name:
        logger = logging.getLogger(name)
    else:
        logger = logging.getLogger()
    if not propagate:
        logger.propagate = False
    logger.setLevel(numeric_level)
    if logfile:
        log_dir,logfile = os.path.split(logfile)
        # Set up file handler
        try:
            os.makedirs(log_dir)
        except OSError:
            # Log dir already exists
            pass
        output_file = os.path.join(log_dir,logfile)
        fhandler = logging.FileHandler(output_file)
        fhandler.setFormatter(formatter)
        logger.addHandler(fhandler)
    # Create console handler
    if console:
        ch = logging.StreamHandler()
        ch.setLevel(numeric_level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

def make_hash(o):
    """Makes a hash from the dict representing the Simulation config. It is
    consistent across runs and handles arbitrary nesting. Right now it ignores
    any settings in the General section because those aren't really important
    when it comes to differentiating simulations"""
    if isinstance(o, (set, tuple, list)):
        return tuple([make_hash(e) for e in o])
    elif not isinstance(o, dict):
        buf = repr(o).encode('utf-8')
        return hashlib.md5(buf).hexdigest()
    new_o = OrderedDict()
    for k, v in sorted(o.items(),key=lambda tup: tup[0]):
        if k == 'General':
            continue
        else:
            new_o[k] = make_hash(v)
    out = repr(tuple(frozenset(sorted(new_o.items())))).encode('utf-8')
    return hashlib.md5(out).hexdigest()

def cmp_dicts(d1,d2):
    """Recursively compares two dictionaries"""
    # First test the keys
    for k1 in d1.keys():
        if k1 not in d2:
            return False
    for k2 in d2.keys():
        if k2 not in d1:
            return False
    # Now we need to test the contents recursively. We store the results of
    # each recursive comparison in a list and assert that they all must be True
    # at the end
    comps = []
    for k1,v1 in d1.items():
        v2 = d2[k1]
        if isinstance(v1,dict) and isinstance(v2,dict):
            comps.append(cmp_dicts(v1,v2))
        else:
            if v1 != v2:
                return False
    return all(comps)

#  def configure_logger(level,logger_name,log_dir,logfile):
#      # Get numeric level safely
#      numeric_level = getattr(logging, level.upper(), None)
#      if not isinstance(numeric_level, int):
#          raise ValueError('Invalid log level: %s' % level)
#      # Set formatting
#      formatter = logging.Formatter('%(asctime)s [%(levelname)s] - %(message)s',datefmt='%m/%d/%Y %I:%M:%S %p')
#      # Get logger with name
#      logger = logging.getLogger(logger_name)
#      logger.setLevel(numeric_level)
#      # Set up file handler
#      try:
#          os.makedirs(log_dir)
#      except OSError:
#          # Log dir already exists
#          pass
#      output_file = os.path.join(log_dir,logfile)
#      #out = open(output_file,'a')
#      fhandler = logging.FileHandler(output_file)
#      fhandler.setFormatter(formatter)
#      logger.addHandler(fhandler)
#      #sl = StreamToLogger(logger,logging.INFO)
#      #sys.stdout = sl
#      #sl = StreamToLogger(logger,logging.ERROR)
#      #sys.stderr = sl
#      ## Duplicate this new log file descriptor to system stdout so we can
#      ## intercept output from external S4 C library
#      ## TODO: Make this go through the logger instead of directly to the file
#      ## right now entries aren't properly formatted
#      #os.dup2(out.fileno(),1)
#      return logger
