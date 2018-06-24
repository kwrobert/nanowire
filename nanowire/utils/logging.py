import logging
import traceback
from functools import wraps


def add_logger(logger=None):
    logger = logger if logger is not None else logging.getLogger()

    def log_decorator(f):
        @wraps(f)
        def inner_dec(*args, **kwargs):
            g = f.__globals__  # use f.func_globals for py < 2.6
            sentinel = object()

            oldvalue = g.get('log', sentinel)
            g['log'] = logger

            try:
                res = f(*args, **kwargs)
            finally:
                if oldvalue is sentinel:
                    del g['log']
                else:
                    g['log'] = oldvalue
            return res
        return inner_dec
    return log_decorator


class LogExceptions:
    def __init__(self, callable):
        self.__callable = callable

    def __call__(self, *args, **kwargs):
        try:
            result = self.__callable(*args, **kwargs)
        except Exception as e:
            # Here we add some debugging help.
            log = logging.getLogger(__name__)
            log.error(traceback.format_exc())
            # Re-raise the original exception so the Pool worker can clean up.
            # This kills the parent process if it calls .get or .wait on the
            # AsyncResult object returned by apply_async
            # raise
        # It was fine, give a normal answer
        return result


# class LoggingPool(Pool):
#     def apply_async(self, func, args=(), kwds={}, callback=None):
#         return Pool.apply_async(self, LogExceptions(func), args, kwds,
#                                 callback)


class StreamToLogger:
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


class IdFilter(logging.Filter):
    """
    A filter to either only pass log records with a matching ID, or reject all
    log records with an ID attribute. This is configurable via a kwarg to the
    init method
    """

    def __init__(self, ID=None, name="", reject=False):
        super(IdFilter, self).__init__(name=name)
        self.ID = ID
        if reject:
            self.filter = self.reject_filter
        else:
            self.filter = self.pass_filter

    def pass_filter(self, record):
        if not hasattr(record, 'ID'):
            return 0
        if record.ID == self.ID:
            return 1
        else:
            return 0

    def reject_filter(self, record):
        if hasattr(record, 'ID'):
            return 0
        else:
            return 1


def configure_logger(level='info', name=None, console=False, logfile=None,
                     propagate=True):
    """
    Creates a logger providing some arguments to make it more configurable.

    :param str name:
        Name of logger to be created. Defaults to the root logger
    :param str level:
        The log level of the logger, defaults to INFO. One of: ['debug', 'info',
        'warning', 'error', 'critical']
    :param bool console:
        Add a stream handler to send messages to the console. Generally
        only necessary for the root logger.
    :param str logfile:
        Path to a file. If specified, will create a simple file handler and send
        messages to the specified file. The parent dirs to the location will
        be created automatically if they don't already exist.
    """

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
        log_dir, logfile = os.path.split(os.path.expandvars(logfile))
        # Set up file handler
        try:
            os.makedirs(log_dir)
        except OSError:
            # Log dir already exists
            pass
        output_file = os.path.join(log_dir, logfile)
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
