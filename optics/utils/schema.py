import copy
import hashlib

from functools import wraps
from contextlib import contextmanager
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, PickleType,create_engine
from sqlalchemy.orm import sessionmaker


__DB_ENGINE__ = None
__SESSION_FACTORY__ = None
Base = declarative_base()

def setup_db(name,verbose=False):
    """Sets up the module-scope database engine and the session factory"""
    global __DB_ENGINE__
    global __SESSION_FACTORY__
    if not __DB_ENGINE__ and not __SESSION_FACTORY__:
        __DB_ENGINE__ = create_engine('sqlite:///{}'.format(name),echo=verbose)
        __SESSION_FACTORY__ = sessionmaker(bind=__DB_ENGINE__)
        Base.metadata.create_all(__DB_ENGINE__)
    else:
        raise RuntimeError('DB Engine already set')

@contextmanager
def session_scope():
    """Provide a transactional scope around a series of operations."""
    session = __SESSION_FACTORY__()
    try:
        yield session
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()

#def dbconnect(func):
#    @wraps(func)
#    def wrapper(*args,**kwargs):
#        session = __SESSION_FACTORY__() 
#        print('Setting up session scope')
#        print('Here is session in wrapper')
#        print(session)
#        return func(*args,**kwargs)
#    return wrapper

class Simulation(Base):

    __tablename__ = 'master'
    
    id = Column(String,primary_key = True)
    conf = Column(PickleType)

    def __init__(self,conf):
        conf_id = make_hash(conf)
        super(Simulation,self).__init__(conf=conf,id=conf_id)

def make_hash(o):
    """Makes a hash from a dictionary, list, tuple or set to any level, that contains
    only other hashable types (including any lists, tuples, sets, and
    dictionaries)."""

    if isinstance(o, (set, tuple, list)):

        return tuple([make_hash(e) for e in o])

    elif not isinstance(o, dict):
        buf = repr(o).encode('utf-8')
        return hashlib.md5(buf).hexdigest()

    new_o = copy.deepcopy(o)
    for k, v in new_o.items():
        new_o[k] = make_hash(v)
    out = repr(tuple(frozenset(sorted(new_o.items())))).encode('utf-8')
    return hashlib.md5(out).hexdigest()

def run(conf,log):
    """The main run methods that decides what kind of simulation to run based on the
    provided config object"""

    basedir = conf['General']['base_dir']
    logdir = os.path.join(basedir,'logs')
    # Configure logger
    logger = configure_logger(log,'sim_wrapper',logdir,'sim_wrapper.log')
    # Just a simple single simulation
    if conf.fixed and not conf.variable and not conf.sorting and not conf.optimized:
        logger.debug('Entering single sim function from main')
        job = make_single_sim(conf)
        #run_sim(job)
    # If we have variable params, do a parameter sweep
    elif conf.variable and not conf.optimized:
        # If we have sorting parameters we need to make the nodes first
        if conf.sorting:
            logger.debug('Creating nodes and leaves')
            jobs = make_nodes(conf)
        else:
            logger.debug('Creating leaves')
            # make_leaves expects a list of tuples with contents (jobpath,jobconfobject)
            jobs = make_leaves([(basedir,conf)])
        execute_jobs(conf,jobs)
    # Looks like we need to run an optimization
    elif conf.optimized and not conf.sorting:
        logger.debug('Entering optimization function from main')
        run_optimization(conf)
    else:
        logger.error('Unsupported configuration for a simulation run. Not a '
        'single sim, sweep, or optimization. Make sure your sweeps are '
        'configured correctly, and if you are running an optimization '
        'make sure you do not have any sorting parameters specified')

def pre_check(conf_path,conf):
    """Checks conf file for invalid values"""
    if not os.path.isfile(conf['General']['sim_script']):
        print('You need to change the sim_script entry in the [General] section of your config \
        file')
        quit()

    # Warn user if they are about to dump a bunch of simulation data and directories into a
    # directory that already exists
    base = conf["General"]["basedir"]
    if os.path.isdir(base):
        print('WARNING!!! You are about to start a simulation in a directory that already exists')
        print('WARNING!!! This will dump a whole bunch of crap into that directory and possibly')
        print('WARNING!!! overwrite old simulation data.')
        ans = input('Would you like to continue? CTRL-C to exit, any other key to continue: ')
    # Copy provided config file to basedir
    try:
        os.makedirs(base)
    except OSError:
        pass
    fname = os.path.basename(conf_path)
    new_path = os.path.join(base,fname)
    try:
        shutil.copy(conf_path,new_path)
    except shutil.SameFileError:
        pass
    # TODO: Add checks between specified x,y,z samples and plane vals in plotting section

#def main():
#
#    parser = ap.ArgumentParser(description="""A wrapper around s4_sim.py to automate parameter
#            sweeps, optimization, directory organization, postproccessing, etc.""")
#    parser.add_argument('config_file',type=str,help="""Absolute path to the INI file
#    specifying how you want this wrapper to behave""")
#    parser.add_argument('--log_level',type=str,default='info',choices=['debug','info','warning','error','critical'],
#                        help="""Logging level for the run""")
#    args = parser.parse_args()
#
#    if os.path.isfile(args.config_file):
#        conf = Config(path=os.path.abspath(args.config_file))
#    else:
#        print("\n The file you specified does not exist! \n")
#        quit()
#
#    #pre_check(os.path.abspath(args.config_file),conf)
#    run(conf,args.log_level)

def db_test():
    with session_scope() as session:
        print(session)

def main():
    print('INSIDE MAIN')
    setup_db('new_test.db')
    db_test()
    #Session = sessionmaker(bind=engine)
    #session = Session()
    #conf = {'some_stuff':'wtih a value'}
    #sim = Simulation(conf)
    #print(sim.__table__)
    #print(sim.id)
    #session.add(sim)
    #session.commit()

if __name__ == '__main__':
    main()
