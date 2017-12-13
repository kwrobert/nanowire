import os
import pytest
from nanowire.optics.simulate import Simulator
from nanowire.optics.utils.config import Config
from nanowire.optics.utils.utils import setup_sim

TESTS_DIR =  os.path.dirname(os.path.realpath(__file__))
@pytest.mark.datafiles(os.path.join(TESTS_DIR, 'test_vacuum_conf.yml'))
@pytest.fixture(scope='module', params=['lpx', 'lpy', 'rhcp', 'lhcp'])
def conf(request):
    """
    This fixture loads the config file and instantiates the config object. It
    is scoped for the entire module, so each config only gets created once and
    is reused across all test functions. It is also parameterized to make
    Config objects with different polarizations of the incident light. So, any
    test that uses this fixture will get run a number of times equal to the
    different number of polarizations shown above 
    """
    path = os.path.join(TESTS_DIR, 'vacuum_test_conf.yml')
    conf_obj = Config(path)
    conf_obj['Simulation']['polarization'] = request.param
    return conf_obj

@pytest.fixture(scope='module')
def simulator(conf):
    """
    This fixture sets up the simulator so we can start calculating things and
    comparing them to analytic results. It is scoped for the entire module, so
    the sim only gets configured once and is resused across all test functions.
    It uses the conf fixture to get the config object.
    """

    sim = Simulator(conf)
    sim = setup_sim(sim)
    sim.get_field()
    return sim
    
def test_real_part(simulator):
    """
    Test that the real part of all the field components matches analytic
    results
    """
    pass

def test_imaginary_part(simulator):
    """
    Test that the imaginary part of all the field components matches
    analytic results 
    """
    pass
