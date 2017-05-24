import os
from distutils.spawn import find_executable
import gc3libs
from gc3libs import Application
from gc3libs.cmdline import SessionBasedScript

MODULE_PATH = os.path.expandvars('$HOME/software/nanowire')


class RCWA_App(Application):
    """Runs the lua script that interfaces with the S4 RCWA library"""

    def __init__(self, conf):
        sim_dir = os.path.basename(conf['General']['sim_dir'])
        print('SIM_DIR: %s'%sim_dir)
        conf_file = os.path.join(sim_dir, 'sim_conf.yml')
        script = os.path.join(MODULE_PATH, 'optics/utils/simulator.py')
        python = find_executable('python2')
        args = [python, script, 'sim_conf.yml']
        mem = gc3libs.quantity.Memory('600 MB')
        if conf.variable_thickness:
            # Allocate a minute for each thickness we need to simulate
            num_minutes = len(conf.variable_thickness)
            wtime = gc3libs.quantity.Duration('{} minutes'.format(num_minutes))
        else:
            wtime = gc3libs.quantity.Duration('1 minutes')
        # ANY_OUTPUT means GC3 will collect anything that ends up in the remote
        # execution directory, which is exactly what we want. Any files that
        # get written by the simulator.py script are necessary
        print('ARGS: {}'.format(args))
        super(RCWA_App, self).__init__(arguments=args,
                                       # (local_path, remote_copy_location)
                                       # remote location relative to execution
                                       # directory
                                       inputs=[(conf_file, 'sim_conf.yml')],
                                       outputs=gc3libs.ANY_OUTPUT,
                                       # outputs=[('field_data.E.npz', conf['General']['sim_dir'])],
                                       # The local directory we want the
                                       # results to end up in
                                       output_dir=sim_dir,
                                       requested_cores=1,
                                       requested_memory=mem,
                                       requested_walltime=wtime,
                                       stdout='sim.out', stderr='sim.err')


class SimWrapper(SessionBasedScript):
    """The class that wraps around all the individual simulations. It handles all the
    directory set up and parameter resolution"""

    def __init__(self):
        super(SimWrapper, self).__init__(version='1.0')

    def new_tasks(self, extra):
        apps_to_run = []
        return apps_to_run

if __name__ == '__main__':
    import rcwa_app
    rcwa_app.SimWrapper().run()
