import os
from distutils.spawn import find_executable
from gc3libs import Application
from gc3libs.cmdline import SessionBasedScript 


class RCWA_App(Application):
    """Runs the lua script that interfaces with the S4 RCWA library"""

    def __init__(self,lua_script,config_file):
        script = os.path.basename(lua_script)
        conf = os.path.basename(config_file)
        lua = find_executable('lua')
        args = [lua,script,conf]

        # NOTE: We need to modify the outputs here. The number of outputs is super
        # configurable, and so is their name. We need to extract this from the config
        # file  
        super(RCWA_App, self).__init__(self,arguments=args,inputs=[lua_script,config_file],
                         outputs=['field_data.E'],stdout='sim.log',stderr='sim.err')



class SimWrapper(SessionBasedScript):
    """The class that wraps around all the individual simulations. It handles all the
    directory set up and parameter resolution"""
    def __init__(self):
        super(SimWrapper,self).__init__(version='1.0')

    def new_tasks(self,extra):
        apps_to_run = []
        return apps_to_run 

if __name__ == '__main__':
    import rcwa_app
    rcwa_app.SimWrapper().run()
