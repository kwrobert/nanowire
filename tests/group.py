import postprocess as pp
import os
import argparse as ap
import time
from utils.config import Config


def main():
    parser = ap.ArgumentParser(description="""A wrapper around s4_sim.py to automate parameter
            sweeps, optimization, directory organization, postproccessing, etc.""")
    parser.add_argument('config_file',type=str,help="""Absolute path to the INI file
    specifying how you want this wrapper to behave""")
    parser.add_argument('-nc','--no_crunch',action="store_true",default=False,help="""Do not perform crunching
            operations. Useful when data has already been crunched but new plots need to be
            generated""")
    parser.add_argument('-ngc','--no_gcrunch',action="store_true",default=False,help="""Do not
            perform global crunching operations. Useful when data has already been crunched but new plots need to be
            generated""")
    parser.add_argument('-np','--no_plot',action="store_true",default=False,help="""Do not perform plotting
            operations. Useful when you only want to crunch your data without plotting""")
    parser.add_argument('-ngp','--no_gplot',action="store_true",default=False,help="""Do not perform global plotting
            operations. Useful when you only want to crunch your data without plotting""")
    parser.add_argument('--log_level',type=str,default='info',choices=['debug','info','warning','error','critical'],
                        help="""Logging level for the run""")
    parser.add_argument('--filter_by',nargs='*',help="""List of parameters you wish to filter by,
            specified like: p1:v1,v2,v3 p2:v1,v2,v3""")
    args = parser.parse_args()
    if os.path.isfile(args.config_file):
        conf = Config(path=os.path.abspath(args.config_file))
    else:
        print("\n The file you specified does not exist! \n")
        quit()

    proc = pp.Processor(conf)
    parkey = ('Simulation','params','array_period','value')
    sortkey = ('Simulation','params','frequency','value')
    proc.collect_sims()
    print('#####################')
    print('Grouping by')
    print('#####################')
    proc.group_by(parkey,sort_key=sortkey)
    count = 1
    for group in proc.sim_groups:
        print('Group %d'%count)
        for sim in group:
            print('Param val = %f'%sim.conf[parkey])
            print('Sorted Param val = %f'%sim.conf[sortkey])
        count += 1
    print('#####################')
    print('Grouping against')
    print('#####################')
    count = 1
    sortkey = ('Simulation','params','array_period','value')
    sortkey2 = ('Layers','NW_AlShell','params','core_radius','value')
    parkey = ('Simulation','params','frequency','value')
    s = time.time()
    proc.group_against(parkey)
    e = time.time()
    print('Duration of group against = %f'%(e-s))
    for group in proc.sim_groups:
        print('Group %d'%count)
        for sim in group:
            print('Param val = %f'%sim.conf[parkey])
            print('Sorted Param val = %f'%sim.conf[sortkey])
            print('Other Param val = %f'%sim.conf[sortkey2])
        count += 1


if __name__ == '__main__':
    main()
