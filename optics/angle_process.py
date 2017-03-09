import os
import numpy as np
import matplotlib.pyplot as plt
from utils.config import *
from postprocess import *
import argparse as ap

def main():
    
    parser = ap.ArgumentParser(description="""A wrapper around s4_sim.py to automate parameter
            sweeps, optimization, directory organization, postproccessing, etc.""")
    parser.add_argument('config_file',type=str,help="""Absolute path to the INI file
    specifying how you want this wrapper to behave""")
    args = parser.parse_args()
    if os.path.isfile(args.config_file):
        conf = Config(path=os.path.abspath(args.config_file))
    else:
        raise ValueError("The file you specified does not exist!")

    crnchr = Cruncher(conf)
    sims, failed_sims = crnchr.collect_sims()
    crnchr.group_against(('Simulation','params','frequency','value'))

    #count = 1
    #for group in proc.sim_groups:
    #    print('########## GROUP %i ###########'%count)
    #    for sim in group:
    #        print('FREQ = %f' %
    #              sim.conf[('Simulation','params','frequency','value')])
    # Configure logger
    lfile = os.path.join(conf['General']['base_dir'],'logs/angle_calcs.log')
    logger = configure_logger(level='INFO',
                              console=True,logfile=lfile)
    #print(crnchr.sims)
    #for sim in crnchr.sims:
    #    print('a sim')
    #    crnchr.transmissionData(sim)
    
    gcrnch = Global_Cruncher(conf,sims=crnchr.sims,sim_groups=crnchr.sim_groups)
    jsc_vals = gcrnch.Jsc()

    res = {}
    for i in range(len(gcrnch.sim_groups)):
        sim = gcrnch.sim_groups[i][0]
        thickness = sim.conf[('Layers','NW_AlShell','params','thickness','value')]
        angle = sim.conf[('Simulation','params','polar_angle','value')]
        if not thickness in res:
            res[thickness] = [(angle,jsc_vals[i])]
        else:
            res[thickness].append((angle,jsc_vals[i]))

    print(res)
    for t,vals in res.items():
        vals.sort()


    plt.figure()
    plt.xlabel('Angle in Degrees from Surface Normal')
    plt.ylabel('Fractional Absorption')
    plt.title('Unpassivated')
    for t in sorted(res.keys()):
        vals = res[t]
        print('######## THICKNESS = %f ############'%t)
        angs,jscs = zip(*vals)
        plt.plot(angs,jscs,label="NW Length %.2f"%t)
    plt.legend(loc='best')
    plt.savefig('angle_study_unpassivated.pdf')
    plt.show()
            

if __name__ == '__main__':
    main()
