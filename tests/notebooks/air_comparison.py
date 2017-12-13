import postprocess as pp
import argparse as ap
import numpy as np
import matplotlib.pyplot as plt

from utils.config import Config


def main():

    parser = ap.ArgumentParser(description="""Compare absorption curves for air
    and no air sims""")
    parser.add_argument('--no_air',type=str,required=True,help="""Path to no air
    config file""")
    parser.add_argument('--with_air',type=str,required=True,help="""Path to
    with air config file""")
    args = parser.parse_args()

    air_config = Config(path=args.with_air)
    noair_config = Config(path=args.no_air)

    air_crunch = pp.Cruncher(air_config)
    noair_crunch = pp.Cruncher(noair_config)

    air_sims, failed_air = air_crunch.collect_sims()
    noair_sims, failed_noair = noair_crunch.collect_sims()

    air_groups = air_crunch.group_against(['Simulation','params','frequency','value'], 
                             air_config.variable)
    noair_groups = noair_crunch.group_against(['Simulation','params','frequency','value'],
                               noair_config.variable)
    
    diffs = np.zeros((len(air_groups[0]),3))
    freqs = np.zeros(len(air_groups[0]))
    count = 0

    for air_sim, noair_sim in zip(air_groups[0],noair_groups[0]):
        ref,trans,absorb = air_crunch.transmissionData(air_sim)
        nref,ntrans,nabsorb = noair_crunch.transmissionData(noair_sim)
        print(ref)
        print(nref)
        diff = np.abs(np.array([ref,trans,absorb]) -
                      np.array([nref,ntrans,nabsorb]))
        print(diff)
        diffs[count,:] = diff
        freqs[count] = air_sim.conf['Simulation']['params']['frequency']['value']
        count += 1
    print(diffs)
    plt.figure()
    plt.title('Air Comparison')
    plt.ylabel('Absolute Difference')
    plt.xlabel('Frequency (Hz)')
    plt.plot(freqs, diffs[:,0], label='Reflectance')
    plt.plot(freqs, diffs[:,1], label='Transmission')
    plt.plot(freqs, diffs[:,2], label='Absorption')
    plt.legend(loc='best')
    plt.savefig('air_comparison.pdf')
    # plt.show()
    

if __name__ == '__main__':
    main()
