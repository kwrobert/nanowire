import argparse as ap
import os
import os.path as osp 

def main():
    parser = ap.ArgumentParser(description="""Find simulations that failed to produce the necessary
    data files""")
    parser.add_argument('base',type=str,help="""Base of the entire tree for the sweep""")
    args = parser.parse_args()

    base = osp.abspath(args.base)
    if not osp.isdir(base):
        raise ValueError('Base directory doesnt exist')
    
    for root,dirs,files in os.walk(base):
        if not dirs:
            if 'sim_conf.ini' in files and not 'field_data.E' in files:
                print('Failed simulation at %s'%root)

if __name__ == '__main__':
    main()
