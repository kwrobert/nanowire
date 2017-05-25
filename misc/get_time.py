import re
from datetime import datetime
import argparse as ap


def main():
    parser = ap.ArgumentParser(description="""Gets total run time of a sim by
    checking the first and last line of the sim_wrapper.log file""")
    parser.add_argument('log', type=str, help="Location of log file")
    args = parser.parse_args()

    with open(args.log,'r') as lfile:
        lines = lfile.readlines()

    start_str = lines[0]
    end_str = lines[-1]

    start_time = ' '.join(start_str.split()[0:3])
    end_time = ' '.join(end_str.split()[0:3])

    print(start_time)
    print(end_time)


    start = datetime.strptime(start_time, '%m/%d/%Y %I:%M:%S %p')
    end = datetime.strptime(end_time, '%m/%d/%Y %I:%M:%S %p')
    print(end-start)
        
if __name__ == '__main__':
    main()
