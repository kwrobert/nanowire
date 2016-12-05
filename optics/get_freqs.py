import glob
import os 

def get_freq(path):
    with open(path,'r') as f:
        for i in range(0,10):
            line = f.readline()
    dat = line.split()
    freq = dat[3]
    return freq

def main():
    path = './comsol_p250_r75'

    files = glob.glob(os.path.join(path,'*.txt'))
    freqs = []
    for p in files:
        print(p)
        freq = get_freq(p)
        freqs.append(freq)
    st = ','.join(freqs)
    print(st)

if __name__ == '__main__':
    main()
