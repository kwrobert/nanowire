import numpy as np

def bad_method(x_samp,y_samp,z_samp,data):
    zs = np.unique(data[:,2])
    xs = np.unique(data[:,0])
    rowlist = []
    for z in zs:
        for x in xs:
            for row in data:
                if row[0] == x and row[2] == z:
                    rowlist.append(row)
    new_data = np.vstack(rowlist)
    return new_data

def slicing_method(x_samples,y_samples,z_samples,data):
    new_data = np.zeros_like(data)
    tot = 0
    for j in range(z_samples):
        for i in range(x_samples):
            if j == 0:
                start = i+j*z_samples
                end = x_samples*y_samples+j*z_samples
            else:
                start = i+j*z_samples+2*j
                end = x_samples*y_samples+j*z_samples+2*j
            step = x_samples
            new_data[tot*y_samples:tot*y_samples+y_samples,:] = data[start:end:step,:]
            tot += 1
    return new_data

def best_method(x_samples,y_samples,z_samples,data):
    # (No need to copy if you don't want to keep the given_dat ordering)
    new_data = np.copy(data).reshape(( z_samples, y_samples, x_samples, 3))
    # swap the "y" and "x" axes
    new_data = np.swapaxes(data, 1,2)
    # back to 2-D array
    new_data = data.reshape((x_samples*y_samples*z_samples,3))
    return new_data

def main():
    # Generate given dat with its ordering
    x_samples = 2
    y_samples = 3
    z_samples = 4
    given_dat = np.zeros(((x_samples*y_samples*z_samples),3))
    row_ind = 0
    for z in range(z_samples):
        for y in range(y_samples):
            for x in range(x_samples):
                row = [x+.1,y+.2,z+.3]
                given_dat[row_ind,:] = row
                row_ind += 1
    for row in given_dat:
        print(row)
    
    # Generate data with desired ordering
    desired_dat = np.zeros(((x_samples*y_samples*z_samples),3))
    row_ind = 0
    for z in range(z_samples):
        for x in range(x_samples):
            for y in range(y_samples):
                row = [x+.1,y+.2,z+.3]
                desired_dat[row_ind,:] = row
                row_ind += 1
    for row in desired_dat:
        print(row)

    # Show that my methods does with I want, but its slow\
    fix = bad_method(x_samples,y_samples,z_samples,given_dat)    
    print('Unreversed data')
    print(given_dat)
    print('Reversed Data')
    print(fix)
    # If it didn't work this will throw an exception
    assert(np.array_equal(desired_dat,fix))
    # Show that the slicing method it better
    better_fix = slicing_method(x_samples,y_samples,z_samples,given_dat)
    assert(np.array_equal(better_fix,desired_dat))
    # Best fix
    best_fix = best_method(x_samples,y_samples,z_samples,given_dat)
    assert(np.array_equal(best_fix,desired_dat))

if __name__ == '__main__':
    main()
