require 'pl'

function lerp(pos1, pos2, perc)
    return (1-perc)*pos1 + perc*pos2 -- Linear Interpolation
end

function arrlerp(x, y, xval)
    -- First find the first x value that exceeds our desired value
    counter = 1
    print(xval)
    while x[counter] < xval do
        counter = counter + 1
    end
    print('Leaving loop')
    print(y[counter])
    print(y)
    return (((y[counter]-y[counter-1])*(xval - x[counter-1]))/(x[counter]-x[counter-1]))+y[counter-1] 
end

function parse_nk(path)
    -- Set path to current input file
    io.input(path)
    -- Skip the header line
    io.read()
    freqvec = List()
    nvec = List()
    kvec = List()
    -- Get the columns
    for freq,n,k in input.fields(3) do
        freqvec:append(freq)
        nvec:append(n)
        kvec:append(k)
    end
    -- input.fields returns iterators. Need to force into lists
    --freqvec = List(freqvec):sorted()
    --nvec = List(nvec):sorted()
    --kvec = List(kvec):sorted()
    return freqvec:reverse(),nvec:reverse(),kvec:reverse()
end


x = {}
y = {}
for i=0,10 do 
    x[i] = i
    y[i] = i*i
end

val = arrlerp(x,y,2.3)
print(val)

f,n,k = parse_nk('/home/kyle_robertson/schoolwork/gradschool/nanowire/code/NK/006_GaAs_nk_Walker_modified_Hz.txt')
print('Freq vec')
print(f)
print('n vec')
print(n)
print('k vec')
print(k)
val = arrlerp(f,n,4e14)
print(val)
--print('Freq, n, k')
--for i,v in ipairs(f) do
--    print(f[i],n[i],k[i])
--end

