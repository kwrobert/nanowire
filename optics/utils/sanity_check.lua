require 'pl'
S4 = require 'RCWA'

function interp1d(x, y, xval,extrapolate)
    -- A function to compute linear interpolation of 1 dimensional data with extrapolation 
    --
    -- Set default values
    extrapolate = extrapolate or false

    -- If x is outside the interpolation range we need to extrapolate
    min,max = x:minmax()
    if xval < min and extrapolate then
        print('Extrapolate left')
        m = (y[2] - y[1])/(x[2] - x[1])
        val = m*(xval-x[1])+y[1]
        --print('Left: ',y[1])
        --print('Value: ',val)
        --print('Right: ',y[2])
    elseif xval > max and extrapolate then
        print('Extrapolate right')
        m = (y[#y] - y[#y-1])/(x[#x] - x[#x-1])
        val = m*(xval-x[#x])+y[#x]
        --print('Left: ',y[#y-1])
        --print('Value: ',val)
        --print('Right: ',y[#y])
    elseif (xval < min or xval > max) and not extrapolate then
        error("The x value is outside the extrapolation range and extrapolation is set to false")
    else 

        -- First find the first x value that exceeds our desired value
        counter = 1
        while x[counter] < xval do
            counter = counter + 1
        end
        val = (((y[counter]-y[counter-1])*(xval - x[counter-1]))/(x[counter]-x[counter-1]))+y[counter-1]
        --print('Left: ',y[counter-1])
        --print('Value: ',val)
        --print('Right: ',y[counter])
    end
    return val
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
    return freqvec:reverse(),nvec:reverse(),kvec:reverse()
end

function interpolate(conf)
    -- Perform interpolation of conf objects
    
    -- For each section of parameters in the configuration table
    for sect,params in pairs(conf) do
        -- For each parameter, value pair
        for param, value in pairs(params) do
            -- Search the value of the param for the interpolation pattern
            --
            -- Matches $(some_params)s where any character EXCEPT a closing parenthesis
            -- ) can be inside the %( )s. Extracts whats inside interp string
            matches = string.gmatch(value,"%%%(([^)]+)%)s") 
            -- For each parameter name extracted from the interpolation string 
            for match in matches do
                -- Get the value of tha param name. Note this only works if the sought
                -- param is in the same section as the discovered interp string
                -- TODO: Fix this so it searches the conf table for the proper KEY recursively
                repl_val = conf[sect][match] 
                -- Now replace the interpolation string by the actual value of the 
                -- sought parameter 
                rep_par = string.gsub(value,"%%%(([^)]+)%)s",repl_val)
                -- Now store the interpolated value of the original parameter
                conf[sect][param] = rep_par
            end
        end
    end
    return conf
end

function get_epsilon(freq,path)
    freqvec,nvec,kvec= parse_nk(path)
    n = interp1d(freqvec,nvec,freq,true)
    k = interp1d(freqvec,kvec,freq,true)
    eps_r = n*n - k*k
    eps_i = 2*n*k
    return eps_r,eps_i
end

function run_air_sim(p,m)
    sim = S4.NewSimulation()
    sim:SetLattice({p['L'],0},{0,p['L']})
    sim:SetNumG(p['numbasis'])
    sim:SetVerbosity(2)
    sim:SetMaterial('vacuum',{1.0,0.0})
    sim:SetLayer('air1',p['layer_t'],'vacuum')
    sim:SetLayer('air2',p['layer_t'],'vacuum')
    -- Set frequency
    --
    c = 299792458
    f_phys = p['freq'] 
    c_conv = c*1E6
    f_conv = f_phys/c_conv
    print('f_phys = ',f_phys)
    print('wavelength = ',(c/f_phys)*1E6)
    print('f_conv = ',f_conv)
    sim:SetFrequency(f_conv)
    E_mag = 1.0 
    sim:SetExcitationPlanewave({0,0},{-E_mag,0},{-E_mag,90})
    x_samp = p['x_samp'] 
    y_samp = p['y_samp'] 
    z_samp = p['z_samp']
    height = p['layer_t']*3 
    dz = height/z_samp
    for i,z in ipairs(List.range(0,height,dz)) do
        sim:GetFieldPlane(z,{x_samp,y_samp},'FileAppend','test_fields')
    end
end

function run_airito_sim(p,m)
    sim = S4.NewSimulation()
    sim:SetLattice({p['L'],0},{0,p['L']})
    sim:SetNumG(p['numbasis'])
    sim:SetVerbosity(2)
    sim:SetMaterial('vacuum',{1.0,0.0})
    sim:SetLayer('air1',p['layer_t'],'vacuum')
    eps_r,eps_i = get_epsilon(p['freq'],m['ito'])
    sim.SetMaterial('ito',{eps_r,eps_i})
    sim:SetLayer('ito',p['layer_t'],'ito')
    sim:SetLayer('air2',p['layer_t'],'vacuum')
    -- Set frequency
    --
    c = 299792458
    f_phys = p['freq'] 
    c_conv = c*1E6
    f_conv = f_phys/c_conv
    print('f_phys = ',f_phys)
    print('wavelength = ',(c/f_phys)*1E6)
    print('f_conv = ',f_conv)
    sim:SetFrequency(f_conv)
    E_mag = 1.0 
    sim:SetExcitationPlanewave({0,0},{-E_mag,0},{-E_mag,90})
    x_samp = p['x_samp'] 
    y_samp = p['y_samp'] 
    z_samp = p['z_samp']
    height = p['layer_t']*2   
    for z in List.range(0,height,height/z_samp) do
        sim:GetFieldPlane(z,{x_samp,y_samp},'FileAppend','test_fields')
    end
end

function main()
    args = lapp [[
Sanity check for S4 library
    --air  Run air simulation
    --ito  Run air-ito simulation
    --sub  Run air-ito-sub simulation
]]

    params = {freq=5E14,layer_t=.5,L=.25,x_samp=50,y_samp=50,z_samp=600,numbasis=40}
    plane = params['x_samp']/2
    params['plane'] = plane
    mats = {ito='/home/kyle_robertson/schoolwork/gradschool/nanowire/code/NK/008_ITO_nk_Hz.txt',
            gaas='/home/kyle_robertson/schoolwork/gradschool/nanowire/code/NK/006_GaAs_nk_Walker_modified_Hz.txt'}
    dir.makepath('sanity_check_run')
    basedir = path.join(path.currentdir(),'sanity_check_run')
    if args['air'] then
        pth = path.join(basedir,'air_sim')
        dir.makepath(pth)
        path.chdir(pth)
        run_air_sim(params,mats)
        print('Finished air sim')
        path.chdir(basedir)
    end
    if args['ito'] then 
        pth = path.join(basedir,'ito_sim')
        dir.makepath(pth)
        path.chdir(pth)
        run_airito_sim(params,mats)
        print('Finished ito sim')
        path.chdir(basedir)
    end
    if args['sub'] then 
        pth = path.join(basedir,'sub_sim')
        dir.makepath(pth)
        path.chdir(pth)
        run_airitosub_sim(params,mats)
        print('Finished ito sim')
        path.chdir(basedir)
    end
end
main()
-----------------------------------------------------------------------
-- NEED TO PORT ALL CODE BELO HERE
-----------------------------------------------------------------------

    

--function run_airitosub_sim(p,m)
--    sim = S4.New(Lattice=((p['L'],0),(0,p['L'])),NumBasis=p['numbasis'])
--    sim.SetVerbosity(2)
--    sim.SetMaterial(Name='vacuum',Epsilon=complex(1.0,0.0))
--    eps = get_epsilon(p['freq'],m['ito'])
--    sim.SetMaterial(Name='ito',Epsilon=eps)
--    eps = get_epsilon(p['freq'],m['gaas'])
--    sim.SetMaterial(Name='gaas',Epsilon=eps)
--    sim.AddLayer(Name='air1',Thickness=p['layer_t'],Material='vacuum')
--    sim.AddLayer(Name='ito',Thickness=p['layer_t'],Material='ito')
--    sim.AddLayer(Name='gaas',Thickness=p['layer_t'],Material='gaas')
--    sim.AddLayerCopy('air2',Thickness=p['layer_t'],Layer='air1')
--    # Set frequency
--    f_phys = p['freq'] 
--    c_conv = constants.c*1E6
--    f_conv = f_phys/c_conv
--    print('f_phys = ',f_phys)
--    print('wavelength = ',(constants.c/f_phys)*1E6)
--    print('f_conv = ',f_conv)
--    sim.SetFrequency(f_conv)
--    E_mag = 1.0 
--    sim.SetExcitationPlanewave(IncidenceAngles=(0,0),sAmplitude=complex(E_mag,0),
--            pAmplitude=complex(0,0))
--    x_samp = p['x_samp'] 
--    y_samp = p['y_samp'] 
--    z_samp = p['z_samp']
--    height = p['layer_t']*4 
--    for z in np.linspace(0,height,z_samp):
--        sim.GetFieldsOnGrid(z,NumSamples=(x_samp,y_samp),
--                            Format='FileAppend',BaseFilename='test_fields')
--end
--
--def run_airitowiresub_sim(params):
--    sim = S4.New(Lattice=((.63,0),(0,.63)),NumBasis=25)
--    sim.SetVerbosity(2)
--    sim.SetMaterial(Name='vacuum',Epsilon=complex(1.0,0.0))
--    sim.SetMaterial(Name='ito',Epsilon=complex(2.0766028416,0.100037324))
--    sim.SetMaterial(Name='gaas',Epsilon=complex(3.5384,0.0))
--    sim.SetMaterial(Name='cyc',Epsilon=complex(1.53531,1.44205E-6))
--    sim.AddLayer(Name='air1',Thickness=.5,Material='vacuum')
--    sim.AddLayer(Name='ito',Thickness=.5,Material='ito')
--    sim.AddLayer(Name='wire',Thickness=.5,Material='cyc')
--    sim.SetRegionCircle('wire','gaas',(0,0),.2)
--    sim.AddLayer(Name='gaas',Thickness=.5,Material='gaas')
--    # Set frequency
--    f_phys = 3E14 
--    c_conv = constants.c/10E-6
--    f_conv = f_phys/c_conv
--    print('f_phys = ',f_phys)
--    print('wavelength = ',(constants.c/f_phys)*10E6)
--    print('f_conv = ',f_conv)
--    sim.SetFrequency(f_conv)
--    E_mag = 1.0 
--    sim.SetExcitationPlanewave(IncidenceAngles=(0,0),sAmplitude=complex(E_mag,0), pAmplitude=complex(0,E_mag))
--    x_samp = 200 
--    y_samp = 200 
--    z_samp = 200
--    height = 2.0 
--    for z in np.linspace(0,height,z_samp):
--        sim.GetFieldsOnGrid(z,NumSamples=(x_samp,y_samp),
--                            Format='FileAppend',BaseFilename='test_fields')


