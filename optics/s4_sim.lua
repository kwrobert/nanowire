require 'pl'
S4 = require 'RCWA'
----------------------------------------------------
-- Functions 
----------------------------------------------------

function getint(conf,section,parameter)
    val = math.floor(conf[section][parameter])
    return val 
end

function getfloat(conf,section,parameter)
    val = tonumber(conf[section][parameter])
    return val 
end

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

function get_incident_amplitude(freq,period,path)
    -- Set path to current input file
    io.input(path)
    -- Skip the header line
    io.read()
    freqvec = List()
    pvec = List()
    -- Get the columns
    for freqval,p in input.fields(2) do
        freqvec:append(freqval)
        pvec:append(p)
    end
    power = interp1d(freqvec:reverse(),pvec:reverse(),freq,true)
    mu_0 = (4*math.pi)*1E-7 
    c = 299792458
    E = math.sqrt(c*mu_0*power)
    return E
end 

function write_config(conf,path)
    io.output(path)
    for sec,params in pairs(conf) do
        io.write(string.format('[%s]\n',sec))
        for par,val in pairs(params) do
            io.write(string.format('%s = %s\n',par,val))
        end
    end
end

function parse_config(path)
    conf = config.read(path)
    conf = interpolate(conf)
    -- Evaluate expressions
    for par, val in pairs(conf['Parameters']) do
        if type(val) == 'string' then
            if stringx.startswith(val,'`') and stringx.endswith(val,'`') then
                tmp = stringx.strip(val,'`')
                tmp = stringx.join('',{'result = ',tmp})
                f = loadstring(tmp)
                f()
                conf['Parameters'][par] = result
            end
        end
    end
    arr = {conf['Parameters']['nw_height'],conf['Parameters']['substrate_t'],
           conf['Parameters']['ito_t'],conf['Parameters']['air_t']}
    height,n = seq.sum(arr,tonumber)
    conf['Parameters']['total_height'] = height
    write_config(conf,path)
    return conf 
end

function build_sim(conf)
    -- Runs the nanowire simulation
    --
    -- Get desired # of basis terms. Note that this is an upper bound and may not be
    -- the actual number of basis terms used
    num_basis = getint(conf,'Parameters','numbasis')
    -- These lattice vectors can be a little confusing. Everything in S4 is normalized so that speed
    -- of light, vacuum permittivity and permeability are all normalized to 1. This means frequencies
    -- must be specified in units of inverse length. This can be clarified as follows 
    -- 1. Define a base length unit (say 1 micron)
    -- 2. Define your lattice vectors as fractions of that base unit 
    -- 3. Whenever you enter a physical frequency (say in Hz), divide it by the speed of light,
    -- where the speed of light has been converted to your base length unit of choice.
    -- 4. Supply that value to the SetFrequency method
    -- Note: The origin is as the corner of the unit cell
    vec_mag = getfloat(conf,'Parameters','array_period')
    sim = S4.NewSimulation()
    sim:SetLattice({vec_mag,0},{0,vec_mag})
    sim:SetNumG(num_basis)
    -- Configure simulation options
    --
    -- Clean up values
    for key,val in pairs(conf['Simulation']) do
        if type(val) == 'number' or stringx.isdigit(val) then
            conf['Simulation'][key] = math.floor(val)
        elseif val == 'True' then
            conf['Simulation'][key] = true 
        elseif val == 'False' then
            conf['Simulation'][key] = false
        end
    end
    -- Actually set the values
    sim:SetVerbosity(getint(conf,'Simulation','Verbosity'))
    sim:SetLatticeTruncation(conf['Simulation']['LatticeTruncation'])
    if conf['Simulation']['DiscretizedEpsilon'] then 
        sim:UseDiscretizedEpsilon() 
        sim:SetResolution(getint(conf['Simulation']['DiscretizationResolution']))
    end
    if conf['Simulation']['PolarizationDecomposition'] then 
        sim:UsePolarizationDecomposition() 
        if conf['Simulation']['PolarizationBasis'] == 'Jones' then
            sim:UseJonesVectorBasis()
        elseif conf['Simulation']['PolarizationBasis'] == 'Normal' then
            sim:UseNormalVectorBasis()
        else 
            print("Using default vector field")
        end
    end
    if conf['Simulation']['LanczosSmoothing'] then
        sim:UseLanczosSmoothing()
    end
    if conf['Simulation']['SubpixelSmoothing'] then
        sim:UseSubpixelSmoothing()
        sim:SetResolution(getint(conf['Simulation']['DiscretizationResolution']))
    end
    if conf['Simulation']['ConserveMemory'] then
        sim:UseLessMemory()
    end
    
    f_phys = getfloat(conf,'Parameters','frequency')
    for mat,path in pairs(conf['Materials']) do
        eps_r,eps_i = get_epsilon(f_phys,path)
        sim:SetMaterial(mat,{eps_r,eps_i})
    end

    -- Set up material
    sim:SetMaterial('vacuum',{1,0})
    -- Add layers. NOTE!!: Order here DOES MATTER, as incident light will be directed at the FIRST
    -- LAYER SPECIFIED
    sim:AddLayer('air',getfloat(conf,'Parameters','air_t'),'vacuum')
    sim:AddLayer('ito',getfloat(conf,'Parameters','ito_t'),'ITO')
    sim:AddLayer('nanowire_alshell',getfloat(conf,'Parameters','alinp_height'),'Cyclotene')
    -- Add patterning to section with AlInP shell
    core_rad = getfloat(conf,'Parameters','nw_radius')
    shell_rad = core_rad + getfloat(conf,'Parameters','shell_t')
    sim:SetLayerPatternCircle('nanowire_alshell','AlInP',{vec_mag/2,vec_mag/2},shell_rad)
    sim:SetLayerPatternCircle('nanowire_alshell','GaAs',{vec_mag/2,vec_mag/2},core_rad)
    -- Si layer and patterning 
    sim:AddLayer('nanowire_sishell',getfloat(conf,'Parameters','sio2_height'),'Cyclotene')
    -- Add patterning to layer with SiO2 shell 
    sim:SetLayerPatternCircle('nanowire_sishell','SiO2',{vec_mag/2,vec_mag/2},shell_rad)
    sim:SetLayerPatternCircle('nanowire_sishell','GaAs',{vec_mag/2,vec_mag/2},core_rad)
    -- Substrate layer and air transmission region
    sim:AddLayer('substrate',getfloat(conf,'Parameters','substrate_t'),'GaAs')
    --sim:AddLayerCopy('air_below',Thickness=conf.getfloat('Parameters','air_t'),Layer='air') 

    c = 299792458
    print(string.format('Physical Frequency = %E',f_phys))
    c_conv = c/getfloat(conf,"General","base_unit")
    f_conv = f_phys/c_conv
    print(string.format('Converted Frequency = %f',f_conv))
    sim:SetFrequency(f_conv)

    -- Define incident light. Normally incident with frequency dependent amplitude
    E_mag = get_incident_amplitude(f_phys,vec_mag,conf["General"]["input_power"])
    -- To define circularly polarized light, basically just stick a j (imaginary number) in front of
    -- one of your components. The handedness is determined by the component you stick the j in front
    -- of. From POV of source, looking away from source toward direction of propagation, right handed
    -- circular polarization has j in front of y component. Magnitudes are the same. This means
    -- E-field vector rotates clockwise when observed from POV of source. Left handed =
    -- counterclockwise. 
    -- In S4, if indicent angles are 0, p-polarization is along x-axis. The minus sign on front of the 
    -- x magnitude is just to get things to look like Anna's simulations.
    sim:SetExcitationPlanewave({0,0},{-E_mag,0},{-E_mag,90})
    --sim.OutputLayerPatternPostscript(Layer='ito',Filename='out.ps')
    --sim.OutputStructurePOVRay(Filename='out.pov')
    output_file = conf["General"]["base_name"]
    glob = stringx.join('',{output_file,".*"})
    cwd = path:currentdir()
    existing_files = dir.getfiles(cwd,glob)
    if existing_files[1] then
        for key,afile in pairs(existing_files) do
            file.delete(afile)
        end
    end
    x_samp = getint(conf,'General','x_samples')
    y_samp = getint(conf,'General','y_samples')
    z_samp = getint(conf,'General','z_samples')
    height = getfloat(conf,'Parameters','total_height') 
    dz = height/z_samp
    zvec = List.range(0,height,dz)
    for i,z in ipairs(zvec) do 
        sim:GetFieldPlane(z,{x_samp,y_samp},'FileAppend',output_file)
    end
end
----------------------------------------------------
-- Main program 
----------------------------------------------------
function main() 
    args = lapp [[
A program that uses the S4 RCWA simulation library to simulate 
the optical properties of a single nanowire in a square lattice
    <config_file> (string) Absolute path to simulation INI file
]]
    if not path.isfile(args['config_file']) then
        error('The config file specified does not exist or is not a file')
    end
    conf = parse_config(args['config_file'])
    build_sim(conf)
end

main()
