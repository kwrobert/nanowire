local pl = require('pl.import_into')()
--require('pl')
local S4 = require('RCWA')
----------------------------------------------------
-- Functions 
----------------------------------------------------

pl.class.Simulator()

function Simulator:_init(conf)
    self.sim = S4.NewSimulation()
    self.conf = self:parse_config(conf)
end

function Simulator:getint(section,parameter)
    local val = math.floor(self.conf[section][parameter])
    return val 
end

function Simulator:getfloat(section,parameter)
    local val = tonumber(self.conf[section][parameter])
    return val 
end

function Simulator:getbool(section,parameter)
    local val = self.conf[section][parameter] 
    if val == 'True' or val == 'true' or val == 'yes' then
        return true
    else
        return false
    end
end

function Simulator:interp1d(x, y, xval,extrapolate)
    -- A function to compute linear interpolation of 1 dimensional data with extrapolation 
    --
    -- Set default values
    local extrapolate = extrapolate or false

    -- If x is outside the interpolation range we need to extrapolate
    local min,max = x:minmax()
    if xval < min and extrapolate then
        print('Extrapolate left')
        local m = (y[2] - y[1])/(x[2] - x[1])
        val = m*(xval-x[1])+y[1]
        --print('Left: ',y[1])
        --print('Value: ',val)
        --print('Right: ',y[2])
    elseif xval > max and extrapolate then
        print('Extrapolate right')
        local m = (y[#y] - y[#y-1])/(x[#x] - x[#x-1])
        val = m*(xval-x[#x])+y[#x]
        --print('Left: ',y[#y-1])
        --print('Value: ',val)
        --print('Right: ',y[#y])
    elseif (xval < min or xval > max) and not extrapolate then
        error("The x value is outside the extrapolation range and extrapolation is set to false")
    else 

        -- First find the first x value that exceeds our desired value
        local counter = 1
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

function Simulator:parse_nk(path)
    -- Set path to current input file
    io.input(path)
    -- Skip the header line
    io.read()
    local freqvec = pl.List()
    local nvec = pl.List()
    local kvec = pl.List()
    -- Get the columns
    for freq,n,k in pl.input.fields(3) do
        freqvec:append(freq)
        nvec:append(n)
        kvec:append(k)
    end
    return freqvec:reverse(),nvec:reverse(),kvec:reverse()
end

function Simulator:search_conf(conf,str)
    -- Search for a dot in str
    local ind = pl.stringx.lfind(str,'.')
    if ind then
        -- Split on the dot to get desired section and par
        local dat = pl.stringx.split(str,'.')
        local sect = dat[1]
        local par = dat[2]
        return conf[sect][par]
    else
        pl.utils.quit("You need to specify both the section and the parameter when using interpolation: section.param")
    end
end

function Simulator:interpolate(conf)
    -- Perform interpolation of conf objects
    
    -- For each section of parameters in the configuration table
    for sect,params in pairs(conf) do
        -- For each parameter, value pair
        for param, value in pairs(params) do
            -- Search the value of the param for the interpolation pattern
            --
            -- Matches $(some_params)s where any character EXCEPT a closing parenthesis
            -- ) can be inside the %( )s. Extracts whats inside interp string
            local matches = string.gmatch(value,"%%%(([^)]+)%)s") 
            -- For each parameter name extracted from the interpolation string 
            for match in matches do
                -- Get the value of tha param name. 
                local repl_val = self:search_conf(conf,match)
                print(repl_val)
                -- Now replace the interpolation string by the actual value of the 
                -- sought parameter 
                local rep_par = string.gsub(value,"%%%(([^)]+)%)s",repl_val)
                -- Now store the interpolated value of the original parameter
                conf[sect][param] = rep_par
            end
        end
    end
    return conf
end

function Simulator:get_epsilon(freq,path)
    local freqvec,nvec,kvec = self:parse_nk(path)
    local n = self:interp1d(freqvec,nvec,freq,true)
    local k = self:interp1d(freqvec,kvec,freq,true)
    print(n)
    local eps_r = n*n - k*k
    local eps_i = 2*n*k
    return eps_r,eps_i
end

function Simulator:get_incident_amplitude(freq,period,path)
    -- Set path to current input file
    io.input(path)
    -- Skip the header line
    io.read()
    local freqvec = pl.List()
    local pvec = pl.List()
    -- Get the columns
    for freqval,p in pl.input.fields(2) do
        freqvec:append(freqval)
        pvec:append(p)
    end
    local power = self:interp1d(freqvec:reverse(),pvec:reverse(),freq,true)
    local mu_0 = (4*math.pi)*1E-7 
    local c = 299792458
    local E = math.sqrt(c*mu_0*power)
    return E
end 

function Simulator:write_config(conf,path)
    io.output(path)
    for sec,params in pairs(conf) do
        io.write(string.format('[%s]\n',sec))
        for par,val in pairs(params) do
            io.write(string.format('%s = %s\n',par,val))
        end
    end
    io.flush()
end

function Simulator:parse_config(path)
    local conf = pl.config.read(path)
    conf = self:interpolate(conf)
    -- Evaluate expressions
    for sect, pars in pairs(conf) do
        for par, val in pairs(pars) do
            if type(val) == 'string' then
                if pl.stringx.startswith(val,'`') and pl.stringx.endswith(val,'`') then
                    tmp = pl.stringx.strip(val,'`')
                    tmp = pl.stringx.join('',{'result = ',tmp})
                    -- This loads the lua statement contained in tmp and evaluates it, making result
                    -- available in the current scope
                    f = load(tmp)
                    f()
                    conf[sect][par] = result
                end
            end
        end
    end
    local arr = {conf['Parameters']['nw_height'],conf['Parameters']['substrate_t'],
           conf['Parameters']['ito_t'],conf['Parameters']['air_t']}
    local height,n = pl.seq.sum(arr,tonumber)
    conf['Parameters']['total_height'] = height
    self:write_config(conf,path)
    return conf 
end

function Simulator:set_lattice()
    -- These lattice vectors can be a little confusing. Everything in S4 is normalized so that speed
    -- of light, vacuum permittivity and permeability are all normalized to 1. This means frequencies
    -- must be specified in units of inverse length. This can be clarified as follows 
    -- 1. Define a base length unit (say 1 micron)
    -- 2. Define your lattice vectors as fractions of that base unit 
    -- 3. Whenever you enter a physical frequency (say in Hz), divide it by the speed of light,
    -- where the speed of light has been converted to your base length unit of choice.
    -- 4. Supply that value to the SetFrequency method
    -- Note: The origin is as the corner of the unit cell
    local vec_mag = self:getfloat('Parameters','array_period')
    self.sim:SetLattice({vec_mag,0},{0,vec_mag})
end

function Simulator:set_basis(num_basis)
    -- Set desired # of basis terms. Note that this is an upper bound and may not be
    -- the actual number of basis terms used
    local num = tonumber(num_basis)
    self.sim:SetNumG(num)
end

function Simulator:configure()
    -- Configure simulation options
    --
    -- Clean up values
    for key,val in pairs(self.conf['Simulation']) do
        if type(val) == 'number' or pl.stringx.isdigit(val) then
            self.conf['Simulation'][key] = math.floor(val)
        elseif val == 'True' then
            self.conf['Simulation'][key] = true 
        elseif val == 'False' then
            self.conf['Simulation'][key] = false
        end
    end
    -- Actually set the values
    self.sim:SetVerbosity(self:getint('Simulation','Verbosity'))
    self.sim:SetLatticeTruncation(self.conf['Simulation']['LatticeTruncation'])
    if self.conf['Simulation']['DiscretizedEpsilon'] then 
        self.sim:UseDiscretizedEpsilon() 
        self.sim:SetResolution(self:getint('Simulation','DiscretizationResolution'))
    end
    if self.conf['Simulation']['PolarizationDecomposition'] then 
        self.sim:UsePolarizationDecomposition() 
        if self.conf['Simulation']['PolarizationBasis'] == 'Jones' then
            self.sim:UseJonesVectorBasis()
        elseif self.conf['Simulation']['PolarizationBasis'] == 'Normal' then
            self.sim:UseNormalVectorBasis()
        else 
            print("Using default vector field")
        end                                    
    end
    if self.conf['Simulation']['LanczosSmoothing'] then
        self.sim:UseLanczosSmoothing()
    end
    if self.conf['Simulation']['SubpixelSmoothing'] then
        self.sim:UseSubpixelSmoothing()
        self.sim:SetResolution(self:getint('Simulation','DiscretizationResolution'))
    end
    if self.conf['Simulation']['ConserveMemory'] then
        self.sim:UseLessMemory()
    end
end

function Simulator:build_device()
    -- Builds out the device geometry and materials 
    local f_phys = self:getfloat('Parameters','frequency')
    local vec_mag = self:getfloat('Parameters','array_period')
    for mat,path in pairs(self.conf['Materials']) do
        eps_r,eps_i = self:get_epsilon(f_phys,path)
        self.sim:SetMaterial(mat,{eps_r,eps_i})
    end

    -- Set up material
    self.sim:SetMaterial('vacuum',{1,0})
    -- Add layers. NOTE!!: Order here DOES MATTER, as incident light will be directed at the FIRST
    -- LAYER SPECIFIED
    self.sim:AddLayer('air',self:getfloat('Parameters','air_t'),'vacuum')
    self.sim:AddLayer('ito',self:getfloat('Parameters','ito_t'),'ITO')
    self.sim:AddLayer('nanowire_alshell',self:getfloat('Parameters','alinp_height'),'Cyclotene')
    -- Add patterning to section with AlInP shell
    core_rad = self:getfloat('Parameters','nw_radius')
    shell_rad = core_rad + self:getfloat('Parameters','shell_t')
    self.sim:SetLayerPatternCircle('nanowire_alshell','AlInP',{vec_mag/2,vec_mag/2},shell_rad)
    self.sim:SetLayerPatternCircle('nanowire_alshell','GaAs',{vec_mag/2,vec_mag/2},core_rad)
    --self.sim:SetLayerPatternCircle('nanowire_alshell','AlInP',{0,0},shell_rad)
    --self.sim:SetLayerPatternCircle('nanowire_alshell','GaAs',{0,0},core_rad)
    -- Si layer and patterning 
    --self.sim:AddLayer('nanowire_sishell',self:getfloat('Parameters','sio2_height'),'Cyclotene')
    ---- Add patterning to layer with SiO2 shell 
    --self.sim:SetLayerPatternCircle('nanowire_sishell','SiO2',{vec_mag/2,vec_mag/2},shell_rad)
    --self.sim:SetLayerPatternCircle('nanowire_sishell','GaAs',{vec_mag/2,vec_mag/2},core_rad)
    -- Substrate layer and air transmission region
    self.sim:AddLayer('substrate',self:getfloat('Parameters','substrate_t'),'GaAs')
    --self.sim:AddLayerCopy('air_below',Thickness=conf.self:getfloat('Parameters','air_t'),Layer='air') 
end

function Simulator:set_excitation()
    -- Defines the excitation of the simulation
    local f_phys = self:getfloat('Parameters','frequency')
    c = 299792458
    print(string.format('Physical Frequency = %E',f_phys))
    local c_conv = c/self:getfloat("General","base_unit")
    local f_conv = f_phys/c_conv
    print(string.format('Converted Frequency = %f',f_conv))
    self.sim:SetFrequency(f_conv)

    -- Define incident light. Normally incident with frequency dependent amplitude
    --local E_mag = self:get_incident_amplitude(f_phys,vec_mag,self.conf["General"]["input_power"])
    local E_mag = 1
    -- To define circularly polarized light, basically just stick a j (imaginary number) in front of
    -- one of your components. The handedness is determined by the component you stick the j in front
    -- of. From POV of source, looking away from source toward direction of propagation, right handed
    -- circular polarization has j in front of y component. Magnitudes are the same. This means
    -- E-field vector rotates clockwise when observed from POV of source. Left handed =
    -- counterclockwise. 
    -- In S4, if indicent angles are 0, p-polarization is along x-axis. The minus sign on front of the 
    -- x magnitude is just to get things to look like Anna's simulations.
    if self.conf['General']['polarization'] == 'rhcp' then
        -- Right hand circularly polarized
        self.sim:SetExcitationPlanewave({0,0},{-E_mag,0},{-E_mag,90})
    elseif self.conf['General']['polarization'] == 'lhcp' then
        -- Left hand circularly polarized
        self.sim:SetExcitationPlanewave({0,0},{-E_mag,90},{-E_mag,0})
    elseif self.conf['General']['polarization'] == 'lpx' then
        -- Linearly polarized along x axis (TM polarixation)
        self.sim:SetExcitationPlanewave({0,0},{0,0},{E_mag,0})
    elseif self.conf['General']['polarization'] == 'lpy' then
        -- Linearly polarized along y axis (TE polarization)
        self.sim:SetExcitationPlanewave({0,0},{E_mag,0},{0,0})
    else 
        pl.utils.quit('Invalid polarization specification')
    end
end

function Simulator:clean_files(path)
    -- Deletes all output files with a given base path
    local glob = pl.stringx.join('',{path,".*"})
    local cwd = pl.path:currentdir()
    local existing_files = pl.dir.getfiles(cwd,glob)
    if existing_files[1] then
        for key,afile in pairs(existing_files) do
            pl.file.delete(afile)
        end
    end
end

function Simulator:get_fields()
    -- Gets the fields throughout the device
    local output_file = pl.path.join(self.conf['General']['sim_dir'],self.conf["General"]["base_name"])
    self:clean_files(output_file)
    local x_samp = self:getint('General','x_samples')
    local y_samp = self:getint('General','y_samples')
    local z_samp = self:getint('General','z_samples')
    --local height = self:getfloat('Parameters','total_height')
    -- Just get the fields from the ITO thru the nanowire and into the 1st micron of substrate
    local height = self:getfloat('Parameters','nw_height')+self:getfloat('Parameters','ito_t')+1
    local dz = height/z_samp
    local vec_mag = self:getfloat('Parameters','array_period')
    local zvec = pl.List.range(0,height,dz)
    -- Get gnoplot output of vector field
    local prefix = pl.path.join(self.conf['General']['sim_dir'],'vecfield')
    self.sim:SetBasisFieldDumpPrefix(prefix)
    if self:getbool('General','adaptive_convergence') then
        self:adaptive_convergence(x_samp,y_samp,zvec,output_file)
    else
        local numbasis = self:getint('Parameters','numbasis')
        self:set_basis(numbasis)
        for i,z in ipairs(zvec) do 
            self.sim:GetFieldPlane(z,{x_samp,y_samp},'FileAppend',output_file)
        end
    end
    -- For some reason this needs to be done after we compute the fields 
    -- Get layer patterning  
    local out = pl.path.join(self.conf['General']['sim_dir'],'pattern.dat')
    self.sim:OutputLayerPatternRealization('nanowire_alshell',x_samp,y_samp,out)
    out = pl.path.join(self.conf['General']['sim_dir'],'eps_realspace.dat')
    outf = io.open(out,'w')
    for x=0,vec_mag,0.005 do
    	for y=0,vec_mag,0.005 do
    		er,ei = self.sim:GetEpsilon({x,y,1}) -- returns real and imag parts
    		outf:write(x .. '\t' .. y .. '\t' .. er .. '\t' .. ei .. '\n')
    	end
    	outf:write('\n')
    end
    outf:close()
    --self.sim:OutputStructurePOVRay(Filename='out.pov')
end

function Simulator:get_indices() 
    local x_samp = self:getint('General','x_samples')
    local y_samp = self:getint('General','y_samples')
    local z_samp = self:getint('General','z_samples')
    local height = self:getfloat('Parameters','total_height') 
    local dz = height/z_samp
    local start_plane = math.floor(self:getfloat('Parameters','air_t')/dz)
    -- Remember lua tables are indexed from 1
    local first = math.floor(start_plane*(x_samp*y_samp)+1)
    local end_plane = math.floor((self:getfloat('Parameters','nw_height')+self:getfloat('Parameters','air_t')+
            self:getfloat('Parameters','ito_t'))/dz)
    local last = math.floor(end_plane*(x_samp*y_samp))
    return first,last
end

function Simulator:calc_diff(d1,d2,exclude) 
    --Pluck out the fields, excluding region outside nanowire
    local exc = exclude or true
    if exc then
        start_row,end_row = self:get_indices()
    else
        start_row = 1
        end_row = -1
    end
    --print(pl.pretty.write(d1))
    local f1 = pl.array2d.slice(d1,start_row,4,end_row,-1)
    local f2 = pl.array2d.slice(d2,start_row,4,end_row,-1)    
    --print(pl.pretty.write(f1,''))
    --print(pl.pretty.write(f2,''))
    -- Get all the differences between the two
    local diffs = pl.array2d.map2('-',2,2,f1,f2)
    -- This is a 2D table where each row contains the difference between the two electric
    -- field vectors at each point in space, which is itself a vector. We want the magnitude
    -- squared of this difference vector 
    -- First square the components
    local diffsq = pl.array2d.map2('*',2,2,diffs,diffs)
    local rows,cols = pl.array2d.size(diffsq)
    -- Now sum the components squared 
    local magdiffsq = pl.array2d.reduce_cols('+',diffsq)
    -- We now compute the norm of the efield from out initial sim at each point in space
    local esq = pl.array2d.map2('*',2,2,f1,f1)
    local normsq = pl.array2d.reduce_cols('+',esq)
    --norm = pl.tablex.map(math.sqrt,normsq)
    -- Error as a percentage should be the square root of the ratio of sum of mag diff vec 
    -- squared to mag efield squared
    local diff = math.sqrt(pl.tablex.reduce('+',magdiffsq)/pl.tablex.reduce('+',normsq))
    print('Percent Difference: '..diff)
    return diff
end

function Simulator:calc_diff_fast(d1,d2,exclude)
    local d1 = d1..'.E'
    local d2 = d2..'.E'
    local exc = exclude or true
    if exc then
        start_row,end_row = self:get_indices()
    else
        start_row = 1
        end_row = -1
    end
    script = self.conf['General']['calc_diff_script']
    cmd = 'python3 '..script
    cmd = cmd..' '..d1..' '..d2..' '..'--start '..start_row..' --end '..end_row     
    print(cmd)
    file = io.popen(cmd,'r')
    out = file:read()
    print('Percent Difference = '..out)
    val = tonumber(out)
    return val
end

function Simulator:adaptive_convergence(x_samp,y_samp,zvec,output)
    print('Beginning adaptive convergence procedure ...')
    -- Gets the fields throughout the device
    local percent_diff = 1
    local output = pl.path.join(self.conf['General']['sim_dir'],self.conf["General"]["base_name"])
    local d1 = output..'1'
    local start_basis = self:getint('Parameters','numbasis')
    self:set_basis(start_basis)
    print('Starting with '..start_basis..' number of basis terms')
    for i, z in ipairs(zvec) do
        self.sim:GetFieldPlane(z,{x_samp,y_samp},'FileAppend',d1)
    end
    --local data1 = pl.data.read(d1..'.E',{no_convert=true})
    local iter = 1
    local new_basis = start_basis 
    local max_diff = self:getfloat('General','max_diff') or .1
    local max_iter = self:getfloat('General','max_iter') or 12
    while percent_diff > max_diff and iter < max_iter do
        print('ITERATION '..iter)
        new_basis = new_basis + 20
        local d2 = output..'2'
        self.sim:SetNumG(new_basis)
        self:set_excitation()
        print('Computing solution with '..new_basis..' number of basis terms')
        for i,z in ipairs(zvec) do 
            self.sim:GetFieldPlane(z,{x_samp,y_samp},'FileAppend',d2)
        end
        --local data2 = pl.data.read(d2..'.E',{no_convert=true})
        --percent_diff = self:calc_diff(data1,data2)
        percent_diff = self:calc_diff_fast(d1,d2)
        data1 = data2 
        -- Clean up the files
        self:clean_files(d1)
        pl.file.move(d2..'.E',d1..'.E')
        pl.file.move(d2..'.H',d1..'.H')
        iter = iter+1
    end
    -- Move files to expected name and record number of basis terms
    if percent_diff > .1 then
        print('Exceeded maximum number of iterations!')
        conv_file = pl.path.join(self.conf['General']['sim_dir'],'not_converged_at.txt')
    else
        print('Converged at '..new_basis..' basis terms')
        conv_file = pl.path.join(self.conf['General']['sim_dir'],'converged_at.txt')
    end
    pl.file.move(d1..'.E',output..'.E')
    pl.file.move(d1..'.H',output..'.H')
    pl.file.write(conv_file,tostring(new_basis)..'\n')
end

function Simulator:get_fluxes() 
    -- Gets the incident, reflected, and transmitted powers
    -- Note: these all return real and imag components of forward and backward waves
    -- as follows: forward_real,backward_real,forward_imaginary,backward_imaginary
    print('Computing fluxes ...')
    local path = pl.path.join(self.conf['General']['sim_dir'],'fluxes.dat')
    local outf = io.open(path,'w')
    outf:write('# Layer,ForwardReal,BackwardReal,ForwardImag,BackwardImag\n')
    for i,layer in ipairs({'air','nanowire_alshell','substrate'}) do
        local inc_fr,inc_br,inc_fi,inc_bi = self.sim:GetPowerFlux(layer)
        local row = string.format('%s,%s,%s,%s,%s\n',layer,inc_fr,inc_br,inc_fi,inc_bi)
        outf:write(row)
    end
    local z = self:getfloat('Parameters','substrate_t')
    local trans_fr,trans_br,trans_fi,trans_bi = self.sim:GetPowerFlux('substrate',z)
    row = string.format('substrate_bottom,%s,%s,%s,%s\n',trans_fr,trans_br,trans_fi,trans_bi)
    outf:write(row)
    z = self:getfloat('Parameters','air_t')
    trans_fr,trans_br,trans_fi,trans_bi = self.sim:GetPowerFlux('air',z)
    row = string.format('air_bottom,%s,%s,%s,%s\n',trans_fr,trans_br,trans_fi,trans_bi)
    outf:write(row)
    outf:close()
end

function Simulator:get_energy_integrals()
    -- Gets energy density integrals
    --layers = {'air','ito','nanowire_alshell','nanowire_sishell','substrate'}    
    local layers = {'air','nanowire_alshell'}    
    print('Computing energy densities ...')
    local path = pl.path.join(self.conf['General']['sim_dir'],'energy_densities.dat')
    local outf = io.open(path,'w')
    outf:write('# Layer, Real, Imaginary\n')
    for i,layer in ipairs(layers) do
        local r,i = self.sim:GetLayerElectricEnergyDensityIntegral(layer)
        local row = string.format('%s,%s,%s\n',layer,r,i)
        outf:write(row)
    end
    outf:close()
end
----------------------------------------------------
-- Main program 
----------------------------------------------------
function main() 
    args = pl.lapp [[
A program that uses the S4 RCWA simulation library to simulate 
the optical properties of a single nanowire in a square lattice
    <config_file> (string) Absolute path to simulation INI file
]]
    if not pl.path.isfile(args['config_file']) then
        error('The config file specified does not exist or is not a file')
    end
    simulator = Simulator(args['config_file'])
    --conf = parse_config(args['config_file'])
    --build_sim(conf)
    simulator:set_lattice()
    simulator:configure()
    simulator:build_device()
    simulator:set_excitation()
    simulator:get_fields()
    simulator:get_fluxes()
    simulator:get_energy_integrals()
end

main()
