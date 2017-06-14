local pl = require('pl.import_into')()
local yml = require('yaml')
--require('pl')
local S4 = require('RCWA')
-- Actually raises error, stop execution, and provides traceback when using
-- pl.utils.raise
pl.utils.on_error('error')

---------------------------------------------------
-- Functions 
----------------------------------------------------

function isArray(t)
	if type(t)~="table" then return false,"Argument is not a table! It is: "..type(t) end
	--check if all the table keys are numerical and count their number
	local count=0
	for k,v in pairs(t) do
		if type(k)~="number" then return false else count=count+1 end
	end
	--all keys are numerical. now let's see if they are sequential and start with 1
	for i=1,count do
		--Hint: the VALUE might be "nil", in that case "not t[i]" isn't enough, that's why we check the type
		if not t[i] and type(t[i])~="nil" then return false end
	end
	return true
end

---------------------------------------------------
-- Classes 
----------------------------------------------------

-- Config Class
-- This class represents the configuration for a particular sim and has the
-- associated methods needed to manipulate the config data structure
--
pl.class.Config()

function Config:_init(path)
    self._conf = self:load_config(path)
    self.dep_graph = {}
    self:parse_config()
    -- Write out this new YAML representation of the config with all references
    -- resolved and expressions evaluated
    out_path = pl.path.join(self:get({'General','sim_dir'}),'sim_conf.yml')
    self:write(out_path)
end

function Config:get(keys) 
    -- A function to retrieve an item from the config given a single key or an
    -- array_like table of keys
    local ret = self._conf
    if type(keys) == 'table' then
        if not isArray(keys) then
            pl.utils.raise("Must provide either a string or array-like table to Config:get")
        end
        for i,key in ipairs(keys) do
            -- If we store a list anywhere in the config, we need to be
            -- able to get at the elements of that list using numbers
            if tonumber(key) ~= nil then
                key = tonumber(key)
            end
            ret = ret[key] 
            local outstr = pl.pretty.write(keys,'')
            if ret == nil then 
                pl.utils.raise('Attempted to access invalid location at: '..outstr)
            end
        end
    else
         ret = self._conf[keys]
    end
    -- Check to make sure we got a valid key sequence. This works because you
    -- cant actually store nil as a value to a key in lua, so if ret is nil we
    -- tried to access a non-existant location
    local outstr = pl.pretty.write(keys,'')
    if ret == nil then 
        pl.utils.raise('Attempted to access invalid location at: '..outstr)
    end
    return ret
end

function Config:set(key,value)
    -- A function to set an item in the config given a single key or an
    -- array_like table of keys and a value
    if value == nil then
        pl.utils.raise('Cannot store nil in a table in Lua. Use remove function for this')
    end
    -- Presumably if we pass in a value that is a string of all numbers, we
    -- want to actually store it as a number
    if tonumber(value) ~= nil then
        value = tonumber(value)
    end
    local ret = self._conf
    if type(key) == 'table' then
        if not isArray(key) then
            pl.utils.raise("Must provide either a string or array-like table to Config:get")
        end
        for i=1,#key-1 do
            local k = key[i]
            if tonumber(k) ~= nil then
                k = tonumber(k)
            end
            ret = ret[k]
        end
        -- This part is necessary to handle tables as leaves in the
        -- config
        local fkey = key[#key]
        if tonumber(fkey) ~= nil then
            fkey = tonumber(fkey)
        end
        ret[fkey] = value
    else
        if tonumber(key) ~= nil then
            key = tonumber(key)
            self._conf[key] = value
        else
            self._conf[key] = value
        end
    end
end

function Config:remove(key)
    -- The accepted way of removing a keyed entry in a lua table is to just set
    -- that element to nil
    self:set(key,nil)
end

function Config:load_config(path)
    -- Simple function to load a lua table from a path to a yaml file
    local conf_file = io.open(path,'r')
    -- *a reads in entire contents of file
    local text = conf_file:read("*a")
    local conf = yml.load(text)
    conf_file:close()
    return conf
end

function Config:parse_config()
    -- A wrapper for parsing the config around the two functions that actually
    -- do all the work
    self:interpolate()
    self:evaluate()
    return conf 
end

function Config:evaluate(in_table,old_key)
    -- Evaluates any expressions surrounded in back ticks `like_so+blah`
    local t = in_table or self._conf
    for key, value in pairs(t) do
        if type(value) == 'table' then
            local new_key
            if old_key then 
                new_key = old_key..'.'..key
            else 
                new_key = key
            end
            self:evaluate(value,new_key)
        elseif type(value) == 'string' then 
            if pl.stringx.startswith(value,'`') and pl.stringx.endswith(value,'`') then
                tmp = pl.stringx.strip(value,'`')
                tmp = pl.stringx.join('',{'result = ',tmp})
                -- This loads the lua statement contained in tmp and evaluates it, making result
                -- available in the current scope
                f = load(tmp)
                f()
                key_seq = pl.stringx.split(old_key,'.')
                table.insert(key_seq,key)
                self:set(key_seq,result)
            end
        end
    end
end

function Config:match_replace(string)
    -- Matches the replace string so I don't have to keep copying and pasting
    -- this weird line everywhere 
    -- Matches $(some_params)s where any character EXCEPT a closing parenthesis
    -- ) can be inside the %( )s. Extracts whatever is inside the parenthesis
    -- and can match more than one group
    -- Note % is the escpae character, and parenthesis need to be
    -- escaped
    local matches = string.gmatch(string,"%%%(([^)]+)%)s") 
    return matches
end

function Config:_find_references(in_table,old_key)
    -- Build out the dependency graph of references in the config 
    --
    local t = in_table or self._conf
    for key,value in pairs(t) do
        -- If we got a table back, recurse
        if type(value) == 'table' then
            local new_key
            if old_key then 
                new_key = old_key..'.'..key
            else 
                new_key = key
            end
            self:_find_references(value,new_key)
        elseif type(value) == 'string' then
            -- If we got something other than a table, check for matches to the
            -- replacement string and loop through all of then
            local matches = self:match_replace(value)
            local new_key = old_key..'.'..key
            for match in matches do
                -- If we've already found this reference before, increment its
                -- reference count and update the list of keys referring to it
                local dep_keys = pl.tablex.keys(self.dep_graph)
                if pl.tablex.find(dep_keys,match) then
                    data = self.dep_graph[match]
                    data['ref_count'] = data['ref_count'] + 1
                    table.insert(data['ref_by'],new_key)
                else
                    self.dep_graph[match] = {ref_count=1,ref_by={new_key}}
                end
            end
        end
    end
end

function Config:build_dependencies()

    -- First we find all the references and the exact location(s) in the config
    -- that each reference ocurrs at
    self:_find_references()
    -- Now we build out the "refers_to" entry for each reference to see if a
    -- reference at one place in the table refers to some other value
    -- For each reference we found
    for ref,data in pairs(self.dep_graph) do
        -- Loop through all the other references. If the above reference exists
        -- in the "ref_by" table, we know the above reference refers to another
        -- value and we need to resolve that value first. Note we also do this
        -- for ref itself so we can catch circular references 
        for other_ref,its_data in pairs(self.dep_graph) do
            if pl.tablex.find(its_data['ref_by'],ref) then
                if other_ref == ref then
                    pl.utils.raise('There is a circular reference in your config file at '..ref)
                else 
                    if data['ref_to'] then
                        table.insert(data['ref_to'],other_ref)
                    else
                        data['ref_to'] = {other_ref}
                    end
                end
            end
        end
    end
    
end

function Config:_check_resolved(refs)
    -- Checks if a list of references have all been resolved
    local bools = {}
    for i,ref in ipairs(refs) do
        local res = self.dep_graph[ref]['resolved']
        -- If there is no 'resolved' key then res will be nil and we need to
        -- store false in the bools array
        if type(res) == 'boolean' then
            bools[i] = res
        else
            bools[i] = false
        end
    end
    local all_res = true
    for i,bool in ipairs(bools) do
        if not bool then 
            all_res = false
            break
        end
    end
    return all_res
end

function Config:_resolve(ref)
    local ref_data = self.dep_graph[ref]
    -- Retrieve the value of this reference
    local key_seq = pl.stringx.split(ref,'.')
    local repl_val = self:get(key_seq)
    -- Loop through all the locations that contain this reference
    for _,loc in ipairs(ref_data['ref_by']) do
        -- Get the string we need to run the replacement on
        local rep_seq = pl.stringx.split(loc,'.')
        local entry_to_repl = self:get(rep_seq)
        -- Run the actual replacement and set the value at this
        -- location to the new string
        local substr = '%('..ref..')s'
        local esc = pl.utils.escape(substr)
        local rep_par = string.gsub(entry_to_repl,esc,repl_val)
        self:set(rep_seq,rep_par)
    end
end

function Config:interpolate()
    -- Before we can actually interpolate, we need to build out a dependency
    -- graph for all the references within the config file
    self:build_dependencies()
    local config_resolved = false
    while not config_resolved do
        print('CONFIG NOT RESOLVED, MAKING PASS')
        -- Now we can actually perform any resolution
        for ref,ref_data in pairs(self.dep_graph) do
            -- If the actual location of this references doesn't itself refer to
            -- something else, we can safely resolve it because we know it has a
            -- value
            if not ref_data['ref_to'] then
                print('NO REFERENCES, RESOLVING')
                self:_resolve(ref) 
                self.dep_graph[ref]['resolved'] = true
            else
                print('CHECKING REFERENCES')
                -- If all the locations this reference points to are resolved, then we
                -- can go ahead and resolve this one 
                local all_res = self:_check_resolved(ref_data['ref_to'])
                print('ALL RES = ',all_res)
                if all_res then
                    self:_resolve(ref)
                    self.dep_graph[ref]['resolved'] = true
                end
            end
        end
        config_resolved = self:_check_resolved(pl.tablex.keys(self.dep_graph))
    end
end

function Config:write(path)
    outfile = io.open(path,'w')
    local dump = yml.dump(self._conf)
    outfile:write(dump)
    outfile:close()
end


-- Simulator
-- Wraps around the S4 simulation object and handles initialization of config
-- object, parameter transforms, nk lookup/interpolation, and other such necessary tasks

pl.class.Simulator()

function Simulator:_init(conf_path)
    self.sim = S4.NewSimulation()
    self.conf = Config(conf_path) 
    -- We access this so often we'll just store it as an attribute for
    -- convenience
    self.dir = self.conf:get({'General','sim_dir'})
end

function Simulator:configure()
    -- Configure simulation options
    --
    -- Clean up values
    for key,val in pairs(self.conf:get('Solver')) do
        --if type(val) == 'number' or pl.stringx.isdigit(val) then
        if tonumber(val) ~= nil then
            self.conf:set({'Solver',key}, math.floor(val))
        end
    end
    -- Actually set the values
    self.sim:SetVerbosity(self.conf:get({'Solver','Verbosity'}))
    self.sim:SetLatticeTruncation(self.conf:get({'Solver','LatticeTruncation'}))
    if self.conf:get({'Solver','DiscretizedEpsilon'}) then 
        self.sim:UseDiscretizedEpsilon() 
        self.sim:SetResolution(self:getint('Solver','DiscretizationResolution'))
    end
    if self.conf:get({'Solver','PolarizationDecomposition'}) then 
        self.sim:UsePolarizationDecomposition() 
        if self.conf:get({'Solver','PolarizationBasis'}) == 'Jones' then
            self.sim:UseJonesVectorBasis()
        elseif self.conf:get({'Solver','PolarizationBasis'}) == 'Normal' then
            self.sim:UseNormalVectorBasis()
        else 
            print("Using default vector field")
        end                                    
    end
    if self.conf:get({'Solver','LanczosSmoothing'}) then
        self.sim:UseLanczosSmoothing()
    end
    if self.conf:get({'Solver','SubpixelSmoothing'}) then
        self.sim:UseSubpixelSmoothing()
        self.sim:SetResolution(self:getint('Solver','DiscretizationResolution'))
    end
    if self.conf:get({'Solver','ConserveMemory'}) then
        self.sim:UseLessMemory()
    end
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
    local vec_mag = self.conf:get({'Simulation','params','array_period','value'})
    self.sim:SetLattice({vec_mag,0},{0,vec_mag})
end

function Simulator:set_basis(num_basis)
    -- Set desired # of basis terms. Note that this is an upper bound and may not be
    -- the actual number of basis terms used
    local num = tonumber(num_basis)
    self.sim:SetNumG(num)
end

function Simulator:get_height()
    -- Just get the fields from the ITO thru the nanowire and into the 1st micron of substrate
    local height = 0
    for layer,ldata in pairs(self.conf:get('Layers')) do
        layer_t = ldata['params']['thickness']['value']
        height = height+layer_t
    end
    return height
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
    elseif xval > max and extrapolate then
        print('Extrapolate right')
        local m = (y[#y] - y[#y-1])/(x[#x] - x[#x-1])
        val = m*(xval-x[#x])+y[#x]
    elseif (xval < min or xval > max) and not extrapolate then
        error("The x value is outside the extrapolation range and extrapolation is set to false")
    else 

        -- First find the first x value that exceeds our desired value
        local counter = 1
        while x[counter] < xval do
            counter = counter + 1
        end
        val = (((y[counter]-y[counter-1])*(xval - x[counter-1]))/(x[counter]-x[counter-1]))+y[counter-1]
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

function Simulator:get_epsilon(freq,path)
    print('Computing epsilon values ...')
    assert(pl.path.isfile(path))
    local freqvec,nvec,kvec = self:parse_nk(path)
    local n = self:interp1d(freqvec,nvec,freq,true)
    local k = self:interp1d(freqvec,kvec,freq,true)
    print('n =',n)
    print('k =',k)
    local eps_r = n*n - k*k
    local eps_i = 2*n*k
    print('Eps Real =',eps_r)
    print('Eps Imag =',eps_i)
    return eps_r,eps_i
end

function Simulator:build_device()
    -- Builds out the device geometry and materials 
    local f_phys = self.conf:get({'Simulation','params','frequency','value'})
    local vec_mag = self.conf:get({'Simulation','params','array_period'})

    -- First we define all the materials 
    local eps_r
    local eps_i
    for mat,mat_path in pairs(self.conf:get('Materials')) do
        print("Adding material "..mat.." ...")
        eps_r, eps_i = self:get_epsilon(f_phys,mat_path)
        self.sim:AddMaterial(mat,{eps_r,eps_i})
    end
    self.sim:AddMaterial('vacuum',{1,0})
    -- We need to properly sort our layers because order DOES matter. Light
    -- will be incident upon the first layer specified
    local layers = self.conf:get('Layers')
    function layer_comp(l1,l2) 
        --pl.pretty.dump(l1)
        --pl.pretty.dump(l2)
        --print(l1['order'])
        --print(l2['order'])
        if l1['order'] < l2['order'] then
            return true
        elseif l1['order'] > l2['order'] then
            return false
        elseif l1['order'] > l2['order'] and not l1 == l2 then
            pl.utils.raise('Two of your layers have the same order')
        end
    end
    for layer,ldata in pl.tablex.sortv(layers,layer_comp) do
        print('Building layer:',layer)
        print(layer)
        print(ldata['order'])
        local base_mat = ldata['base_material']
        local layer_thickness = ldata['params']['thickness']['value']
        self.sim:AddLayer(layer,layer_thickness,base_mat)
        -- If the layer has any internal geometry, deal with it here
        if ldata['geometry'] then
            for shape,sdata in pl.tablex.sortv(ldata['geometry'],layer_comp) do
                print(shape)
                print(sdata['order'])
                print(sdata['type'])
                shape_mat = sdata['material']
                if sdata['type'] == 'circle' then
                    rad = sdata['radius']
                    cent = sdata['center']
                    self.sim:SetLayerPatternCircle(layer,shape_mat,cent,rad)
                else
                    pl.utils.raise('Shape '..sdata['type']..' is not yet implemented')
                end
            end
        end
    end
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

function Simulator:set_excitation()
    -- Defines the excitation of the simulation
    local f_phys = self.conf:get({'Simulation','params','frequency','value'})
    c = 299792458
    print(string.format('Physical Frequency = %E',f_phys))
    local c_conv = c/self.conf:get({'Simulation',"base_unit"})
    local f_conv = f_phys/c_conv
    print(string.format('Converted Frequency = %f',f_conv))
    self.sim:SetFrequency(f_conv)

    -- Define incident light. Normally incident with frequency dependent amplitude
    local E_mag = self:get_incident_amplitude(f_phys,vec_mag,self.conf:get({"Simulation","input_power"}))
    -- Get angle of incident light
    local polar = self.conf:get({'Simulation','params','polar_angle','value'})
    local azimuthal = self.conf:get({'Simulation','params','azimuthal_angle','value'})
    --local E_mag = 1
    -- To define circularly polarized light, basically just stick a j (imaginary number) in front of
    -- one of your components. The handedness is determined by the component you stick the j in front
    -- of. From POV of source, looking away from source toward direction of propagation, right handed
    -- circular polarization has j in front of y component. Magnitudes are the same. This means
    -- E-field vector rotates clockwise when observed from POV of source. Left handed =
    -- counterclockwise. 
    -- In S4, if indicent angles are 0, p-polarization is along x-axis. The minus sign on front of the 
    -- x magnitude is just to get things to look like Anna's simulations.
    if self.conf:get({'Simulation','polarization'}) == 'rhcp' then
        -- Right hand circularly polarized
        self.sim:SetExcitationPlanewave({polar,azimuthal},{-E_mag,0},{-E_mag,90})
    elseif self.conf:get({'Simulation','polarization'}) == 'lhcp' then
        -- Left hand circularly polarized
        self.sim:SetExcitationPlanewave({polar,azimuthal},{-E_mag,90},{-E_mag,0})
    elseif self.conf:get({'Simulation','polarization'}) == 'lpx' then
        -- Linearly polarized along x axis (TM polarixation)
        self.sim:SetExcitationPlanewave({polar,azimuthal},{0,0},{E_mag,0})
    elseif self.conf:get({'Simulation','polarization'}) == 'lpy' then
        -- Linearly polarized along y axis (TE polarization)
        self.sim:SetExcitationPlanewave({polar,azimuthal},{E_mag,0},{0,0})
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
    local output_file = pl.path.join(self.dir,self.conf:get({"General","base_name"}))
    self:clean_files(output_file)
    local x_samp = self.conf:get({'Simulation','x_samples'})
    local y_samp = self.conf:get({'Simulation','y_samples'})
    local z_samp = self.conf:get({'Simulation','z_samples'})
    local height = self:get_height()
    local dz = height/z_samp
    local zvec = pl.List.range(0,height,dz)
    local vec_mag = self.conf:get({'Simulation','params','array_period','value'})
    if self.conf:get({'General','adaptive_convergence'}) then
        self:adaptive_convergence(x_samp,y_samp,zvec,output_file)
    else
        local numbasis = self.conf:get({'Simulation','params','numbasis','value'})
        self:set_basis(numbasis)
        for i,z in ipairs(zvec) do 
            self.sim:GetFieldPlane(z,{x_samp,y_samp},'FileAppend',output_file)
        end
    end
    -- Get layer patterning  
    -- For some reason this needs to be done after we compute the fields 
    if self.conf:get({'General','output_pattern'}) then
        -- Get gnoplot output of vector field
        local prefix = pl.path.join(self.dir,'vecfield')
        self.sim:SetBasisFieldDumpPrefix(prefix)
        -- Get dielectric patterning
        print('Outputting patterning of dielectric profile in each layer ...')
        for layer, ldata in pairs(self.conf:get('Layers')) do 
            local out = pl.path.join(self.dir,string.format('%s_pattern.dat',layer))
            self.sim:OutputLayerPatternRealization(layer,x_samp,y_samp,out)
            out = pl.path.join(self.dir,string.format('%s_eps_realspace.dat',layer))
            outf = io.open(out,'w')
            for x=0,vec_mag,0.005 do
            	for y=0,vec_mag,0.005 do
            		er,ei = self.sim:GetEpsilon({x,y,1}) -- returns real and imag parts
            		outf:write(x .. '\t' .. y .. '\t' .. er .. '\t' .. ei .. '\n')
            	end
            	outf:write('\n')
            end
            outf:close()
        end
    end
    --self.sim:OutputStructurePOVRay(Filename='out.pov')
end

function Simulator:get_indices() 
    -- TODO:
    -- This function is not general at all, we need to find a way to generalize
    -- it. Maybe we could provide a list of layers we care about convergence in
    -- in the config file, or a list of layers we DONT care about convergence
    -- in. Either way, we either need to assume we take the difference for the
    -- whole device or that the user can tell us what to calculate. For now,
    -- just do what I want it to do which is exclude the air and substrate
    -- regions
    local x_samp = self.conf:get({'Simulation','x_samples'})
    local y_samp = self.conf:get({'Simulation','y_samples'})
    local z_samp = self.conf:get({'Simulation','z_samples'})
    local height = self:get_height() 
    local dz = height/z_samp
    local first
    local last
    if pl.tablex.find(pl.tablex.keys(self.conf:get('Layers')),'Air') then
        local air_t = self.conf:get({'Layers','Air','params','thickness','value'})
        local start_plane = math.floor(air_t/dz)
        -- Remember lua tables are indexed from 1
        first = math.floor(start_plane*(x_samp*y_samp)+1)
    else
        first = 1
    end
    if pl.tablex.find(pl.tablex.keys(self.conf:get('Layers')),'Substrate') then
        local sub_t = self.conf:get({'Layers','Substrate','params','thickness','value'})
        local sans_sub_height = height - sub_t 
        local end_plane = math.floor(sans_sub_height/dz)
        last = math.floor(end_plane*(x_samp*y_samp))
    else
        last = x_samp*y_samp*z_samp
    end
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
    script = self.conf:get({'General','calc_diff_script'})
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
    local output = pl.path.join(self.dir,self.conf:get({"General","base_name"}))
    local d1 = output..'1'
    local start_basis = self.conf:get({'Simulation','params','numbasis','value'})
    self:set_basis(start_basis)
    print('Starting with '..start_basis..' number of basis terms')
    for i, z in ipairs(zvec) do
        self.sim:GetFieldPlane(z,{x_samp,y_samp},'FileAppend',d1)
    end
    --local data1 = pl.data.read(d1..'.E',{no_convert=true})
    local iter = 1
    local new_basis = start_basis 
    local max_diff = self.conf:get({'General','max_diff'}) or .1
    print('MAX DIFF =',max_diff)
    local max_iter = self.conf:get({'General','max_iter'}) or 12
    percent_diff = 100
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
    if percent_diff > max_diff then
        print('Exceeded maximum number of iterations!')
        conv_file = pl.path.join(self.dir,'not_converged_at.txt')
    else
        print('Converged at '..new_basis..' basis terms')
        conv_file = pl.path.join(self.dir,'converged_at.txt')
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
    local path = pl.path.join(self.dir,'fluxes.dat')
    local outf = io.open(path,'w')
    outf:write('# Layer,ForwardReal,BackwardReal,ForwardImag,BackwardImag\n')
    for layer,ldata in pairs(self.conf:get('Layers')) do
        print('Computing fluxes through layer:',layer)
        -- This gets flux at top of layer
        local inc_fr,inc_br,inc_fi,inc_bi = self.sim:GetPowerFlux(layer)
        local row = string.format('%s,%s,%s,%s,%s\n',layer,inc_fr,inc_br,inc_fi,inc_bi)
        outf:write(row)
        -- This gets flux at the bottom
        local offset = ldata['params']['thickness']['value']
        inc_fr,inc_br,inc_fi,inc_bi = self.sim:GetPowerFlux(layer,offset)
        row = string.format('%s_bottom,%s,%s,%s,%s\n',layer,inc_fr,inc_br,inc_fi,inc_bi)
        outf:write(row)
    end
    outf:close()
end

function Simulator:get_energy_integrals()
    -- Gets energy density integrals
    --layers = {'air','ito','nanowire_alshell','nanowire_sishell','substrate'}    
    local layers = {'air','nanowire_alshell'}    
    print('Computing energy densities ...')
    local path = pl.path.join(self.dir,'energy_densities.dat')
    local outf = io.open(path,'w')
    outf:write('# Layer, Real, Imaginary\n')
    for layer,ldata in pairs(self.conf:get('Layers')) do
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
    simulator:set_lattice()
    simulator:configure()
    simulator:build_device()
    simulator:set_excitation()
    simulator:get_fields()
    simulator:get_fluxes()
    simulator:get_energy_integrals()
end

main()
