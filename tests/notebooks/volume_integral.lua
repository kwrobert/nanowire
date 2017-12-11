S = S4.NewSimulation()
-- Params
period = .25
numbasis = 300
eps_real = 13.77
eps_imag = 0.8
thickness = .5
freq = 1.3342563807926082
--
S:SetVerbosity(9)
S:SetLattice({period,0}, {0,period})
S:SetNumG(numbasis)
S:AddMaterial('Vacuum', {1.0, 0})
S:AddMaterial('GaAs', {eps_real, eps_imag})
S:AddLayer('top', 0, 'Vacuum')
S:AddLayer('slab', thickness, 'GaAs')
S:AddLayerCopy('bottom', 0, 'top')
S:SetExcitationPlanewave(
        {0, 0}, -- phi in [0,180), theta in [0,360)
        {1, 0}, -- s-polarization amplitude and phase in degrees
        {0, 0}) -- p-polarization
S:SetFrequency(freq)
tfr, tbr, tfi, tbi = S:GetPowerFlux('top')
bfr, bbr, bfi, bbi = S:GetPowerFlux('bottom')
print("### Top ###")
print(tfr)
print(tbr)
print(tfi)
print(tbi)
print("### Bottom ###")
print(bfr)
print(bbr)
print(bfi)
print(bbi)
pabs_r = (tfr - bfr) + (tbr- bbr)
pabs_i = (tfi - bfi) + (tbi- bbi)
print("Power Absorbed Real: " .. pabs_r)
print("Power Absorbed Imag: " .. pabs_i)
int_r, int_i = S:GetLayerElectricEnergyDensityIntegral('slab')
print("Integral of epsilon |E|^2 Real: " .. freq*.5*int_r)
print("Integral of epsilon |E|^2 Imag: " .. freq*.5*int_i)
int_r, int_i = S:GetLayerElectricFieldIntensityIntegral('slab')
print("Integral of |E|^2 epsilon outside Real: " .. freq*.5*eps_real*int_r)
print("Integral of |E|^2 epsilon outside Imag: " .. freq*.5*eps_imag*int_i)
