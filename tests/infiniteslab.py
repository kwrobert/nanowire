import sys
import os
import S4
import numpy as np
import scipy.constants as consts
import scipy.integrate as intg
import matplotlib.pyplot as plt

def compute_fields(sim, z_vals, ongrid=False, plot=True):
    Ex = np.zeros_like(z_vals, dtype=np.complex)
    Ey = np.zeros_like(z_vals, dtype=np.complex)
    Ez = np.zeros_like(z_vals, dtype=np.complex)
    for i, z in enumerate(z_vals):
        if ongrid:
            E, H = sim.GetFieldsOnGrid(z=z, NumSamples=(25, 25), Format='Array')
            Ex[i] = E[0][0][0]
            Ey[i] = E[0][0][1]
            Ez[i] = E[0][0][2]
        else:
            E, H = sim.GetFields(0, 0, z)
            Ex[i] = E[0]
            Ey[i] = E[1]
            Ez[i] = E[2]
    if plot:
        fig, axes = plt.subplots(2, 2, figsize=(14, 7))
        ax1, ax2, ax3, ax4 = axes.flatten()
        ax1.plot(z_vals, Ex.real, label = "Real")
        ax1.plot(z_vals, Ex.imag, label = "Imag")
        ax1.set_ylabel('Ex')
        ax1.set_xlabel('z [um]')
        ax1.legend()
        ax2.plot(z_vals, Ey.real, label = "Real")
        ax2.plot(z_vals, Ey.imag, label = "Imag")
        ax2.set_ylabel('Ey')
        ax2.set_xlabel('z [um]')
        ax2.legend()
        ax3.plot(z_vals, Ez.real, label = "Real")
        ax3.plot(z_vals, Ez.imag, label = "Imag")
        ax3.set_ylabel('Ez')
        ax3.set_xlabel('z [um]')
        ax3.legend()
        normE = np.sqrt(np.absolute(Ex)**2 + np.absolute(Ey)**2 + np.absolute(Ez)**2)
        ax4.plot(z_vals, normE, label = "|E|")
        ax4.legend()
        ax4.set_ylabel('|E|')
        ax4.set_xlabel('z [um]')
        plt.tight_layout()
        plt.show()
    return Ex, Ey, Ez

def analytic_waves(SI_freq, n_slab, Einc, z_vals, polar_angle):
    # polar_angle must be in radians!!
    # Define constants and scalar magnitudes of vectors
    si_omega = 2*np.pi*SI_freq
    theta_T = np.arcsin((1/n_slab)*np.sin(polar_angle))
    beta = n_slab
    alpha = np.cos(theta_T)/np.cos(polar_angle)
    v_slab = 1e6*consts.c / n_slab
    ktrans = si_omega / v_slab
    kinc = si_omega / (1e6*consts.c)
    kref = kinc
    Eref = Einc*(alpha - beta)/(alpha + beta)
    Etrans = Einc*2/(alpha + beta)
    # Define vector amplitudes and wave vectors
    vEinc = Einc*np.array([np.cos(polar_angle), 0, -np.sin(polar_angle)])
    vkinc = kinc*np.array([np.sin(polar_angle), 0, np.cos(polar_angle)])
    vEtrans = Etrans*np.array([np.cos(theta_T), 0, -np.sin(theta_T)])
    vktrans = ktrans*np.array([np.sin(theta_T), 0, np.cos(theta_T)])
    vEref = Eref*np.array([np.cos(polar_angle), 0, np.sin(polar_angle)])
    vkref = kref*np.array([np.sin(polar_angle), 0, -np.cos(polar_angle)])
    # Define 3D positions
    r_vals = np.zeros((z_vals.shape[0], 3))
    r_vals[:, 2] = z_vals
    # Get phases
    inc_phase = np.exp(1j*np.dot(r_vals, vkinc))
    ref_phase = np.exp(1j*np.dot(r_vals, vkref))
    trans_phase = np.exp(1j*np.dot(r_vals, vktrans))
    inc_wave = vEinc*np.column_stack((inc_phase, inc_phase, inc_phase))
    ref_wave = vEref*np.column_stack((ref_phase, ref_phase, ref_phase))
    trans_wave = vEtrans*np.column_stack((trans_phase, trans_phase, trans_phase))
    air_wave = inc_wave + ref_wave
    ind_arr = np.where(z_vals < 0)
    total_wave = np.zeros_like(inc_wave)
    total_wave[ind_arr, :] = air_wave[ind_arr, :]
    ind_arr = np.where(z_vals >= 0)
    total_wave[ind_arr, :] = trans_wave[ind_arr, :]
    Ex = total_wave[:, 0]
    Ey = total_wave[:, 1]
    Ez = total_wave[:, 2]
    return Ex, Ey, Ez 

def main(ongrid):
    L = .75
    numbasis = 5
    sim = S4.New(Lattice=((L, 0), (0, L)), NumBasis=numbasis)
    nn_slab = 2
    k_slab = .2
    eps_vac = complex(1, 0)
    eps_slab = complex(nn_slab**2 - k_slab**2, 2*nn_slab*k_slab)
    n_slab = complex(nn_slab, k_slab)
    z_vals = np.linspace(-5, 5, 1000)
    # z_vals = np.array([1.])
    sim.SetMaterial(Name = "TestMaterial", Epsilon = eps_slab)
    sim.SetMaterial(Name = "Vacuum", Epsilon = eps_vac)

    sim.AddLayer(Name='VacuumAbove', Thickness = 0, Material='Vacuum')
    sim.AddLayer(Name='Slab', Thickness = 0, Material='TestMaterial')

    SI_freq = 2e14 # Hertz
    SI_wvlgth = 1e6*consts.c / SI_freq # micrometers
    print(SI_wvlgth)
    LH_freq = SI_freq/(consts.c * 1e6) # 1/micrometers
    print(LH_freq)
    sim.SetFrequency(LH_freq)

    polar_angle = np.pi/4
    incident_amp = complex(1, 0)
    sim.SetExcitationPlanewave(IncidenceAngles=(np.rad2deg(polar_angle), 0), sAmplitude=complex(0, 0), pAmplitude=incident_amp)
    Ex, Ey, Ez = compute_fields(sim, z_vals, ongrid=ongrid, plot=False)
    aEx, aEy, aEz = analytic_waves(SI_freq, n_slab, incident_amp, z_vals, polar_angle)

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    for i, comp in enumerate([(Ex, aEx, 'Ex'), (Ey, aEy, 'Ey'), (Ez, aEz, 'Ez')]):
        numeric_result = comp[0]
        analytic_result = comp[1]
        lab = comp[2]
        diff = numeric_result - analytic_result
        axes[i, 0].plot(z_vals, numeric_result.real, label="%s Real"%lab)
        axes[i, 0].plot(z_vals, numeric_result.imag, label="%s Imag"%lab)
        axes[i, 0].plot(z_vals, analytic_result.real, label="Analytic %s Real"%lab)
        axes[i, 0].plot(z_vals, analytic_result.imag, label="Analytic %s Imag"%lab)
        axes[i, 0].set_xlabel('z [um]')
        axes[i, 0].set_ylabel(lab)
        axes[i, 0].legend()
        axes[i, 1].plot(z_vals, diff.real, label="%s Difference Real"%lab)
        axes[i, 1].plot(z_vals, diff.imag, label="%s Difference Imag"%lab)
        axes[i, 1].set_xlabel('z [um]')
        axes[i, 1].set_ylabel('Difference')
        axes[i, 1].legend()
    plt.show()

if __name__ == '__main__':
    args = sys.argv[1:]
    if args:
        main(bool(args[0]))
    else:
        main(False)
