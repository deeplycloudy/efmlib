import numpy as np

#Calibrate raw voltage signal to efield:
#-------

#Conversion constants for efield calibration and conversion:
#------------

def get_omegaRC(f_spin, capacitance, resistance):
    """ f_spin is an assumed spin rate. It only needs to be close for an
    accurate calibration. as long as if 1/omega_RC**2 is <<, it won't
    meaningfully affect the calibration.

    It is also important that omega_RC is clos to about 100 so that Ez and Eh
    remain separable.

    capacitance and resistance are the values in the circuit, in farads and ohms
    """
    omega  = 2*np.pi*f_spin # Angular frequency of spheres' spin
    omega_RC    = omega*resistance*capacitance
    return omega_RC


def calibrated_E(peak_to_peak_range,
                 capacitance=1e-7,
                 resistance=1e8,
                 assumed_spin_rate=2.0):
    omega_RC = get_omegaRC(assumed_spin_rate, capacitance, resistance)
    E_mag = (8.8e11 * np.sqrt((1 + 1.0/(omega_RC*omega_RC))) *
             capacitance * peak_to_peak_range/2.0
            )
    return E_mag
