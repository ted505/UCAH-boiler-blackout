"""
Axial Aerothermal Profile Module
Computes 1D spatial surface heat flux and radiation-equilibrium wall temperature
along a geometry contour CSV at peak stagnation heating conditions.
"""

import numpy as np
import pandas as pd


def compute_profile_at_max_q(csv_path, freestream_state, q_stag_peak, Rn, 
                             emissivity=0.85, Re_trans=5e5, Re_length=1e5):
    """
    Computes heat flux and radiation-equilibrium wall temperature along a geometry contour.

    Parameters:
    -----------
    csv_path : str or Path
        Path to nose contour CSV (must contain 'x', 's', and 'radial' columns).
    freestream_state : dict
        Freestream state containing 'rho', 'u', 'mu', 'p', 'T', and optional 'cp'.
    q_stag_peak : float
        Peak stagnation heat flux [W/m^2].
    Rn : float
        Reference nose radius [m].
    emissivity : float
        TPS surface emissivity (default: 0.85).
    Re_trans : float
        Critical transition Reynolds number (default: 5e5).
    Re_length : float
        Transition width scale in Reynolds number space (default: 1e5).

    Returns:
    --------
    pd.DataFrame
        Dataframe containing spatial metrics: x_m, radial_m, s_m, Re_s, is_turbulent,
        q_flux_MW_m2, and T_wall_K.
    """
    # 1. Load OML contour (mapped to nose_contour.csv structure)
    df = pd.read_csv(csv_path)
    x = df['x'].values
    s = df['s'].values
    y = df['radial'].values  # Radial distance from axial centerline

    # 2. Local surface incline angle theta relative to axial flow
    dy_dx = np.gradient(y, x)
    theta = np.abs(np.arctan(dy_dx))

    # 3. Extract freestream flow properties
    rho_inf = freestream_state['rho']
    u_inf = freestream_state['u']
    mu_inf = freestream_state['mu']
    p_inf = freestream_state['p']
    T_inf = freestream_state['T']
    cp = freestream_state.get('cp', 1005.0)

    # 4. Modified Newtonian Surface Pressure and Flow Edge Approximation
    q_dyn = 0.5 * rho_inf * (u_inf**2)
    Cp_max = 1.83  # Hypersonic shock pressure factor
    p_local = p_inf + q_dyn * Cp_max * (np.sin(np.maximum(theta, 0.01))**2)

    gamma = 1.4
    rho_edge = rho_inf * (p_local / p_inf)**(1.0 / gamma)
    u_edge = np.sqrt(np.maximum(0.0, 2.0 * cp * T_inf * (1.0 - (p_local / (q_dyn + p_inf))**((gamma - 1.0) / gamma))))

    # 5. Local Running Reynolds Number (Re_s) along surface distance s
    s_safe = np.maximum(s, 1e-4)
    Re_s = (rho_edge * u_edge * s_safe) / mu_inf

    # 6. Dhawan-Narasimha style smooth transition intermittency model
    gamma_trans = 0.5 * (1.0 + np.tanh((Re_s - Re_trans) / Re_length))
    n = 0.5 + 0.3 * gamma_trans  # Exponent scales from 0.5 (laminar) to 0.8 (turbulent)

    # 7. Compute Local Heat Flux Profile q(s)
    q_local = np.zeros_like(s)
    for i in range(len(s)):
        if s[i] <= Rn:
            # Stagnation to nose-shoulder decay
            q_local[i] = q_stag_peak * (np.cos(s[i] / Rn)**1.5)
        else:
            # Downstream body scaling with transition exponent
            sine_factor = (np.sin(np.maximum(theta[i], 0.02)))**n[i]
            length_factor = (Rn / s_safe[i])**(1.0 - n[i])
            q_local[i] = q_stag_peak * sine_factor * length_factor

    # 8. Radiation-Equilibrium Wall Temperature
    sigma = 5.670374419e-8
    T_wall = (q_local / (emissivity * sigma))**0.25

    return pd.DataFrame({
        'x_m': x,
        'radial_m': y,
        's_m': s,
        'Re_s': Re_s,
        'is_turbulent': gamma_trans > 0.5,
        'q_flux_MW_m2': q_local / 1e6,
        'T_wall_K': T_wall
    })