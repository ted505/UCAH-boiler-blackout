"""
UCAH stagnation point heating analysis using a prescribed CSV trajectory.
By: Jonathan Cats

INSTALL:
   cantera, numpy, scipy, matplotlib, ambiance, pandas

Workflow
1. load_trajectory_csv imports the time/altitude/speed samples.
2. calculate_freestream_properties of air at each imported altitude (Cantera).
3. solve_equilibrium_normal_shock: calculates the compressed gas properties (Cantera + scipy).
4. solve_stagnation_edge_state: isentropically decelerates the post-shock gas to the stagnation-point boundary-layer edge (Cantera)
5. solve_wall_temperatures: Fay-Riddell Radiative Equillibrium (stefan boltzman)
6. save_results and plot_results write the profiles and figures.

LIMITS
200-5000 K neutral-gas model; no finite-rate chemistry, ionization, thermal
inertia, conduction, ablation, or incoming radiation. Time is an independent
trajectory coordinate, NOT a transient wall-energy-balance calculation.
Samples at Mach <= 1 are explicitly marked skipped with NaN heating values;
no normal-shock solution is forced. Later supersonic samples are still used.
Low-Mach and continuum-model warnings remain. Passing numerical checks
is not validation of the trajectory, aerodynamic model, or heating physics.
A missing/invalid CSV causes an error, never fallback to the old solver.

OUTPUTS (default: ucah_csv_results beside this script)
profile.csv, results.npz, run_summary.json, heat_flux.png, wall_temperature.png.
Time is the default plot x axis; --plot-x altitude selects altitude instead.
Wall arrays in results.npz have shape (N, 1), preserving [:, 0] access for
one stagnation point; there is no redundant angle-of-attack sweep.
Source CSV data row numbers and a SHA-256 fingerprint are recorded.
Heating-loop failures save partial profiles; uncomputed values stay NaN.

literally every single correlation in here can be found in Andersons "Hypersonic and High Heat Gas Dynamics" book in OneDrive
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import NamedTuple

import numpy as np
try:
    import pandas as pd
except ImportError as exc:
    raise SystemExit("[DEPENDENCY FAILURE] Install pandas: python -m pip install pandas") from exc
from scipy.optimize import brentq
from ambiance import Atmosphere
try:
    import cantera as cantera
except ImportError as exc:
    raise SystemExit("[DEPENDENCY FAILURE] Install Cantera: "
                     "python -m pip install 'cantera>=3.1,<4'") from exc
from ruamel.yaml import YAML  # Installed as a Cantera dependency


# =============================================================================
# Heating parameters -- original values retained; CSV prescribes the trajectory
# =============================================================================
# LEWIS-NUMBER CONVENTION FOR THIS FILM MODEL:
# The original comment uses mass diffusivity / thermal diffusivity, D/alpha,
# where alpha = k/(rho*cp). Many references instead define Le = alpha/D.
# The positive Le**0.63 multiplier below retains the ORIGINAL convention/value;
# do not substitute the reciprocal convention without changing the correlation.
# This is an effective specified parameter; species diffusion is not solved.
effective_lewis_number = 1.4                         # Effective mass / thermal diffusivity ratio
mass_transfer_exponent = 0.63                   # Heat/mass-transfer exponent, not catalycity
nose_radius_m = 0.001  # 1 mm nose curvature radius; not the drag reference area.
wall_emissivity = 0.8  # Dimensionless thermal-radiation efficiency, between 0 and 1.
stefan_boltzmann_constant_W_m2_K4 = 5.67e-8
FREESTREAM_SPECIES_MASS_FRACTIONS = {"N2": 0.75523, "O2": 0.23142, "Ar": 0.01288, "CO2": 0.00047}
AIR_SPECIES_NAMES = ["N2", "O2", "N", "O", "NO", "NO2", "N2O", "Ar", "CO2", "CO", "C"]
MIN_SUPPORTED_TEMPERATURE_K = 200.0
MAX_SUPPORTED_TEMPERATURE_K = 5000.0              # Conservative neutral-gas implementation cap
CONSERVATION_RELATIVE_TOLERANCE = 2e-6


class WallHeatingResult(NamedTuple):
    """One surface-chemistry case at one altitude; each name includes its units."""
    temperature_K: float
    heat_flux_W_m2: float
    relative_heat_balance_error: float
    sensible_driving_enthalpy_J_kg: float
    recombination_enthalpy_J_kg: float


class AnalysisFailure(RuntimeError):
    pass


def raise_analysis_failure(stage, reason, **failure_context):
    detail = "\n".join(f"  {key}: {value}" for key, value in failure_context.items())
    raise AnalysisFailure(f"[{stage}] {reason}\n{detail}")


# NUMERICAL METHOD: Brent's bracketed scalar root method, f(x) = 0.
# Requires a continuous residual and opposite endpoint signs (or an endpoint root).
# Combines bracketing/bisection and interpolation; it is not a physical equation.
# xtol and rtol are absolute and relative root-coordinate tolerances.
def solve_bracketed_root(residual_function, lower_bound, upper_bound, stage, failure_context, xtol=1e-9, rtol=1e-10):
    if not np.all(np.isfinite([lower_bound, upper_bound])) or lower_bound >= upper_bound:
        raise_analysis_failure(stage, "Invalid root-search interval.", bracket=(lower_bound, upper_bound), **failure_context)
    lower_bound_residual, upper_bound_residual = residual_function(lower_bound), residual_function(upper_bound)
    diagnostic = dict(failure_context, bracket=(lower_bound, upper_bound), residual_low=lower_bound_residual,
                      residual_high=upper_bound_residual)
    if not np.all(np.isfinite([lower_bound_residual, upper_bound_residual])):
        raise_analysis_failure(stage, "Nonfinite residual at a bracket endpoint.", **diagnostic)
    if lower_bound_residual == 0:
        return lower_bound
    if upper_bound_residual == 0:
        return upper_bound
    if np.signbit(lower_bound_residual) == np.signbit(upper_bound_residual):
        reason = "Residuals have the same sign; no root is bracketed."
        if stage == "WALL TEMPERATURE" and lower_bound_residual < 0:
            reason += (" Radiation exceeds incoming heat even at the minimum "
                       "supported temperature. Increasing the upper bound will "
                       "not fix this; do not extrapolate the thermodynamic data.")
        raise_analysis_failure(stage, reason, **diagnostic)
    try:
        result, solver_information = brentq(residual_function, lower_bound, upper_bound, xtol=xtol, rtol=rtol,
                              maxiter=150, full_output=True, disp=False)
    except AnalysisFailure:
        raise
    except Exception as exc:
        raise_analysis_failure(stage, f"brentq raised {type(exc).__name__}: {exc}", **diagnostic)
    if not solver_information.converged:
        raise_analysis_failure(stage, "Iteration limit reached.", flag=solver_information.flag, **diagnostic)
    return result


# =============================================================================
# Initialize thermochemistry <-- this instead of CEA wrap
# =============================================================================
class AirPropertyCalculator:
    def __init__(self):
        def load_species_records(filename):
            paths = [Path(data_directory)/filename for data_directory in cantera.get_data_directories()]
            path = next((path for path in paths if path.is_file()), None)
            if path is None:
                raise_analysis_failure("THERMO DATABASE", "Required Cantera data file was not found.",
                     filename=filename, searched=paths)
            data = YAML(typ="safe").load(path.read_text())
            return {record["name"]: record for record in data["species"]}

        # Construct only the needed species, rather than all 748 NASA entries.
        # SPECIES THERMODYNAMICS: NASA polynomial fits, evaluated by Cantera.
        # For a NASA7 record: cp_i^0/Ru = a1+a2*T+a3*T^2+a4*T^3+a5*T^4;
        # h_i^0/(Ru*T) = a1+a2*T/2+a3*T^2/3+a4*T^3/4+a5*T^4/5+a6/T;
        # s_i^0/Ru = a1*ln(T)+a2*T+a3*T^2/2+a4*T^3/3+a5*T^4/4+a7.
        # Use each record's actual thermo model and temperature interval. Enthalpy
        # includes the chemical reference/formation contribution; it is not just cp*T.
        # Source: https://cantera.org/3.1/reference/thermo/species-thermo.html
        nasa = load_species_records("nasa_gas.yaml")
        transport_records = load_species_records("gri30.yaml")
        selected = []
        for name in AIR_SPECIES_NAMES:
            record = dict(nasa[name])
            record["transport"] = transport_records[name.upper()]["transport"]
            species_definition = cantera.Species.from_dict(record)
            selected.append(species_definition)
        # IDEAL-GAS MIXTURE EQUATION OF STATE: p = rho*Rmix*T,
        # Rmix = Ru/Wmix. Ideal gas does NOT mean constant cp or fixed composition.
        # Mixture-averaged transport uses molecular transport data; it does not solve
        # boundary-layer species diffusion or a surface chemical reaction mechanism.
        self.cantera_gas = cantera.Solution(thermo="ideal-gas", species=selected,
                               transport_model="mixture-averaged")
        self.minimum_temperature_K = max(MIN_SUPPORTED_TEMPERATURE_K, self.cantera_gas.min_temp)
        self.maximum_temperature_K = min(MAX_SUPPORTED_TEMPERATURE_K, self.cantera_gas.max_temp)
        # Cantera uses TPY = temperature (K), pressure (Pa), species mass fractions.
        self.cantera_gas.TPY = 298.15, cantera.one_atm, FREESTREAM_SPECIES_MASS_FRACTIONS
        self.freestream_mass_fractions = self.cantera_gas.Y.copy()
        self.freestream_element_mass_fractions = self.get_element_mass_fractions()
        self.equilibrium_retry_count = 0

    # ELEMENTAL MASS CONSERVATION: b_j = sum_i Y_i*n_ji*W_j/W_i.
    # Y_i is species mass fraction, n_ji atoms of element j in species i,
    # and W_j/W_i the atomic/molecular mass ratio. Reactions redistribute species
    # but preserve these elemental fractions in this closed-composition model.
    def get_element_mass_fractions(self):
        return np.array([self.cantera_gas.elemental_mass_fraction(element_index)
                         for element_index in range(self.cantera_gas.n_elements)])

    def validate_gas_state(self, stage):
        cantera_gas = self.cantera_gas
        values = [cantera_gas.T, cantera_gas.P, cantera_gas.density, cantera_gas.cp_mass, cantera_gas.cv_mass,
                  cantera_gas.enthalpy_mass, cantera_gas.entropy_mass]
        if not np.all(np.isfinite(values)) or min(values[:5]) <= 0:
            raise_analysis_failure(stage, "Nonfinite or nonphysical gas properties.", properties=values)
        if not self.minimum_temperature_K - 1e-6 <= cantera_gas.T <= self.maximum_temperature_K + 1e-6:
            raise_analysis_failure(stage, "State exceeds the supported temperature range. "
                 "No constant-cp fallback or polynomial extrapolation is permitted.",
                 temperature_K=cantera_gas.T, supported_K=(self.minimum_temperature_K, self.maximum_temperature_K))
        if not np.all(np.isfinite(cantera_gas.Y)) or np.min(cantera_gas.Y) < -1e-12:
            raise_analysis_failure(stage, "Invalid species mass fractions.")
        if not np.allclose(self.get_element_mass_fractions(), self.freestream_element_mass_fractions, atol=2e-9, rtol=2e-7):
            raise_analysis_failure(stage, "Elemental mass fractions were not conserved.",
                 actual=self.get_element_mass_fractions(), expected=self.freestream_element_mass_fractions)

    def frozen_composition_state(self, temperature_K, pressure_Pa, species_mass_fractions):
        if not np.isfinite(temperature_K) or not self.minimum_temperature_K <= temperature_K <= self.maximum_temperature_K:
            raise_analysis_failure("THERMODYNAMIC RANGE", "Requested temperature is unsupported.",
                 requested_K=temperature_K, supported_K=(self.minimum_temperature_K, self.maximum_temperature_K))
        # FROZEN-COMPOSITION STATE: evaluate properties at (T,p,Y_fixed).
        # No equilibrium composition update occurs here. "Frozen" refers to chemistry,
        # not a fixed temperature or constant specific heats.
        self.cantera_gas.TPY = temperature_K, pressure_Pa, species_mass_fractions
        self.validate_gas_state("FROZEN GAS")
        return self.get_current_properties()

    def calculate_equilibrium_state(self, fixed_property_pair, target_property_value, pressure_Pa, initial_mass_fractions=None, initial_temperature_K=300.0):
        """Hold a property pair fixed: TP=temperature/pressure, HP=enthalpy/pressure,
        SP=entropy/pressure. Retry numerical algorithms using the same physics.
        """
        messages = []
        for solver in ("element_potential", "gibbs", "vcs"):
            try:
                self.cantera_gas.TPY = initial_temperature_K, pressure_Pa, (self.freestream_mass_fractions if initial_mass_fractions is None else initial_mass_fractions)
                setattr(self.cantera_gas, fixed_property_pair, (target_property_value, pressure_Pa))
                # CHEMICAL EQUILIBRIUM: element-constrained equilibrium composition.
                # At fixed T,p this minimizes G = sum_i n_i*chemical_potential_i.
                # HP and SP additionally solve for T to preserve specified h or s at p.
                # The three algorithms are numerical alternatives for the same constraints;
                # none integrates finite-rate reaction kinetics.
                self.cantera_gas.equilibrate(fixed_property_pair, solver=solver, rtol=1e-10,
                                     max_steps=2000, max_iter=300)
                self.validate_gas_state(f"EQUILIBRIUM {fixed_property_pair}")
                if messages:
                    self.equilibrium_retry_count += 1
                    if self.equilibrium_retry_count <= 3:
                        print(f"[EQUILIBRIUM RETRY] {fixed_property_pair} recovered with {solver}; "
                              f"earlier solver: {messages[0]}", flush=True)
                return self.get_current_properties()
            except AnalysisFailure:
                raise
            except cantera.CanteraError as exc:
                messages.append(f"{solver}: {exc}")
        raise_analysis_failure("CHEMICAL EQUILIBRIUM", "All equilibrium algorithms failed. "
             "No ideal-gas-constant-cp fallback was used.", fixed_property_pair=fixed_property_pair,
             target=target_property_value, pressure_Pa=pressure_Pa, solver_errors="\n".join(messages))

    def get_current_properties(self):
        cantera_gas = self.cantera_gas
        # MIXTURE PROPERTY RELATIONS (computed internally by Cantera):
        # h = sum_i Y_i*h_i(T); cp_frozen = sum_i Y_i*cp_i(T).
        # For fixed composition, cv = cp-Rmix and gamma_frozen = cp/cv.
        # FROZEN SOUND SPEED: a^2 = (dp/drho)_(s,Y) = gamma_frozen*Rmix*T.
        # The returned cp/cv and sound speed are local frozen-composition properties,
        # even if this state's composition was obtained from an equilibrium solve.
        # They are not derivatives along an equilibrium-composition path.
        # Source: https://cantera.org/3.1/python/thermo.html
        return dict(temperature_K=cantera_gas.T, pressure_Pa=cantera_gas.P, density_kg_m3=cantera_gas.density, enthalpy_J_kg=cantera_gas.enthalpy_mass,
                    entropy_J_kg_K=cantera_gas.entropy_mass, specific_heat_cp_J_kg_K=cantera_gas.cp_mass, specific_heat_cv_J_kg_K=cantera_gas.cv_mass,
                    sound_speed_m_s=cantera_gas.sound_speed, species_mass_fractions=cantera_gas.Y.copy(), specific_gas_constant_J_kg_K=cantera.gas_constant/cantera_gas.mean_molecular_weight)

    def calculate_transport_properties(self):
        cantera_gas = self.cantera_gas
        dynamic_viscosity_Pa_s, thermal_conductivity_W_m_K, specific_heat_cp_J_kg_K = cantera_gas.viscosity, cantera_gas.thermal_conductivity, cantera_gas.cp_mass
        if not np.all(np.isfinite([dynamic_viscosity_Pa_s, thermal_conductivity_W_m_K, specific_heat_cp_J_kg_K])) or min(dynamic_viscosity_Pa_s, thermal_conductivity_W_m_K, specific_heat_cp_J_kg_K) <= 0:
            raise_analysis_failure("TRANSPORT", "Invalid mixture transport properties.",
                 T_K=cantera_gas.T, p_Pa=cantera_gas.P, viscosity=dynamic_viscosity_Pa_s, thermal_conductivity_W_m_K=thermal_conductivity_W_m_K, specific_heat_cp_J_kg_K=specific_heat_cp_J_kg_K)
        # PRANDTL NUMBER: Pr = mu*cp/k = nu/alpha (dimensionless).
        # mu: dynamic viscosity [Pa s]; k: thermal conductivity [W/(m K)];
        # nu = mu/rho and alpha = k/(rho*cp). First returned value is mu.
        return dynamic_viscosity_Pa_s, dynamic_viscosity_Pa_s * specific_heat_cp_J_kg_K / thermal_conductivity_W_m_K


# =============================================================================
# Standard atmosphere at each CSV altitude; 
# =============================================================================
def calculate_freestream_properties(altitude_m, air_properties):
    # STANDARD-ATMOSPHERE LOOKUP (Ambiance): supplies T and p from altitude.
    # Underlying layered atmosphere combines hydrostatic balance dp/dH = -rho*g0,
    # ideal-gas density, and a prescribed temperature lapse rate T = Tb+L*(H-Hb).
    # The library handles its altitude convention internally; this script supplies
    # Altitude_m directly and does not implement or modify the atmosphere equations.
    standard_atmosphere = Atmosphere(float(altitude_m))
    return air_properties.frozen_composition_state(float(standard_atmosphere.temperature[0]), float(standard_atmosphere.pressure[0]),
                         air_properties.freestream_mass_fractions)


# =============================================================================
# Equilibrium Normal Shock: mass, momentum, and total enthalpy conservation
# =============================================================================
def solve_equilibrium_normal_shock(freestream, freestream_speed_m_s, air_properties):
    freestream_density_kg_m3 = freestream["density_kg_m3"]
    freestream_pressure_Pa = freestream["pressure_Pa"]
    freestream_enthalpy_J_kg = freestream["enthalpy_J_kg"]
    freestream_entropy_J_kg_K = freestream["entropy_J_kg_K"]
    # MACH NUMBER: M_inf = V_inf/a_inf, using the frozen sound speed above.
    freestream_mach_number = freestream_speed_m_s/freestream["sound_speed_m_s"]
    if freestream_mach_number <= 1:
        raise_analysis_failure("SHOCK DOMAIN", "A compressive normal shock requires M > 1.", mach=freestream_mach_number)
    # RANKINE-HUGONIOT JUMP CONDITIONS for a steady, adiabatic normal shock:
    # Mass:     j = rho_inf*V_inf = rho_2*V_2                 [kg/(m^2 s)].
    # Momentum: J = p_inf+rho_inf*V_inf^2 = p_2+rho_2*V_2^2  [Pa].
    # Energy:   h0 = h_inf+V_inf^2/2 = h_2+V_2^2/2           [J/kg].
    # Subscript 2 means downstream of the shock. No constant-gamma jump formula
    # is used for the final state; h includes composition-dependent chemical energy.
    mass_flux_kg_m2_s = freestream_density_kg_m3*freestream_speed_m_s
    momentum_flux_Pa = freestream_pressure_Pa + freestream_density_kg_m3*freestream_speed_m_s**2
    total_enthalpy_J_kg = freestream_enthalpy_J_kg + 0.5*freestream_speed_m_s**2
    # NUMERICAL NORMALIZATION: choose a nonzero enthalpy scale for residuals.
    # The 1 J/kg floor avoids division by zero; it is not added physical energy.
    enthalpy_error_scale_J_kg = max(abs(total_enthalpy_J_kg), abs(freestream_enthalpy_J_kg), 0.5*freestream_speed_m_s**2, 1.0)
    cache = {}

    def calculate_trial_post_shock_state(downstream_speed_ratio):
        if downstream_speed_ratio not in cache:
            # TRIAL-STATE PARAMETERIZATION: r = V_2/V_inf (compression has 0<r<1).
            # Rearrange momentum and energy conservation: p_2 = J-j*V_2,
            # h_2 = h0-V_2^2/2. An equilibrium HP solve supplies rho_2(T,p,Y).
            post_shock_speed_m_s = downstream_speed_ratio*freestream_speed_m_s
            post_shock_pressure_Pa = momentum_flux_Pa - mass_flux_kg_m2_s*post_shock_speed_m_s
            post_shock_enthalpy_J_kg = total_enthalpy_J_kg - 0.5*post_shock_speed_m_s**2
            cache[downstream_speed_ratio] = air_properties.calculate_equilibrium_state("HP", post_shock_enthalpy_J_kg, post_shock_pressure_Pa,
                                          initial_mass_fractions=freestream["species_mass_fractions"], initial_temperature_K=freestream["temperature_K"])
        return cache[downstream_speed_ratio]

    def relative_mass_conservation_residual(downstream_speed_ratio):
        # DIMENSIONLESS CONTINUITY RESIDUAL: f(r) = rho_2(r)*r*V_inf/j - 1.
        # The physical shock root satisfies f(r)=0; r=1 is the unwanted no-shock root.
        return calculate_trial_post_shock_state(downstream_speed_ratio)["density_kg_m3"]*downstream_speed_ratio*freestream_speed_m_s/mass_flux_kg_m2_s - 1.0

    # Constant-gamma expression is used ONLY to seed a bracket, not as physics.
    # CALORICALLY PERFECT-GAS NORMAL-SHOCK VELOCITY RATIO (INITIAL GUESS):
    # r_guess = rho_inf/rho_2 = [(gamma-1)*M_inf^2+2]/[(gamma+1)*M_inf^2].
    # Here gamma is the local upstream cp/cv. This approximate expression only
    # helps bracket the variable-cp equilibrium root; it does not set that root.
    initial_specific_heat_ratio = freestream["specific_heat_cp_J_kg_K"]/freestream["specific_heat_cv_J_kg_K"]
    initial_downstream_speed_ratio = ((initial_specific_heat_ratio-1)*freestream_mach_number**2 + 2)/((initial_specific_heat_ratio+1)*freestream_mach_number**2)
    lower_speed_ratio, upper_speed_ratio = max(1e-4, 0.1*initial_downstream_speed_ratio), 0.5*(1+initial_downstream_speed_ratio)
    lower_ratio_residual, upper_ratio_residual = relative_mass_conservation_residual(lower_speed_ratio), relative_mass_conservation_residual(upper_speed_ratio)
    if lower_ratio_residual*upper_ratio_residual > 0:
        candidates = np.unique(np.r_[np.geomspace(1e-4, 0.9, 60),
                                     1-np.geomspace(1e-7, 0.1, 35)])
        values = [(downstream_speed_ratio, relative_mass_conservation_residual(downstream_speed_ratio)) for downstream_speed_ratio in candidates]
        brackets = [(lower_candidate_ratio, upper_candidate_ratio) for (lower_candidate_ratio, lower_candidate_residual), (upper_candidate_ratio, upper_candidate_residual) in zip(values[:-1], values[1:])
                    if lower_candidate_residual <= 0 <= upper_candidate_residual]
        if not brackets:
            raise_analysis_failure("SHOCK BRACKET", "No compressive equilibrium-shock root found. "
                 "The upstream/no-shock solution is deliberately excluded.",
                 mach=freestream_mach_number, first_residual=lower_ratio_residual, last_residual=upper_ratio_residual)
        lower_speed_ratio, upper_speed_ratio = brackets[0]
    downstream_speed_ratio = solve_bracketed_root(relative_mass_conservation_residual, lower_speed_ratio, upper_speed_ratio, "SHOCK SOLVER", dict(mach=freestream_mach_number))
    post_shock = calculate_trial_post_shock_state(downstream_speed_ratio)
    post_shock_speed_m_s = downstream_speed_ratio*freestream_speed_m_s
    # CONSERVATION DIAGNOSTICS: normalized errors in j, J, and h0 above.
    # The maximum absolute component is an infinity norm, not another flow law.
    relative_conservation_errors = np.array([
        (post_shock["density_kg_m3"]*post_shock_speed_m_s-mass_flux_kg_m2_s)/mass_flux_kg_m2_s,
        (post_shock["pressure_Pa"]+post_shock["density_kg_m3"]*post_shock_speed_m_s**2-momentum_flux_Pa)/momentum_flux_Pa,
        (post_shock["enthalpy_J_kg"]+0.5*post_shock_speed_m_s**2-total_enthalpy_J_kg)/enthalpy_error_scale_J_kg,
    ])
    if np.max(np.abs(relative_conservation_errors)) > CONSERVATION_RELATIVE_TOLERANCE:
        raise_analysis_failure("SHOCK CONSERVATION", "Root returned but conservation checks failed.",
             relative_mass_momentum_energy_errors=relative_conservation_errors, mach=freestream_mach_number)
    # SECOND-LAW / COMPRESSION-BRANCH CHECK: rho_2>rho_inf, p_2>p_inf,
    # s_2>=s_inf (within tolerance). An adiabatic shock increases entropy;
    # it is not an isentropic compression.
    if post_shock["density_kg_m3"] <= freestream_density_kg_m3 or post_shock["pressure_Pa"] <= freestream_pressure_Pa or post_shock["entropy_J_kg_K"] < freestream_entropy_J_kg_K-1e-5:
        raise_analysis_failure("SHOCK BRANCH", "Computed root is not an entropy-admissible compression shock.",
             density_ratio=post_shock["density_kg_m3"]/freestream_density_kg_m3, pressure_ratio=post_shock["pressure_Pa"]/freestream_pressure_Pa,
             entropy_change_J_kgK=post_shock["entropy_J_kg_K"]-freestream_entropy_J_kg_K, mach=freestream_mach_number)
    return post_shock, total_enthalpy_J_kg, float(np.max(np.abs(relative_conservation_errors)))


# ISENTROPIC POST-SHOCK DECELERATION TO THE STAGNATION EDGE:
# s_e = s_2 and h_e = h0 because V_e tends to zero at the stagnation point.
# Composition remains in equilibrium along this assumed inviscid deceleration.
# This is isentropic AFTER the shock; upstream-to-edge entropy is not constant.
def solve_stagnation_edge_state(post_shock, total_enthalpy_J_kg, air_properties):
    cache = {}

    # POSITIVE-PRESSURE ROOT COORDINATE: ell = ln(p / 1 Pa), p = exp(ell) Pa.
    # Using a logarithm keeps trial pressures positive; this is numerical scaling.
    def calculate_state_at_log_pressure(log_pressure_Pa):
        if log_pressure_Pa not in cache:
            cache[log_pressure_Pa] = air_properties.calculate_equilibrium_state(
                "SP", post_shock["entropy_J_kg_K"], np.exp(log_pressure_Pa),
                initial_mass_fractions=post_shock["species_mass_fractions"], initial_temperature_K=post_shock["temperature_K"])
        return cache[log_pressure_Pa]

    enthalpy_error_scale_J_kg = max(abs(total_enthalpy_J_kg), abs(post_shock["enthalpy_J_kg"]), 1.0)

    def residual(log_pressure_Pa):
        # STAGNATION ENTHALPY RESIDUAL: [h(s_2,p)-h0]/h_scale = 0.
        # The SP equilibrium call already enforces s=s_2 at each pressure trial.
        return (calculate_state_at_log_pressure(log_pressure_Pa)["enthalpy_J_kg"]-total_enthalpy_J_kg)/enthalpy_error_scale_J_kg

    # NUMERICAL BRACKET EXPANSION: start at p_2 to 2*p_2; adding ln(2)
    # to the upper log-pressure doubles its physical pressure.
    lower_bound, upper_bound = np.log(post_shock["pressure_Pa"]), np.log(2*post_shock["pressure_Pa"])
    for _ in range(12):
        if residual(upper_bound) >= 0:
            break
        upper_bound += np.log(2.0)
    root = solve_bracketed_root(residual, lower_bound, upper_bound, "STAGNATION EDGE", dict(h0_J_kg=total_enthalpy_J_kg))
    stagnation_edge = calculate_state_at_log_pressure(root)
    # NORMALIZED EDGE CHECKS: |h_e-h0|/h_scale and |s_e-s_2|/s_scale.
    # These test the two imposed constraints independently after root finding.
    relative_energy_error = abs(stagnation_edge["enthalpy_J_kg"]-total_enthalpy_J_kg)/enthalpy_error_scale_J_kg
    relative_entropy_error = abs(stagnation_edge["entropy_J_kg_K"]-post_shock["entropy_J_kg_K"])/max(abs(post_shock["entropy_J_kg_K"]), 1.0)
    if max(relative_energy_error, relative_entropy_error) > CONSERVATION_RELATIVE_TOLERANCE:
        raise_analysis_failure("EDGE CONSERVATION", "Isentropic stagnation-state checks failed.",
             relative_energy_error=relative_energy_error, relative_entropy_error=relative_entropy_error)
    air_properties.frozen_composition_state(stagnation_edge["temperature_K"], stagnation_edge["pressure_Pa"], stagnation_edge["species_mass_fractions"])
    stagnation_edge["dynamic_viscosity_Pa_s"], stagnation_edge["prandtl_number"] = air_properties.calculate_transport_properties()
    stagnation_edge["conservation_error"] = max(relative_energy_error, relative_entropy_error)
    return stagnation_edge


# =============================================================================
# Catalytic / Noncatalytic Fay-Riddell-prefactor wall heat balance
# =============================================================================
def calculate_catalytic_driving_enthalpy(total_enthalpy_J_kg, frozen_wall_enthalpy_J_kg, equilibrium_wall_enthalpy_J_kg, lewis_number=effective_lewis_number):
    # CATALYTIC FROZEN-BOUNDARY-LAYER FILM ENTHALPY MODEL (retained approximation):
    # Delta_h_cat = (h0-h_w,frozen) + Le^n*(h_w,frozen-h_w,equilibrium), n=0.63.
    # First term: sensible cooling at frozen edge composition.
    # Second term: an effective recombination contribution at the same wall T,p.
    # For a noncatalytic wall only the first term is used (see below).
    # This is the implemented engineering analogy, not a finite-rate surface law
    # or an assertion that the complete classical Fay-Riddell solution is solved.
    return (total_enthalpy_J_kg-frozen_wall_enthalpy_J_kg) + lewis_number**mass_transfer_exponent*(frozen_wall_enthalpy_J_kg-equilibrium_wall_enthalpy_J_kg)


def solve_wall_temperatures(stagnation_edge, freestream, total_enthalpy_J_kg, air_properties, failure_context):
    # STAGNATION PRESSURE DIFFERENCE: Delta_p = p_e-p_inf.
    stagnation_pressure_rise_Pa = stagnation_edge["pressure_Pa"]-freestream["pressure_Pa"]
    if stagnation_pressure_rise_Pa <= 0 or not np.isfinite(stagnation_pressure_rise_Pa):
        raise_analysis_failure("VELOCITY GRADIENT", "Stagnation pressure must exceed ambient pressure.",
             pressure_difference_Pa=stagnation_pressure_rise_Pa, **failure_context)
    # Spherical nose; do not invent a cosine correction for angle of attack.
    # SPHERICAL-NOSE STAGNATION VELOCITY-GRADIENT APPROXIMATION:
    # (du_e/ds)_stag = sqrt[2*(p_e-p_inf)/rho_e] / R_n, units 1/s.
    # s is distance along the surface. This pressure-based engineering estimate
    # supplies the gradient required by the heating correlation; it is not a
    # resolved external-flow velocity field or the exact potential-flow sphere law.
    stagnation_velocity_gradient_per_s = np.sqrt(2*stagnation_pressure_rise_Pa/stagnation_edge["density_kg_m3"])/nose_radius_m
    cache = {}

    def calculate_wall_gas_states(wall_temperature_K):
        if wall_temperature_K not in cache:
            # WALL GAS STATES AT COMMON PRESSURE (thin-boundary-layer approximation):
            # p_w = p_e. Frozen wall: Y_w = Y_e at T_w.
            # Catalytic limit: Y_w = Y_equilibrium(T_w,p_e) with the same elements.
            # These are GAS properties adjacent to the wall, not solid material properties.
            frozen_composition_state = air_properties.frozen_composition_state(wall_temperature_K, stagnation_edge["pressure_Pa"], stagnation_edge["species_mass_fractions"])
            frozen_composition_state["dynamic_viscosity_Pa_s"], _ = air_properties.calculate_transport_properties()
            equilibrium_wall_state = air_properties.calculate_equilibrium_state("TP", wall_temperature_K, stagnation_edge["pressure_Pa"], initial_mass_fractions=air_properties.freestream_mass_fractions,
                                            initial_temperature_K=wall_temperature_K)
            equilibrium_wall_state["dynamic_viscosity_Pa_s"], _ = air_properties.calculate_transport_properties()
            cache[wall_temperature_K] = frozen_composition_state, equilibrium_wall_state
        return cache[wall_temperature_K]

    def calculate_wall_heat_fluxes(wall_temperature_K, is_catalytic_wall):
        frozen_composition_state, equilibrium_wall_state = calculate_wall_gas_states(wall_temperature_K)
        wall_gas_state = equilibrium_wall_state if is_catalytic_wall else frozen_composition_state
        # FAY-RIDDELL-TYPE LAMINAR STAGNATION HEAT-TRANSFER PREFACTOR:
        # C_h = 0.76*Pr_e^(-0.6)*(rho_e*mu_e)^0.4*(rho_w*mu_w)^0.1
        #       *sqrt[(du_e/ds)_stag], in kg/(m^2 s).
        # Combined below with the retained frozen-film driving enthalpy. The numerical
        # coefficient 0.76 and the wall-state choice are preserved from the input code.
        # C_h is NOT a temperature-based h coefficient in W/(m^2 K).
        heat_transfer_prefactor_kg_m2_s = (0.76*stagnation_edge["prandtl_number"]**-0.6*(stagnation_edge["density_kg_m3"]*stagnation_edge["dynamic_viscosity_Pa_s"])**0.4
                     *(wall_gas_state["density_kg_m3"]*wall_gas_state["dynamic_viscosity_Pa_s"])**0.1*np.sqrt(stagnation_velocity_gradient_per_s))
        # SENSIBLE ENTHALPY DIFFERENCE: Delta_h_s = h0-h(T_w,p_e,Y_e).
        sensible_driving_enthalpy_J_kg = total_enthalpy_J_kg-frozen_composition_state["enthalpy_J_kg"]
        # EFFECTIVE RECOMBINATION ENTHALPY: Delta_h_rec = h_w,frozen-h_w,eq.
        # Same T_w and p_e; the difference arises from the two compositions.
        recombination_enthalpy_J_kg = frozen_composition_state["enthalpy_J_kg"]-equilibrium_wall_state["enthalpy_J_kg"]
        total_driving_enthalpy_J_kg = (calculate_catalytic_driving_enthalpy(total_enthalpy_J_kg, frozen_composition_state["enthalpy_J_kg"], equilibrium_wall_state["enthalpy_J_kg"])
                   if is_catalytic_wall else sensible_driving_enthalpy_J_kg)
        # ENTHALPY-BASED FILM HEATING: q_conv = C_h*Delta_h [W/m^2].
        # Delta_h is Delta_h_cat or Delta_h_s depending on surface chemistry.
        convective_heat_flux_W_m2 = heat_transfer_prefactor_kg_m2_s*total_driving_enthalpy_J_kg
        # STEFAN-BOLTZMANN LAW, gray surface with no incoming radiation:
        # q_rad = epsilon*sigma*T_w^4 [W/m^2], positive out of the wall.
        # No surroundings-temperature fourth-power term is included in this code.
        radiative_heat_flux_W_m2 = wall_emissivity*stefan_boltzmann_constant_W_m2_K4*wall_temperature_K**4
        values = [convective_heat_flux_W_m2, radiative_heat_flux_W_m2, heat_transfer_prefactor_kg_m2_s, sensible_driving_enthalpy_J_kg, recombination_enthalpy_J_kg]
        if not np.all(np.isfinite(values)):
            raise_analysis_failure("WALL RESIDUAL", "Nonfinite heat-transfer terms.", Tw_K=wall_temperature_K,
                 is_catalytic_wall=is_catalytic_wall, terms=values, **failure_context)
        return convective_heat_flux_W_m2, radiative_heat_flux_W_m2, sensible_driving_enthalpy_J_kg, recombination_enthalpy_J_kg

    results = []
    for is_catalytic_wall in (True, False):
        label = "fully catalytic equilibrium wall" if is_catalytic_wall else "noncatalytic frozen wall"

        def wall_heat_balance_residual(wall_temperature_K):
            convective_heat_flux_W_m2, radiative_heat_flux_W_m2, _, _ = calculate_wall_heat_fluxes(wall_temperature_K, is_catalytic_wall)
            # INSTANTANEOUS RADIATIVE-EQUILIBRIUM WALL ENERGY BALANCE:
            # f(T_w) = q_conv(T_w)-q_rad(T_w) = 0.
            # There is no rho_s*cp_s*dT_w/dt storage term or solid conduction term;
            # CSV time labels independent equilibrium calculations, not thermal history.
            return convective_heat_flux_W_m2-radiative_heat_flux_W_m2

        # WALL ROOT SEARCH DOMAIN: supported gas-data minimum <= T_w <= T_e.
        # This is a model/search restriction, not proof that every condition has a root.
        minimum_wall_temperature_K, maximum_wall_temperature_K = air_properties.minimum_temperature_K, stagnation_edge["temperature_K"]
        wall_failure_context = dict(failure_context, wall_model=label, edge_temperature_K=stagnation_edge["temperature_K"],
                   edge_pressure_Pa=stagnation_edge["pressure_Pa"], h0_J_kg=total_enthalpy_J_kg,
                   gradient_per_s=stagnation_velocity_gradient_per_s, edge_Pr=stagnation_edge["prandtl_number"], effective_lewis_number=effective_lewis_number)
        wall_temperature_K = solve_bracketed_root(wall_heat_balance_residual, minimum_wall_temperature_K, maximum_wall_temperature_K, "WALL TEMPERATURE", wall_failure_context, xtol=1e-5)
        convective_heat_flux_W_m2, radiative_heat_flux_W_m2, sensible_driving_enthalpy_J_kg, recombination_enthalpy_J_kg = calculate_wall_heat_fluxes(wall_temperature_K, is_catalytic_wall)
        # NORMALIZED WALL ENERGY RESIDUAL: |q_conv-q_rad|/max(|q_conv|,|q_rad|,1).
        # The 1 W/m^2 floor avoids a zero denominator at small heat flux.
        relative_heat_balance_error = abs(convective_heat_flux_W_m2-radiative_heat_flux_W_m2)/max(abs(convective_heat_flux_W_m2), abs(radiative_heat_flux_W_m2), 1.0)
        if relative_heat_balance_error > 2e-6:
            raise_analysis_failure("WALL ENERGY BALANCE", "Temperature root returned but heat balance failed.",
                 wall_temperature_K=wall_temperature_K, q_in_W_m2=convective_heat_flux_W_m2, q_out_W_m2=radiative_heat_flux_W_m2, **wall_failure_context)
        # At a converged wall balance q_rad equals q_conv within tolerance.
        # The reported heat_flux_W_m2 is q_rad; net wall heat input is approximately zero.
        results.append(WallHeatingResult(wall_temperature_K, radiative_heat_flux_W_m2, relative_heat_balance_error, sensible_driving_enthalpy_J_kg, recombination_enthalpy_J_kg))
    return results


# =============================================================================
# Prescribed trajectory input
# =============================================================================
CSV_COLUMN_MAP = {
    "Time_s": "time_s",
    "Altitude_m": "altitude_m",
    "Speed_mps": "speed_m_s",
    "Mach": "source_mach_number",
    "Range_km": "source_range_km",
    "FlightPath_deg": "source_flight_path_angle_deg",
    "Alpha_deg": "source_angle_of_attack_deg",
    "DynamicPressure_kPa": "source_dynamic_pressure_kPa",
    "LiftLoad_g": "source_lift_load_g",
    "CL": "source_lift_coefficient",
    "CD": "source_drag_coefficient",
}


def load_trajectory_csv(csv_path: Path, segment: str = "all") -> tuple[dict, dict]:
    """Read exact CSV samples; return arrays and JSON-serializable provenance.

    Do NOT sort by altitude: one altitude can occur on different trajectory
    branches with different speeds. Time establishes the sample order.
    """
    csv_path = Path(csv_path).expanduser().resolve()
    if segment not in {"all", "descent"}:
        raise_analysis_failure("TRAJECTORY INPUT", "Unknown segment.", segment=segment)
    if not csv_path.is_file():
        raise_analysis_failure(
            "TRAJECTORY INPUT",
            "Trajectory CSV not found. Put it beside this script or use --trajectory-csv.",
            searched_path=str(csv_path),
        )
    try:
        # Round-trip parsing preserves the provided floating-point samples.
        table = pd.read_csv(
            csv_path, encoding="utf-8-sig", float_precision="round_trip",
            on_bad_lines="error", skip_blank_lines=False,
        )
    except (OSError, UnicodeError, ValueError, pd.errors.ParserError) as exc:
        raise_analysis_failure("TRAJECTORY INPUT", f"Cannot read CSV: {exc}", path=str(csv_path))

    table.columns = [str(name).strip() for name in table.columns]
    if len(set(table.columns)) != len(table.columns):
        raise_analysis_failure("TRAJECTORY INPUT", "Duplicate CSV column names after trimming spaces.")
    missing = [name for name in ("Time_s", "Altitude_m", "Speed_mps") if name not in table.columns]
    if missing:
        raise_analysis_failure(
            "TRAJECTORY INPUT", "Required CSV columns are missing.",
            missing=missing, available=list(table.columns),
        )
    if len(table) < 2:
        raise_analysis_failure("TRAJECTORY INPUT", "Need at least two trajectory rows.", rows=len(table))

    inputs = {}
    for csv_name, output_name in CSV_COLUMN_MAP.items():
        if csv_name not in table.columns:
            continue
        try:
            values = pd.to_numeric(table[csv_name], errors="raise").to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise_analysis_failure("TRAJECTORY INPUT", f"Nonnumeric data: {exc}", column=csv_name)
        bad = np.flatnonzero(~np.isfinite(values))
        if len(bad):
            raise_analysis_failure(
                "TRAJECTORY INPUT", "NaN, infinity, or a missing value in an input column.",
                column=csv_name, source_csv_row=int(bad[0]) + 2,
            )
        inputs[output_name] = values.copy()

    time_steps = np.diff(inputs["time_s"])
    bad_time = np.flatnonzero(time_steps <= 0)
    if len(bad_time):
        first = int(bad_time[0])
        raise_analysis_failure(
            "TRAJECTORY INPUT",
            "Time_s must be strictly increasing in CSV order. No automatic sorting or deduplication is performed.",
            source_csv_rows=[first + 2, first + 3],
            times_s=inputs["time_s"][first:first + 2].tolist(),
        )
    bad_speed = np.flatnonzero(inputs["speed_m_s"] < 0)
    if len(bad_speed):
        raise_analysis_failure(
            "TRAJECTORY INPUT", "Speed_mps must be a nonnegative speed magnitude.",
            source_csv_row=int(bad_speed[0]) + 2,
        )

    total_rows = len(table)
    inputs["source_csv_row"] = np.arange(2, total_rows + 2, dtype=int)
    peak_index = int(np.argmax(inputs["altitude_m"]))
    start_index = peak_index if segment == "descent" else 0
    if total_rows - start_index < 2:
        raise_analysis_failure(
            "TRAJECTORY INPUT", "Fewer than two samples remain in the selected segment.",
            segment=segment, peak_source_csv_row=peak_index + 2,
        )
    inputs = {name: values[start_index:].copy() for name, values in inputs.items()}
    altitude_steps = np.diff(inputs["altitude_m"])
    try:
        fingerprint = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    except OSError as exc:
        raise_analysis_failure("TRAJECTORY INPUT", f"Cannot fingerprint input CSV: {exc}")

    info = {
        "source": "prescribed_csv",
        "csv_path": str(csv_path),
        "csv_sha256": fingerprint,
        "csv_input_rows": total_rows,
        "selected_rows": len(inputs["time_s"]),
        "segment": segment,
        "first_source_csv_row": int(inputs["source_csv_row"][0]),
        "last_source_csv_row": int(inputs["source_csv_row"][-1]),
        "maximum_altitude_source_csv_row": peak_index + 2,
        "contains_climb_and_descent": bool(np.any(altitude_steps > 0) and np.any(altitude_steps < 0)),
        "time_range_s": [float(inputs["time_s"][0]), float(inputs["time_s"][-1])],
        "altitude_range_m": [float(np.min(inputs["altitude_m"])), float(np.max(inputs["altitude_m"]))],
        "speed_range_m_s": [float(np.min(inputs["speed_m_s"])), float(np.max(inputs["speed_m_s"]))],
        "imported_columns": {name: dest for name, dest in CSV_COLUMN_MAP.items() if name in table.columns},
        "unused_csv_columns": [name for name in table.columns if name not in CSV_COLUMN_MAP],
        "sample_handling": "Original time order; no sorting, interpolation, extrapolation, or speed integration.",
        "speed_reference_assumption": "Speed_mps is freestream-relative speed, not ground/inertial speed.",
        "altitude_handling": "Altitude_m passed unchanged to the existing atmosphere model.",
        "mach_used_by_heating": "Imported speed / local sound speed from the existing air model.",
    }
    return inputs, info


# =============================================================================
# Outputs, failure preservation, and plotting
# =============================================================================
def save_results(output_directory: Path, profile_arrays: dict, run_summary: dict) -> None:
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_directory / "results.npz", **profile_arrays)
    with (output_directory / "run_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(run_summary, stream, indent=2, allow_nan=False)
        stream.write("\n")
    row_count = len(profile_arrays["time_s"])
    columns = {}
    for name, values in profile_arrays.items():
        # Composition is in results.npz; all scalar-per-sample data are in CSV.
        if name in {"species_names", "edge_species_mass_fractions"}:
            continue
        values = np.asarray(values)
        if values.shape == (row_count,):
            columns[name] = values
        elif values.shape == (row_count, 1):
            columns[name] = values[:, 0]
        else:
            raise_analysis_failure("OUTPUT", "Unexpected output array shape.", name=name, shape=values.shape)
    pd.DataFrame(columns).to_csv(
        output_directory / "profile.csv", index=False, float_format="%.17g", na_rep="NaN",
    )


def run_analysis(options):
    values = [nose_radius_m, effective_lewis_number, wall_emissivity,
              stefan_boltzmann_constant_W_m2_K4, mass_transfer_exponent]
    if (not np.all(np.isfinite(values))
            or min(nose_radius_m, effective_lewis_number, stefan_boltzmann_constant_W_m2_K4) <= 0
            or not 0 < wall_emissivity <= 1):
        raise_analysis_failure("INPUT", "Invalid nose radius, Lewis number, emissivity, or thermal-radiation constant.")

    profile_arrays, trajectory_info = load_trajectory_csv(options.trajectory_csv, options.segment)
    altitude_m = profile_arrays["altitude_m"]
    speed_m_s = profile_arrays["speed_m_s"]
    time_s = profile_arrays["time_s"]
    sample_count = len(time_s)

    print(f"[INITIALIZE] Cantera {cantera.__version__}; variable cp and equilibrium chemistry.", flush=True)
    air_properties = AirPropertyCalculator()
    print(f"[GAS MODEL] {', '.join(AIR_SPECIES_NAMES)}; supported "
          f"{air_properties.minimum_temperature_K:g}--{air_properties.maximum_temperature_K:g} K.", flush=True)
    print("[WALL MODEL] Original frozen-boundary-layer film analogy; catalytic and noncatalytic limits. "
          "Spherical-nose stagnation point; no added AoA correction.", flush=True)
    print(f"[TRAJECTORY] CSV: {trajectory_info['csv_path']}", flush=True)
    print(f"[TRAJECTORY] Using {sample_count}/{trajectory_info['csv_input_rows']} rows; "
          f"segment={options.segment}; t={time_s[0]:.6g}--{time_s[-1]:.6g} s. "
          "No drag/gravity integration or trajectory resampling.", flush=True)
    print("[INPUT ASSUMPTION] Speed_mps is air-relative speed; Altitude_m is used without a datum conversion.", flush=True)
    if trajectory_info["contains_climb_and_descent"]:
        print("[TRAJECTORY] Both climb and descent are present; original time order is preserved.", flush=True)
    if "source_mach_number" in profile_arrays:
        print("[MACH] CSV Mach is retained as source_mach_number. Heating uses imported speed / local sound speed.", flush=True)
    if trajectory_info["unused_csv_columns"]:
        print(f"[TRAJECTORY] Unused additional columns: {trajectory_info['unused_csv_columns']}", flush=True)

    scalar_outputs = (
        "mach_number", "freestream_temperature_K", "freestream_pressure_Pa",
        "freestream_density_kg_m3", "freestream_sound_speed_m_s", "freestream_dynamic_pressure_Pa",
        "edge_pressure_Pa", "edge_temperature_K", "edge_density_kg_m3",
        "edge_total_enthalpy_J_kg", "edge_specific_heat_cp_J_kg_K", "edge_prandtl_number",
        "shock_conservation_error", "edge_conservation_error", "wall_heat_balance_error",
        "nose_radius_knudsen_number",
    )
    for name in scalar_outputs:
        profile_arrays[name] = np.full(sample_count, np.nan)
    for name in ("wall_temperature_K_fully_catalytic", "wall_temperature_K_noncatalytic",
                 "heat_flux_W_m2_fully_catalytic", "heat_flux_W_m2_noncatalytic"):
        # Preserve the old [:, 0] access pattern, without 100 identical AoA columns.
        profile_arrays[name] = np.full((sample_count, 1), np.nan)
    profile_arrays["edge_species_mass_fractions"] = np.full((sample_count, len(AIR_SPECIES_NAMES)), np.nan)
    profile_arrays["species_names"] = np.asarray(AIR_SPECIES_NAMES)
    profile_arrays["sample_status"] = np.full(sample_count, "pending", dtype="U32")
    run_summary = {
        "status": "running", "cantera_version": cantera.__version__,
        "trajectory": trajectory_info, "requested_samples": sample_count, "returned_samples": sample_count,
        "completed_heating_samples": 0, "skipped_mach_le_1_samples": 0,
        "model": "equilibrium neutral air + frozen-BL catalytic film analogy",
        "species": AIR_SPECIES_NAMES, "effective_lewis_number": effective_lewis_number,
        "catalytic_exponent": mass_transfer_exponent, "nose_radius_m": nose_radius_m,
        "emissivity": wall_emissivity,
        "supported_temperature_K": [air_properties.minimum_temperature_K, air_properties.maximum_temperature_K],
        "wall_array_shape": "(trajectory_sample, single_stagnation_point)",
        "plot_x": options.plot_x,
        "limitations": [
            "Input trajectory and aerodynamic model are prescribed, not validated by this script.",
            "Time is not used to integrate material temperature: walls are in instantaneous radiative equilibrium.",
            "No finite-rate chemistry, ionization, conduction, ablation, or incoming radiation.",
            "Mach <= 1 rows are retained with NaN heating and an explicit skipped status.",
        ],
    }
    started = time.monotonic()
    low_mach_notice = False
    kn_notice = False
    subsonic_notice = False
    sample_index = 0
    try:
        for sample_index in range(sample_count):
            failure_context = {
                "index": sample_index,
                "source_csv_row": int(profile_arrays["source_csv_row"][sample_index]),
                "time_s": float(time_s[sample_index]),
                "altitude_m": float(altitude_m[sample_index]),
                "speed_m_s": float(speed_m_s[sample_index]),
            }
            try:
                freestream = calculate_freestream_properties(altitude_m[sample_index], air_properties)
                # MACH NUMBER AGAIN AT EACH CSV SAMPLE: M = V_csv/a_frozen.
                # Imported CSV Mach is retained separately, not used to change speed.
                mach_number = speed_m_s[sample_index] / freestream["sound_speed_m_s"]
                failure_context["mach"] = float(mach_number)
                for name, value in {
                    "mach_number": mach_number,
                    "freestream_temperature_K": freestream["temperature_K"],
                    "freestream_pressure_Pa": freestream["pressure_Pa"],
                    "freestream_density_kg_m3": freestream["density_kg_m3"],
                    "freestream_sound_speed_m_s": freestream["sound_speed_m_s"],
                    # DYNAMIC PRESSURE: q_dynamic = (1/2)*rho_inf*V_inf^2 [Pa].
                    # This is neither stagnation pressure nor aerodynamic heat flux.
                    "freestream_dynamic_pressure_Pa": 0.5 * freestream["density_kg_m3"] * speed_m_s[sample_index]**2,
                }.items():
                    profile_arrays[name][sample_index] = value

                if mach_number <= 1.0:
                    profile_arrays["sample_status"][sample_index] = "skipped_mach_le_1"
                    run_summary["skipped_mach_le_1_samples"] += 1
                    if not subsonic_notice:
                        print(f"[MODEL DOMAIN] M<=1 at t={time_s[sample_index]:.6g} s. "
                              "These rows retain NaN heating; later supersonic rows will still be processed.", flush=True)
                        subsonic_notice = True
                    continue
                if mach_number < 3 and not low_mach_notice:
                    print(f"[MODEL LIMITATION] M<3 at t={time_s[sample_index]:.6g} s; "
                          "near-sonic heating is outside established hypersonic usage. "
                          "Imported speed is not changed or clamped.", flush=True)
                    low_mach_notice = True

                air_properties.frozen_composition_state(
                    freestream["temperature_K"], freestream["pressure_Pa"], freestream["species_mass_fractions"],
                )
                freestream_viscosity_Pa_s, _ = air_properties.calculate_transport_properties()
                # VISCOSITY-BASED MEAN-FREE-PATH ESTIMATE AND KNUDSEN NUMBER:
                # lambda ~= (mu_inf/p_inf)*sqrt(pi*Rmix*T_inf/2); Kn_Rn = lambda/R_n.
                # This is a kinetic-theory engineering estimate for the freestream mixture.
                # Kn >= 0.01 is a continuum/no-slip screening threshold, not a correction
                # applied to the heating or an exact reacting-mixture collision calculation.
                nose_radius_knudsen_number = (
                    freestream_viscosity_Pa_s / freestream["pressure_Pa"]
                    * np.sqrt(np.pi * freestream["specific_gas_constant_J_kg_K"] * freestream["temperature_K"] / 2)
                    / nose_radius_m
                )
                if nose_radius_knudsen_number >= 0.01 and not kn_notice:
                    print(f"[CONTINUUM LIMITATION] Estimated freestream Kn(Rn)={nose_radius_knudsen_number:.4g}; "
                          "continuum/no-slip heating needs assessment.", flush=True)
                    kn_notice = True

                # CSV speed and altitude feed the ORIGINAL shock and wall functions.
                post_shock, total_enthalpy_J_kg, shock_conservation_error = solve_equilibrium_normal_shock(
                    freestream, speed_m_s[sample_index], air_properties,
                )
                stagnation_edge = solve_stagnation_edge_state(post_shock, total_enthalpy_J_kg, air_properties)
                fully_catalytic_result, noncatalytic_result = solve_wall_temperatures(
                    stagnation_edge, freestream, total_enthalpy_J_kg, air_properties, failure_context,
                )
                for name, value in {
                    "edge_pressure_Pa": stagnation_edge["pressure_Pa"],
                    "edge_temperature_K": stagnation_edge["temperature_K"],
                    "edge_density_kg_m3": stagnation_edge["density_kg_m3"],
                    "edge_total_enthalpy_J_kg": total_enthalpy_J_kg,
                    "edge_specific_heat_cp_J_kg_K": stagnation_edge["specific_heat_cp_J_kg_K"],
                    "edge_prandtl_number": stagnation_edge["prandtl_number"],
                    "shock_conservation_error": shock_conservation_error,
                    "edge_conservation_error": stagnation_edge["conservation_error"],
                    "wall_heat_balance_error": max(fully_catalytic_result.relative_heat_balance_error,
                                                   noncatalytic_result.relative_heat_balance_error),
                    "nose_radius_knudsen_number": nose_radius_knudsen_number,
                }.items():
                    profile_arrays[name][sample_index] = value
                profile_arrays["edge_species_mass_fractions"][sample_index] = stagnation_edge["species_mass_fractions"]
                for label, result in (("fully_catalytic", fully_catalytic_result), ("noncatalytic", noncatalytic_result)):
                    profile_arrays["wall_temperature_K_" + label][sample_index, 0] = result.temperature_K
                    profile_arrays["heat_flux_W_m2_" + label][sample_index, 0] = result.heat_flux_W_m2
                profile_arrays["sample_status"][sample_index] = "completed"
                run_summary["completed_heating_samples"] += 1
                if sample_index == 0 or (sample_index + 1) % 50 == 0 or sample_index == sample_count - 1:
                    print(f"[PROGRESS] {sample_index+1}/{sample_count}: t={time_s[sample_index]:.3f} s, "
                          f"z={altitude_m[sample_index]:.1f} m, V={speed_m_s[sample_index]:.3f} m/s, "
                          f"M={mach_number:.3f}, Te={stagnation_edge['temperature_K']:.1f} K, "
                          f"Tw(cat/non)={fully_catalytic_result.temperature_K:.1f}/{noncatalytic_result.temperature_K:.1f} K; "
                          f"elapsed={time.monotonic()-started:.1f}s", flush=True)
            except AnalysisFailure as exc:
                raise_analysis_failure("TRAJECTORY SAMPLE", str(exc), **failure_context)
            except Exception as exc:
                raise_analysis_failure("SOFTWARE/PROPERTY FAILURE", f"{type(exc).__name__}: {exc}", **failure_context)
    except AnalysisFailure as exc:
        profile_arrays["sample_status"][sample_index] = "failed"
        run_summary.update(status="failed", failure=str(exc))
        save_results(options.output_directory, profile_arrays, run_summary)
        print(f"[PARTIAL OUTPUT] Saved {run_summary['completed_heating_samples']} completed heating samples. "
              f"Unsolved values remain NaN: {options.output_directory}", flush=True)
        raise

    solved = run_summary["completed_heating_samples"]
    skipped = run_summary["skipped_mach_le_1_samples"]
    status = "no_supersonic_samples" if solved == 0 else ("completed_with_skipped_samples" if skipped else "completed")
    run_summary.update(status=status, equilibrium_retry_count=air_properties.equilibrium_retry_count)
    for output_name, array_name in (
        ("max_relative_shock_error", "shock_conservation_error"),
        ("max_relative_edge_error", "edge_conservation_error"),
        ("max_relative_wall_error", "wall_heat_balance_error"),
    ):
        run_summary[output_name] = float(np.nanmax(profile_arrays[array_name])) if solved else None
    if "source_mach_number" in profile_arrays:
        run_summary["max_absolute_mach_difference_from_csv"] = float(np.max(np.abs(
            profile_arrays["mach_number"] - profile_arrays["source_mach_number"],
        )))
    if solved:
        for label in ("fully_catalytic", "noncatalytic"):
            peak_temperature_index = int(np.nanargmax(profile_arrays["wall_temperature_K_" + label][:, 0]))
            peak_flux_index = int(np.nanargmax(profile_arrays["heat_flux_W_m2_" + label][:, 0]))
            peak = {
                "temperature_K": float(profile_arrays["wall_temperature_K_" + label][peak_temperature_index, 0]),
                "time_s": float(time_s[peak_temperature_index]),
                "altitude_m": float(altitude_m[peak_temperature_index]),
                "source_csv_row": int(profile_arrays["source_csv_row"][peak_temperature_index]),
                "max_heat_flux_W_m2": float(profile_arrays["heat_flux_W_m2_" + label][peak_flux_index, 0]),
                "max_heat_flux_time_s": float(time_s[peak_flux_index]),
                "max_heat_flux_altitude_m": float(altitude_m[peak_flux_index]),
            }
            run_summary["peak_" + label] = peak
            print(f"[RESULT {label}] Max Tw={peak['temperature_K']:.2f} K at t={peak['time_s']:.3f} s; "
                  f"max heat flux={peak['max_heat_flux_W_m2']:.6e} W/m^2.", flush=True)
    save_results(options.output_directory, profile_arrays, run_summary)
    if solved:
        print(f"[SUCCESS] {solved} heating samples passed conservation and wall-energy checks; "
              f"{skipped} Mach<=1 rows skipped. Saved to {Path(options.output_directory).resolve()}", flush=True)
    else:
        print(f"[MODEL DOMAIN] No supersonic samples; input and skipped statuses saved to "
              f"{Path(options.output_directory).resolve()}", flush=True)
    return profile_arrays


def plot_results(profile_arrays, options):
    import matplotlib
    if options.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    use_time = options.plot_x == "time"
    x = profile_arrays["time_s"] if use_time else profile_arrays["altitude_m"]
    x_label = "Time (s)" if use_time else "Altitude (m)"
    output_directory = Path(options.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    figures = []
    for key, vertical_axis_label, filename in (
        ("heat_flux_W_m2_", "Stagnation heat flux (W/m^2)", "heat_flux.png"),
        ("wall_temperature_K_", "Radiative-equilibrium wall temperature (K)", "wall_temperature.png"),
    ):
        figure, axes = plt.subplots(figsize=(9, 5.5))
        figures.append(figure)
        axes.plot(x, profile_arrays[key + "fully_catalytic"][:, 0],
                  linewidth=2, label="Fully catalytic equilibrium wall")
        axes.plot(x, profile_arrays[key + "noncatalytic"][:, 0],
                  linestyle="--", linewidth=2, label="Noncatalytic frozen wall")
        axes.set_xlabel(x_label)
        axes.set_ylabel(vertical_axis_label)
        axes.set_title(
            "Wall Temperature vs. Time (NOT transient)"
            if key == "wall_temperature_K_"
            else "Stagnation Point Heat Flux vs. Time (NOT transient)"
        )
        # Only reverse an altitude axis for a consistently descending segment.
        if not use_time and np.all(np.diff(x) <= 0) and np.any(np.diff(x) < 0):
            axes.invert_xaxis()
        if not np.any(np.isfinite(profile_arrays[key + "fully_catalytic"])):
            axes.text(0.5, 0.5, "No supported supersonic heating samples", ha="center", transform=axes.transAxes)
        axes.legend()
        axes.grid(True, alpha=0.3)
        figure.tight_layout()
        figure.savefig(output_directory / filename, dpi=160)
    if not options.no_show:
        plt.show()
    for figure in figures:
        plt.close(figure)


def main():
    script_directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="UCAH heating on a prescribed CSV trajectory (no trajectory integration).")
    parser.add_argument(
        "--trajectory-csv", type=Path, default=script_directory / "glide_result_collocation.csv",
        help="Input CSV with Time_s, Altitude_m, and Speed_mps. Default: beside this script.",
    )
    parser.add_argument(
        "--segment", choices=("all", "descent"), default="all",
        help="Use all CSV rows, or start at the greatest sampled altitude and retain subsequent rows.",
    )
    parser.add_argument("--plot-x", choices=("time", "altitude"), default="time", help="Figure x axis (default: time).")
    parser.add_argument("--no-show", action="store_true", help="Save figures without opening plot windows.")
    parser.add_argument("--skip-plots", action="store_true", help="Save numerical results only.")
    parser.add_argument("--output-dir", dest="output_directory", type=Path, default=script_directory / "ucah_csv_results")
    options = parser.parse_args()
    options.output_directory = options.output_directory.expanduser().resolve()
    try:
        profile_arrays = run_analysis(options)
        if not options.skip_plots:
            plot_results(profile_arrays, options)
    except AnalysisFailure as exc:
        print(f"\n[ANALYSIS FAILED]\n{exc}", file=sys.stderr, flush=True)
        return 1
    except Exception as exc:
        print(f"[UNEXPECTED SOFTWARE FAILURE] {type(exc).__name__}: {exc}\n"
              "This is not physical nonconvergence; inspect the traceback.", file=sys.stderr)
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
