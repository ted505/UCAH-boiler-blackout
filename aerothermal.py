"""UCAH stagnation-point heating — Jonathan Cats.
note the installs you might need below.

this code takes in altitude and velocity csv, gets fresstream properties, calculates post shock proerties, isentropically solves for properties at the wall, and then does radiative equillibrium to get wall temp for which qdot in = qdot out. its then solves for heat flux and saves/plots results

"""
from pathlib import Path
from typing import NamedTuple
import numpy as np
import pandas as pd
import cantera
from scipy.optimize import brentq
from ambiance import Atmosphere
from ruamel.yaml import YAML

# File imports / exports
TRAJECTORY_CSV = Path(__file__).resolve().parent / "glide_result_collocation.csv"
OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "ucah_results"
SHOW_PLOTS = True

# parameters 
effective_lewis_number = 1.4 #catalytic effects variable, simply a reasonable value from Anderson
mass_transfer_exponent = 0.63 # same as above except always constant 
nose_radius_m = 0.001 #nose radius in meters 
wall_emissivity = 0.8 #thermal emmisivity (dimensionless)
stefan_boltzmann_constant_W_m2_K4 = 5.67e-8 #stefan boltzman constant in Watts per meters squared kelvin to the fourth

#Thermodynamic properties stuff. essentially the inputs to cantera
FREESTREAM_SPECIES_MASS_FRACTIONS = {"N2": 0.75523, "O2": 0.23142, "Ar": 0.01288, "CO2": 0.00047}
AIR_SPECIES_NAMES = ["N2", "O2", "N", "O", "NO", "NO2", "N2O", "Ar", "CO2", "CO", "C"]
MIN_SUPPORTED_TEMPERATURE_K = 200.0
MAX_SUPPORTED_TEMPERATURE_K = 5000.0

# just initialization
class WallHeatingResult(NamedTuple):
    temperature_K: float
    heat_flux_W_m2: float
    sensible_driving_enthalpy_J_kg: float
    recombination_enthalpy_J_kg: float


# does what CEA-wrap does if CEA_wrap actually worked :( Basically a bunch of chemistry that we dont need to understand thru line 158
class AirPropertyCalculator:

    def __init__(self):

        def load_species_records(filename):
            paths = [Path(data_directory) / filename for data_directory in cantera.get_data_directories()]
            path = next((path for path in paths if path.is_file()), None)
            if path is None:
                raise ValueError('Required Cantera data file was not found.')
            # Read structured coefficient/parameter records into Python dictionaries.
            # No thermodynamic equations are evaluated by the YAML reader.
            data = YAML(typ='safe').load(path.read_text())
            return {record['name']: record for record in data['species']}
        # NASA species polynomial thermodynamics: variable cp(T), h(T), s(T).
        # NASA SPECIES THERMODYNAMIC POLYNOMIALS (evaluated internally by Cantera).
        # For NASA7 records, with molar properties and the universal gas constant Ru:
        # cp_i^0/Ru = a1 + a2*T + a3*T^2 + a4*T^3 + a5*T^4.
        # h_i^0/(Ru*T) = a1 + a2*T/2 + a3*T^2/3 + a4*T^3/4 + a5*T^4/5 + a6/T.
        # s_i^0/Ru = a1*ln(T) + a2*T + a3*T^2/2 + a4*T^3/3 + a5*T^4/4 + a7.
        # Coefficients and valid temperature intervals come from each species record.
        # Chemical reference/formation energy is included in h; h is not simply cp*T.
        nasa = load_species_records('nasa_gas.yaml')
        # TRANSPORT DATA: molecular parameters used by Cantera to compute viscosity
        # and thermal conductivity. This does not import or integrate GRI reaction kinetics.
        transport_records = load_species_records('gri30.yaml')
        selected = []
        for name in AIR_SPECIES_NAMES:
            record = dict(nasa[name])
            record['transport'] = transport_records[name.upper()]['transport']
            species_definition = cantera.Species.from_dict(record)
            selected.append(species_definition)
        # Ideal-gas mixture: p = rho*Rmix*T; cp is not assumed constant.
        self.cantera_gas = cantera.Solution(thermo='ideal-gas',
            species=selected,
            transport_model='mixture-averaged')
        # DATA DOMAIN INTERSECTION: allowed T is the overlap of the requested
        # 200–5000 K range and the species database range. This is not a flow equation.
        self.minimum_temperature_K = max(MIN_SUPPORTED_TEMPERATURE_K, self.cantera_gas.min_temp)
        self.maximum_temperature_K = min(MAX_SUPPORTED_TEMPERATURE_K, self.cantera_gas.max_temp)
        self.cantera_gas.TPY = (298.15, cantera.one_atm, FREESTREAM_SPECIES_MASS_FRACTIONS)
        self.freestream_mass_fractions = self.cantera_gas.Y.copy()
        
    def frozen_composition_state(self, temperature_K, pressure_Pa, species_mass_fractions):
        # FROZEN CHEMISTRY: evaluate the ideal-gas mixture at specified (T,p,Y).
        # Y is held fixed; temperature-dependent specific heats are still used.
        self.cantera_gas.TPY = (temperature_K, pressure_Pa, species_mass_fractions)
        return self.get_current_properties()

    def calculate_equilibrium_state(self,
        fixed_property_pair,
        target_property_value,
        pressure_Pa,
        initial_mass_fractions=None,
        initial_temperature_K=300.0):
        # CHEMICAL EQUILIBRIUM under elemental conservation.
        # At specified T,p: minimize Gibbs energy G = sum_i n_i*chemical_potential_i.
        # TP holds temperature and pressure; HP holds enthalpy and pressure;
        # SP holds entropy and pressure. HP/SP also determine temperature.
        # These are equilibrium algorithms, not time integration of reaction rates.
        for solver in ('element_potential', 'gibbs', 'vcs'):
            try:
                self.cantera_gas.TPY = (initial_temperature_K,
                    pressure_Pa,
                    self.freestream_mass_fractions if initial_mass_fractions is None else initial_mass_fractions)
                setattr(self.cantera_gas, fixed_property_pair, (target_property_value, pressure_Pa))
                self.cantera_gas.equilibrate(fixed_property_pair,
                    solver=solver,
                    rtol=1e-10,
                    max_steps=2000,
                    max_iter=300)
                return self.get_current_properties()
            except cantera.CanteraError:
                if solver == 'vcs':
                    raise

    def get_current_properties(self):
        cantera_gas = self.cantera_gas
        if not self.minimum_temperature_K - 1e-06 <= cantera_gas.T <= self.maximum_temperature_K + 1e-06:
            raise ValueError(f'Gas temperature {cantera_gas.T:.2f} K is outside the supported range.')
        # Cantera mixture h=sum(Y_i*h_i); frozen sound speed a=sqrt((cp/cv)*Rmix*T).
        # MIXTURE PROPERTY EQUATIONS, evaluated internally by Cantera:
        # Rmix = Ru/Wmix; p = rho*Rmix*T (ideal-gas equation of state).
        # h = sum_i Y_i*h_i(T); cp_frozen = sum_i Y_i*cp_i(T).
        # cv_frozen = cp_frozen - Rmix; gamma_frozen = cp_frozen/cv_frozen.
        # a_frozen^2 = (dp/drho)_(s,Y) = gamma_frozen*Rmix*T (sound-speed relation).
        # An equilibrium composition does not make these derivatives equilibrium-path cp/a.
        return dict(temperature_K=cantera_gas.T,
            pressure_Pa=cantera_gas.P,
            density_kg_m3=cantera_gas.density,
            enthalpy_J_kg=cantera_gas.enthalpy_mass,
            entropy_J_kg_K=cantera_gas.entropy_mass,
            specific_heat_cp_J_kg_K=cantera_gas.cp_mass,
            specific_heat_cv_J_kg_K=cantera_gas.cv_mass,
            sound_speed_m_s=cantera_gas.sound_speed,
            species_mass_fractions=cantera_gas.Y.copy(),
            specific_gas_constant_J_kg_K=cantera.gas_constant / cantera_gas.mean_molecular_weight)

    def calculate_transport_properties(self):
        cantera_gas = self.cantera_gas
        dynamic_viscosity_Pa_s, thermal_conductivity_W_m_K, specific_heat_cp_J_kg_K = (cantera_gas.viscosity,
            cantera_gas.thermal_conductivity,
            cantera_gas.cp_mass)
        # PRANDTL NUMBER: Pr = mu*cp/k = nu/alpha (dimensionless),
        # where nu=mu/rho and alpha=k/(rho*cp). Returns viscosity [Pa s] and Pr.
        return (dynamic_viscosity_Pa_s,
            dynamic_viscosity_Pa_s * specific_heat_cp_J_kg_K / thermal_conductivity_W_m_K)

def calculate_freestream_properties(altitude_m, air_properties):
    # Standard atmosphere: Ambiance gives temperature and pressure at altitude.
    # STANDARD ATMOSPHERE: prescribed temperature layers and hydrostatic
    # balance dp/dH=-rho*g0 with ideal-gas density. Ambiance handles the lookup
    # and its altitude convention; Cantera then evaluates the gas at that T,p.
    standard_atmosphere = Atmosphere(float(altitude_m))
    return air_properties.frozen_composition_state(float(standard_atmosphere.temperature[0]),
        float(standard_atmosphere.pressure[0]),
        air_properties.freestream_mass_fractions)

#simple thermo + mach definition an what not <-- stuff to be conserved
def solve_equilibrium_normal_shock(freestream, freestream_speed_m_s, air_properties):
    freestream_density_kg_m3 = freestream['density_kg_m3']
    freestream_pressure_Pa = freestream['pressure_Pa']
    freestream_enthalpy_J_kg = freestream['enthalpy_J_kg']
    # MACH NUMBER: M_inf = V_inf/a_inf. V_inf is imported air-relative speed.
    freestream_mach_number = freestream_speed_m_s / freestream['sound_speed_m_s']
    # Rankine-Hugoniot mass: j = rho*V (constant across the shock).
    mass_flux_kg_m2_s = freestream_density_kg_m3 * freestream_speed_m_s
    # Rankine-Hugoniot momentum: J = p + rho*V^2.
    momentum_flux_Pa = freestream_pressure_Pa + freestream_density_kg_m3 * freestream_speed_m_s ** 2
    # Steady adiabatic energy: h0 = h + V^2/2, including chemical enthalpy.
    total_enthalpy_J_kg = freestream_enthalpy_J_kg + 0.5 * freestream_speed_m_s ** 2
    cache = {}

    #Cantera does the shock chemistry! so we arent using the 337 constant specific heat correlations 
    def calculate_trial_post_shock_state(downstream_speed_ratio):
        if downstream_speed_ratio not in cache:
            # TRIAL VELOCITY RATIO: r = V_2/V_inf, so V_2 = r*V_inf.
            # Subscript 2 means immediately downstream of the normal shock.
            post_shock_speed_m_s = downstream_speed_ratio * freestream_speed_m_s
            # MOMENTUM JUMP CONDITION, rearranged: p_2 = J - j*V_2.
            post_shock_pressure_Pa = momentum_flux_Pa - mass_flux_kg_m2_s * post_shock_speed_m_s
            # TOTAL-ENTHALPY CONSERVATION, rearranged: h_2 = h0 - V_2^2/2.
            # Cantera's HP equilibrium solve provides rho_2 for this h_2 and p_2.
            post_shock_enthalpy_J_kg = total_enthalpy_J_kg - 0.5 * post_shock_speed_m_s ** 2
            cache[downstream_speed_ratio] = air_properties.calculate_equilibrium_state('HP',
                post_shock_enthalpy_J_kg,
                post_shock_pressure_Pa,
                initial_mass_fractions=freestream['species_mass_fractions'],
                initial_temperature_K=freestream['temperature_K'])
        return cache[downstream_speed_ratio]

    def relative_mass_conservation_residual(downstream_speed_ratio):
        # CONTINUITY ROOT EQUATION: f(r) = rho_2(r)*r*V_inf/j - 1 = 0.
        # The zero residual enforces the remaining mass conservation condition.
        return calculate_trial_post_shock_state(downstream_speed_ratio)['density_kg_m3'] * downstream_speed_ratio * freestream_speed_m_s / mass_flux_kg_m2_s - 1.0
    # Perfect-gas shock velocity ratio seeds the bracket ONLY; final gas has variable cp.
    initial_specific_heat_ratio = freestream['specific_heat_cp_J_kg_K'] / freestream['specific_heat_cv_J_kg_K']
    # CALORICALLY PERFECT-GAS NORMAL-SHOCK VELOCITY RATIO FROM 337/334:
    # r_guess = [(gamma-1)*M_inf^2+2]/[(gamma+1)*M_inf^2].
    # Only the starting bracket uses this approximation, not the final shock state.
    initial_downstream_speed_ratio = ((initial_specific_heat_ratio - 1) * freestream_mach_number ** 2 + 2) / ((initial_specific_heat_ratio + 1) * freestream_mach_number ** 2)
    # NUMERICAL BRACKET HEURISTIC: sample around r_guess, staying below r=1.
    # r=1 is the unwanted no-shock solution. These multipliers are search choices,
    # not additional shock equations.
    lower_speed_ratio, upper_speed_ratio = (max(0.0001,
        0.1 * initial_downstream_speed_ratio),
        0.5 * (1 + initial_downstream_speed_ratio))
    lower_ratio_residual, upper_ratio_residual = (relative_mass_conservation_residual(lower_speed_ratio),
        relative_mass_conservation_residual(upper_speed_ratio))
    if lower_ratio_residual * upper_ratio_residual > 0:
        # FALLBACK ROOT SEARCH: evaluate f(r) on candidate ratios and find a
        # negative-to-positive crossing. This is numerical bracketing, not a gas law.
        candidates = np.unique(np.r_[np.geomspace(0.0001, 0.9, 60), 1 - np.geomspace(1e-07, 0.1, 35)])
        values = [(ratio, relative_mass_conservation_residual(ratio)) for ratio in candidates]
        brackets = [(lower_candidate_ratio,
            upper_candidate_ratio) for (lower_candidate_ratio,
            lower_candidate_residual),
            (upper_candidate_ratio,
            upper_candidate_residual) in zip(values[:-1],
            values[1:]) if lower_candidate_residual <= 0 <= upper_candidate_residual]
        if not brackets:
            raise ValueError('No compressive equilibrium-shock root found. The upstream/no-shock solution is deliberately excluded.')
        lower_speed_ratio, upper_speed_ratio = brackets[0]
    # BRENT ROOT METHOD: solve f(r)=0 inside a sign-changing bracket.
    # Combines interpolation and bisection; tolerances refer to the velocity ratio.
    downstream_speed_ratio = brentq(relative_mass_conservation_residual,
        lower_speed_ratio,
        upper_speed_ratio,
        rtol=1e-10,
        maxiter=150,
        xtol=1e-09)
    post_shock = calculate_trial_post_shock_state(downstream_speed_ratio)
    # COMPRESSION-BRANCH CHECK: a compressive shock has rho_2>rho_inf
    # and p_2>p_inf. This rejects a non-compressive root.
    if post_shock['density_kg_m3'] <= freestream_density_kg_m3 or post_shock['pressure_Pa'] <= freestream_pressure_Pa:
        raise ValueError('Root is not a compression shock.')
    return (post_shock, total_enthalpy_J_kg)

def solve_stagnation_edge_state(post_shock, total_enthalpy_J_kg, air_properties):
    cache = {}

    # Isentropic post-shock deceleration: s_edge=s_postshock; find h_edge=h0.
    # LOG-PRESSURE COORDINATE: ell=ln(p/1 Pa), so p=exp(ell) Pa.
    # This keeps trial pressure positive while the SP solve holds s=s_2.
    def calculate_state_at_log_pressure(log_pressure_Pa):
        if log_pressure_Pa not in cache:
            cache[log_pressure_Pa] = air_properties.calculate_equilibrium_state('SP',
                post_shock['entropy_J_kg_K'],
                np.exp(log_pressure_Pa),
                initial_mass_fractions=post_shock['species_mass_fractions'],
                initial_temperature_K=post_shock['temperature_K'])
        return cache[log_pressure_Pa]
    # RESIDUAL NORMALIZATION: a nonzero enthalpy scale makes f dimensionless.
    # The 1 J/kg floor is numerical scaling, not an extra energy contribution.
    enthalpy_error_scale_J_kg = max(abs(total_enthalpy_J_kg), abs(post_shock['enthalpy_J_kg']), 1.0)

    def residual(log_pressure_Pa):
        # STAGNATION ENERGY ROOT: f(ell) = [h(s_2,p)-h0]/h_scale = 0.
        # At the stagnation point V_edge tends to zero, so h_edge=h0.
        return (calculate_state_at_log_pressure(log_pressure_Pa)['enthalpy_J_kg'] - total_enthalpy_J_kg) / enthalpy_error_scale_J_kg
    lower_bound, upper_bound = (np.log(post_shock['pressure_Pa']), np.log(2 * post_shock['pressure_Pa']))
    for _ in range(12):
        if residual(upper_bound) >= 0:
            break
        # BRACKET EXPANSION: adding ln(2) doubles the trial pressure.
        upper_bound += np.log(2.0)
    # BRENT ROOT METHOD: find the pressure satisfying h_edge=h0 at s_edge=s_2.
    # Only post-shock deceleration is isentropic; the shock itself is irreversible.
    root = brentq(residual, lower_bound, upper_bound, rtol=1e-10, maxiter=150, xtol=1e-09)
    stagnation_edge = calculate_state_at_log_pressure(root)
    air_properties.frozen_composition_state(stagnation_edge['temperature_K'],
        stagnation_edge['pressure_Pa'],
        stagnation_edge['species_mass_fractions'])
    stagnation_edge['dynamic_viscosity_Pa_s'], stagnation_edge['prandtl_number'] = air_properties.calculate_transport_properties()
    return stagnation_edge

def calculate_catalytic_driving_enthalpy(total_enthalpy_J_kg,
    frozen_wall_enthalpy_J_kg,
    equilibrium_wall_enthalpy_J_kg,
    lewis_number=effective_lewis_number):
    # Catalytic film enthalpy: (h0-h_frozen) + Le^0.63*(h_frozen-h_equilibrium).
    # FROZEN-BOUNDARY-LAYER CATALYTIC FILM APPROXIMATION:
    # Delta_h_cat = Delta_h_sensible + Le^n*Delta_h_recombination, n=0.63.
    # Le retains the original effective convention/value; it is not computed here.
    # This is the implemented engineering model, not a surface reaction-rate law.
    return total_enthalpy_J_kg - frozen_wall_enthalpy_J_kg + lewis_number ** mass_transfer_exponent * (frozen_wall_enthalpy_J_kg - equilibrium_wall_enthalpy_J_kg)

def solve_wall_temperatures(stagnation_edge, freestream, total_enthalpy_J_kg, air_properties):
    # STAGNATION PRESSURE DIFFERENCE: Delta_p = p_edge - p_inf [Pa].
    stagnation_pressure_rise_Pa = stagnation_edge['pressure_Pa'] - freestream['pressure_Pa']
    # Spherical-nose gradient estimate: du/ds = sqrt(2*(p_edge-p_inf)/rho_edge)/Rn.
    # PRESSURE-BASED SPHERICAL-NOSE VELOCITY-GRADIENT APPROXIMATION.
    # s is surface distance, Rn is curvature radius; the result has units 1/s.
    # This supplies the gradient for the heat-transfer model, not a resolved flow field.
    stagnation_velocity_gradient_per_s = np.sqrt(2 * stagnation_pressure_rise_Pa / stagnation_edge['density_kg_m3']) / nose_radius_m
    cache = {}

    def calculate_wall_gas_states(wall_temperature_K):
        if wall_temperature_K not in cache:
            # THIN-BOUNDARY-LAYER PRESSURE APPROXIMATION: p_wall=p_edge.
            # Noncatalytic/frozen gas state: Y_wall=Y_edge at the trial wall temperature.
            # These wall properties are for adjacent GAS, not the solid material.
            frozen_composition_state = air_properties.frozen_composition_state(wall_temperature_K,
                stagnation_edge['pressure_Pa'],
                stagnation_edge['species_mass_fractions'])
            frozen_composition_state['dynamic_viscosity_Pa_s'], _ = air_properties.calculate_transport_properties()
            # FULLY CATALYTIC EQUILIBRIUM-WALL LIMIT:
            # Y_wall=Y_equilibrium(T_wall,p_edge) with the same elemental composition.
            equilibrium_wall_state = air_properties.calculate_equilibrium_state('TP',
                wall_temperature_K,
                stagnation_edge['pressure_Pa'],
                initial_mass_fractions=air_properties.freestream_mass_fractions,
                initial_temperature_K=wall_temperature_K)
            equilibrium_wall_state['dynamic_viscosity_Pa_s'], _ = air_properties.calculate_transport_properties()
            cache[wall_temperature_K] = (frozen_composition_state, equilibrium_wall_state)
        return cache[wall_temperature_K]
    
    #Fay-Riddell Stuff, Radiative Equillibrium, Also Catalytics effects stuff
    def calculate_wall_heat_fluxes(wall_temperature_K, is_catalytic_wall):
        frozen_composition_state, equilibrium_wall_state = calculate_wall_gas_states(wall_temperature_K)
        wall_gas_state = equilibrium_wall_state if is_catalytic_wall else frozen_composition_state
        # Fay-Riddell-type prefactor: 0.76*Pr^-0.6*(rho*mu)_e^0.4*(rho*mu)_w^0.1*sqrt(du/ds).
        # LAMINAR STAGNATION HEAT-TRANSFER PREFACTOR, Fay-Riddell-type.
        # Units are kg/(m^2 s); multiplying by J/kg gives W/m^2.
        # The complete implemented model combines this prefactor with the frozen-film
        # enthalpy analogy; it is not an exact classical Fay-Riddell solution.
        heat_transfer_prefactor_kg_m2_s = 0.76 * stagnation_edge['prandtl_number'] ** (-0.6) * (stagnation_edge['density_kg_m3'] * stagnation_edge['dynamic_viscosity_Pa_s']) ** 0.4 * (wall_gas_state['density_kg_m3'] * wall_gas_state['dynamic_viscosity_Pa_s']) ** 0.1 * np.sqrt(stagnation_velocity_gradient_per_s)
        # SENSIBLE ENTHALPY DIFFERENCE: Delta_h_s = h0 - h(T_wall,p_edge,Y_edge).
        sensible_driving_enthalpy_J_kg = total_enthalpy_J_kg - frozen_composition_state['enthalpy_J_kg']
        # EFFECTIVE RECOMBINATION ENTHALPY: Delta_h_rec = h_wall,frozen-h_wall,eq.
        # Same temperature and pressure; only chemical composition differs.
        recombination_enthalpy_J_kg = frozen_composition_state['enthalpy_J_kg'] - equilibrium_wall_state['enthalpy_J_kg']
        # SURFACE-CHEMISTRY CHOICE: catalytic uses Delta_h_s+Le^n*Delta_h_rec;
        # noncatalytic uses only Delta_h_s.
        total_driving_enthalpy_J_kg = calculate_catalytic_driving_enthalpy(total_enthalpy_J_kg,
            frozen_composition_state['enthalpy_J_kg'],
            equilibrium_wall_state['enthalpy_J_kg']) if is_catalytic_wall else sensible_driving_enthalpy_J_kg
        # Enthalpy film heating: q_conv = prefactor * driving enthalpy.
        convective_heat_flux_W_m2 = heat_transfer_prefactor_kg_m2_s * total_driving_enthalpy_J_kg
        # Stefan-Boltzmann radiation: q_rad = epsilon*sigma*Tw^4; no incoming radiation.
        radiative_heat_flux_W_m2 = wall_emissivity * stefan_boltzmann_constant_W_m2_K4 * wall_temperature_K ** 4
        return (convective_heat_flux_W_m2,
            radiative_heat_flux_W_m2,
            sensible_driving_enthalpy_J_kg,
            recombination_enthalpy_J_kg)
    results = []
    for is_catalytic_wall in (True, False):

        def wall_heat_balance_residual(wall_temperature_K):
            convective_heat_flux_W_m2, radiative_heat_flux_W_m2, _, _ = calculate_wall_heat_fluxes(wall_temperature_K,
                is_catalytic_wall)
            # Radiative equilibrium: q_conv(Tw)-q_rad(Tw)=0; no thermal storage or conduction.
            return convective_heat_flux_W_m2 - radiative_heat_flux_W_m2
        minimum_wall_temperature_K, maximum_wall_temperature_K = (air_properties.minimum_temperature_K,
            stagnation_edge['temperature_K'])
        # BRENT ROOT METHOD FOR WALL TEMPERATURE: find q_conv=q_rad.
        # The bracket spans the supported minimum temperature to T_edge.
        # No time derivative is solved: each CSV sample is a separate equilibrium state.
        wall_temperature_K = brentq(wall_heat_balance_residual,
            minimum_wall_temperature_K,
            maximum_wall_temperature_K,
            xtol=1e-05,
            rtol=1e-10,
            maxiter=150)
        convective_heat_flux_W_m2, radiative_heat_flux_W_m2, sensible_driving_enthalpy_J_kg, recombination_enthalpy_J_kg = calculate_wall_heat_fluxes(wall_temperature_K,
            is_catalytic_wall)
        # Reported heat flux is outward radiation, equal to incoming convection
        # within root tolerance. Their difference (net wall heating) is approximately zero.
        results.append(WallHeatingResult(wall_temperature_K,
            radiative_heat_flux_W_m2,
            sensible_driving_enthalpy_J_kg,
            recombination_enthalpy_J_kg))
    return results

#just calls everything
def run_analysis(trajectory):
    air_properties = AirPropertyCalculator()
    records = []
    for time_s, altitude_m, speed_m_s in trajectory[["Time_s",
        "Altitude_m",
        "Speed_mps"]].itertuples(index=False,
        name=None):
        freestream = calculate_freestream_properties(altitude_m, air_properties)
        post_shock, total_enthalpy = solve_equilibrium_normal_shock(freestream, speed_m_s, air_properties)
        edge = solve_stagnation_edge_state(post_shock, total_enthalpy, air_properties)
        catalytic, noncatalytic = solve_wall_temperatures(edge, freestream, total_enthalpy, air_properties)
        records.append({
            "time_s": time_s, "altitude_m": altitude_m, "speed_m_s": speed_m_s,
            # MACH NUMBER: M=V_csv/a_frozen; recorded without changing the CSV speed.
            "mach_number": speed_m_s/freestream["sound_speed_m_s"],
            "wall_temperature_K_fully_catalytic": catalytic.temperature_K,
            "wall_temperature_K_noncatalytic": noncatalytic.temperature_K,
            "heat_flux_W_m2_fully_catalytic": catalytic.heat_flux_W_m2,
            "heat_flux_W_m2_noncatalytic": noncatalytic.heat_flux_W_m2,
        })
    return pd.DataFrame(records)

#CSV's and whatnot
def save_results(results, trajectory, output_directory):
    output_directory.mkdir(parents=True, exist_ok=True)
    # Preserve original CSV columns alongside calculated quantities.
    combined = pd.concat([trajectory.add_prefix("source_"), results], axis=1)
    combined.to_csv(output_directory/"profile.csv", index=False)
    arrays = {name: results[name].to_numpy() for name in results}
    for name in arrays:
        if name.startswith(("wall_temperature_K_", "heat_flux_W_m2_")):
            arrays[name] = arrays[name][:, None]  # Retain original [:, 0] access.
    np.savez_compressed(output_directory/"results.npz", **arrays)

#Plotting
def plot_results(results):
    import matplotlib
    if not SHOW_PLOTS:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for key, label, title, filename in (
        ("wall_temperature_K_", "Wall temperature (K)",
         "Wall Temperature vs. Time (NOT transient)", "wall_temperature.png"),
        ("heat_flux_W_m2_", "Stagnation heat flux (W/m^2)",
         "Stagnation Point Heat Flux vs. Time (NOT transient)", "heat_flux.png"),
    ):
        figure, axes = plt.subplots(figsize=(9, 5.5))
        axes.plot(results["time_s"], results[key + "fully_catalytic"],
                  linewidth=2, label="Fully catalytic equilibrium wall")
        axes.plot(results["time_s"], results[key + "noncatalytic"],
                  "--", linewidth=2, label="Noncatalytic frozen wall")
        axes.set_xlabel("Time (s)")
        axes.set_ylabel(label)
        axes.set_title(title)
        axes.legend()
        axes.grid(True, alpha=0.3)
        figure.tight_layout()
        figure.savefig(OUTPUT_DIRECTORY / filename, dpi=160)
    if SHOW_PLOTS:
        plt.show()
    plt.close("all")


# Read the trajectory, calculate each sample, then save and plot the results.
if __name__ == "__main__":
    trajectory = pd.read_csv(TRAJECTORY_CSV)
    results = run_analysis(trajectory)
    save_results(results, trajectory, OUTPUT_DIRECTORY)
    plot_results(results)
