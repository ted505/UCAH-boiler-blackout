%{
UCAH 2026 Temperature Profile First Order Analysis
By: Jonathan Cats

Basic Concept Follows this Paper Fay-Riddell stagnation heat flux theory.
Ctrl + F "Fay" in Andersons "Hypersonic and High Heat Gas Dynamics"
Textbook in Onedrive
REQUIRES AEROSPACE TOOLBOX TO RUN

Potential Issues: 
-No Consideration of AoA or C_d and whatnot
-Here, we assume a linear mach profile from 8 - 3 which is obviously not
true
-We assume constant specific heats which is bad (can use CEA_wrap to obtain
varaible gamma, but p_e, rho_e, T_e are all derived assumming constant
specific heat

Things to add/do:
ALL OF THIS IS IN methods.pdf in the teams
-Compare these methods to stagnation point heating in methods.pdf
-do lumped heating at solid nosetip
-Ablative calculations if deemed neccesasy
-Windward-surface heating analysis & to wind leading edges. Decipher which
stuff is applicable
-Fay-Riddell actually has a chemistry term to look into

%}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Parameters

altitude = linspace(30480,1, 1000);

    %Fay-Riddell Stuff
pr = 0.71; %Prandtl Number 
gamma = 1.4; %specific heat 
R = 287; %specific gas constant in J / (kg * K)
sigma = 5.67E-8; %Stefan-Boltzmann Constant in W/(m^2 * K^4)
mu_ref = 1.716E-5; %Viscosity reference in Pa*s
T_ref = 273.15; %Room temp in Kelvin
S = 110.4; %Sutherland Constant in Kelvin
c_p = (gamma / (gamma - 1)) * R; %Specific Heat at Constant Pressure (so assumed constant right now)
M_star = linspace(8,3,1000); %linear mach vector (change this to something realistic later for when we care more about transient effects

    %Velocity-Gradient Stuff
Rn = .001; %nose radius in m
epsilon = 0.8; %wall emmisivity of material 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Fay-Riddell

for i = 1:length(altitude)

    %Atmopheric Conditions 
    [T_inf_star(i), a_star(i), P_inf_star(i), rho_inf_star(i)] = atmoscoesa(altitude(i)); 

    %Mach 
    u_star(i) =  M_star(i) * a_star(i);
    
    %Straight out of Andersons Modern Compressible Flow (In One Drive Lit
    %Review) also 337 slides
    p_e(i) = P_inf_star(i) * ((2 * gamma * M_star(i)^2 - (gamma - 1)) / (gamma + 1));  %pressure at BL EDGE
    
    %Straight from 337 normal shocks lecture notes (the bow shock is normal
    %at the stagnation point)
    rho_e(i) = rho_inf_star(i) * (((gamma + 1) * M_star(i)^2) / ((gamma - 1) * M_star(i)^2 + 2)); %gas density at BL EDGE in kg/m^3
    T_e(i) = T_inf_star(i) * ((1 + 0.5*(gamma - 1)*M_star(i)^2) * ((2*gamma/(gamma - 1))*M_star(i)^2 - 1)) / (((gamma + 1)^2/(2*(gamma - 1))) * M_star(i)^2);; %Temperature at the BL EDGE in K
    
    %Straight up first law thermo
    h_aw(i) = c_p * T_inf_star(i) + (u_star(i)^2) / 2; %enthalpy at the wall (adiabatic) in J/kg
    
    %Sutherlands Law from 33300
    mu_e(i) = mu_ref * (T_e(i) / T_ref)^(3/2) * (T_ref + S) / (T_e(i) + S); %dynamic viscosity at BL EDGE in Pa/s
    
    %In texbook at top (Fay-Riddell Theory)
    due_dx(i) = (1 / Rn) * sqrt(2 * (p_e(i) - P_inf_star(i)) / rho_e(i)); %velocity gradient at stagnation point
   
    %Fay-Riddell Function. It is simply q_dot_fay_riddell (heat flux in) -
    %q_dot_stefan_boltzman (heat flux out). It has Tw as a variable
    
    FR = @(Tw) 0.76 * pr^(-0.6) * (rho_e(i) * mu_e(i))^0.4 * ((p_e(i) / (R * Tw)) * (mu_ref * (Tw / T_ref)^(3/2) * (T_ref + S) / (Tw + S)))^0.1 * sqrt(due_dx(i)) * (h_aw(i) - c_p * Tw) - epsilon * sigma * Tw^4;

    % finds Tw such that FR = 0. We bound fzero to search with values of
    % Tw from 1 K to h_aw(i) / c_p (the adiabatic case)
    Tw(i) = fzero(FR, [1, h_aw(i) / c_p]);

    %calcualte heat flux
    q_stag(i) = epsilon * sigma * Tw(i)^4;

end

%Outputs
fprintf("Stagnation Heat Flux %d W/m^2\n", max(q_stag))
fprintf("Max Temp: %d K\n", max(Tw))

%plots

figure(1)
plot(altitude, q_stag)
xlabel("Altitude in Meters (m)");
ylabel("Stagantion Heat Flux in W/m^2");
title("Stagnation Heat Flux vs. Altitude")
grid()

figure(2)
plot(altitude, Tw)
xlabel("Altitude in Meters (m)");
ylabel("Wall Temperature in Kelvin (k)");
title("Wall Temperature vs. Altitude")
grid()