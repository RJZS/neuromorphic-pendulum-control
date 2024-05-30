using ModelingToolkit, StructArrays, OhMyREPL
using DifferentialEquations: solve
using Plots, ColorSchemes
using IfElse, JLD
using Distributions: Normal
using Statistics: mean
using Peaks

@variables t
D = Differential(t)
Tf = 3.5

phaseControl_switch_time = 2.6 # 2.2 2.5
pulse_duration = 0.05
pulse_decay_rate = log(2)/pulse_duration # Half-life expression.
pulse_height = -0.3 # Size of (inhibitory) pulses.
pulse_freq = 1.1*2*pi

# Mechanical System
rho = 1000; L = 0.36*1.0; r = 0.02;
vol = pi * r^2 * L; m = rho * vol; g = 9.81;
Jperp = (1/12)*m*(L^2 + 3*r^2) + m*(L/2)^2; # Second term is parallel angle theorem.
# J = m * L^2; # Assume point mass at the bottom:
damp = 1.0 # Damping.
ω_n = sqrt( (m*g*(L/2)) / Jperp );
f_n = ω_n / (2*pi) # Resonant peak is slightly different for nonlinear pendulum.

K = 1

# Nondimensionalised System
K_ND = K / (m*g*L/2)
α = (damp*L/2)*(1/sqrt(Jperp*m*g*L/2))
t_factor = sqrt(m*g*(L/2)/Jperp) # ̃t = t_factor * t
println("K_ND = $K_ND")
println("α = $α")
println("t_factor = $t_factor")

# Init_motor_config = 0 ?
# If the config_switch_time is 155, and init_angle = 4, init_d_angle = 5/t_factor
# I end up with small osc.
# If config_switch_time = 165, I end up with large osc.
# So -> phase control.

# Init_motor_config = 1 ?
# This is always guaranteed I think.

init_angle =  0
init_d_angle = 0

input_fn1(t) = 0.3 < t < 0.6 ? -1.2 : -1
input_fn2(t) = -1
input_fn3(t) = 0.8 < t < 1.1 ? -1.2 : -1
input_fn4(t) = -1


@register_symbolic input_fn1(t)
@register_symbolic input_fn2(t)
@register_symbolic input_fn3(t)
@register_symbolic input_fn4(t)

function sigmoid(u, s, d, imin, height)
    imin + height / ( 1 + exp(-(u-d)*s) )
end
# Ensure a2 and a3 always scale by the same proportion.
function sigmoid_a2(u)
    sigmoid(u, 1, 1.75, 1.5, 0.5)
end
function sigmoid_a3(u)
    sigmoid(u, 1, 1.25, 1.125, 0.5)
end
function sigmoid_a4(u)
    sigmoid(u, 2, 2.25, 1.5, 6.) # NB the max is quite high atm.
end

function Syn(vs, gain)
    gain / ( 1 + exp(-2*(vs+1)) )
end

rate_idx = 2 # For the rate code. 2, 3 or 4 (not using 1).
a4s = [1.57 1.73 2.49 3.56] # Found using 'hco_pendulum.jl'.
k3s = [0.6 0.7 0.9 1.2]

a2 = 0.8*2; # a3 = k3s[rate_idx]*1.5;  
a4 = a4s[rate_idx]

@variables v(t) vs(t) vus(t) v1s(t) v2s(t) v3s(t) v4s(t) input(t) gsyn(t)
@variables pulse(t) rectified_pulse(t) controlled_pulse(t)
@variables a3(t)
@parameters a1=2. τm=0.001 τs=0.05 τus=2.5 a_fback=0
@named first_neur = ODESystem([
    input ~ input_fn1(t),
    τm * D(v) ~ -v + a1*tanh(v) - a2*tanh(vs) + 
        a3*tanh(vs+0.9) - a4*tanh(vus+0.9) +
        Syn(v2s, -0.2) + input + controlled_pulse,
    τs * D(vs)  ~ (v - vs),
    τus * D(vus)  ~ (v - vus),
    a3 ~ 1.05, #IfElse.ifelse(t > 3, 1.2*1.5, 0.7*1.5),
    rectified_pulse ~ IfElse.ifelse(pulse < -0.5, pulse_height, 0),
    controlled_pulse ~ IfElse.ifelse(t > phaseControl_switch_time, rectified_pulse, 0),
])

@named second_neur = ODESystem([
    input ~ input_fn2(t),
    τm * D(v) ~ -v + a1*tanh(v) - a2*tanh(vs) + 
        a3*tanh(vs+0.9) - a4*tanh(vus+0.9) +
        Syn(v1s, -0.2) + input + controlled_pulse,
    τs * D(vs)  ~ (v - vs),
    τus * D(vus)  ~ (v - vus),
    a3 ~ 1.05, #IfElse.ifelse(t > 3, 1.2*1.5, 0.7*1.5),
    rectified_pulse ~ IfElse.ifelse(pulse > 0.5, pulse_height, 0),
    controlled_pulse ~ IfElse.ifelse(t > phaseControl_switch_time, rectified_pulse, 0),
])

## Compare with the same HCO, but without phase control.
@named first_neur_uncontrolled = ODESystem([
    input ~ input_fn1(t),
    τm * D(v) ~ -v + a1*tanh(v) - a2*tanh(vs) + 
        a3*tanh(vs+0.9) - a4*tanh(vus+0.9) +
        Syn(v2s, -0.2) + input,
    τs * D(vs)  ~ (v - vs),
    τus * D(vus)  ~ (v - vus),
    a3 ~ 1.05, #IfElse.ifelse(t > 3, 1.2*1.5, 0.7*1.5),
])

@named second_neur_uncontrolled = ODESystem([
    input ~ input_fn2(t),
    τm * D(v) ~ -v + a1*tanh(v) - a2*tanh(vs) + 
        a3*tanh(vs+0.9) - a4*tanh(vus+0.9) +
        Syn(v1s, -0.2) + input,
    τs * D(vs)  ~ (v - vs),
    τus * D(vus)  ~ (v - vus),
    a3 ~ 1.05, #IfElse.ifelse(t > 3, 1.2*1.5, 0.7*1.5),
])


## Mechanical system
@variables angle(t) d_angle(t) torque(t) direction(t) v1(t) v3(t) disturbance(t)
@variables torque_one(t) torque_two(t) direction_two(t)
@variables motor_config(t)

# Switch on inhibitory pulses.
@parameters tswitch=phaseControl_switch_time
switchOnPhase = (t == tswitch) => 
    [pulse ~ -1] # Inhibit the first neuron first.

# Use a fixed period to determine the time to the next pulse.
# Use the exponential decay of `pulse` to determine the time since the last pulse.
pulse_period = 2*pi/pulse_freq
pulse_val_when_switchover = exp(-pulse_decay_rate*pulse_period/2)

root_eqs = [abs(pulse) ~ pulse_val_when_switchover]
affect = [pulse ~ -sign(pulse)] 
continuous_events =  # equation => affect
    [root_eqs => affect]


full_sys = compose(ODESystem([
        torque ~ IfElse.ifelse(v1 > -0.5, K, 0),
        v1 ~ first_neur.v,

        D(pulse) ~ -pulse_decay_rate*pulse,

        first_neur.v2s ~ second_neur.vs,
        second_neur.v1s ~ first_neur.vs,
        first_neur_uncontrolled.v2s ~ second_neur_uncontrolled.vs,
        second_neur_uncontrolled.v1s ~ first_neur_uncontrolled.vs,

        first_neur.pulse ~ pulse,
        second_neur.pulse ~ pulse,
    ], t, [], [tswitch]; continuous_events, name=:full_sys, discrete_events=[switchOnPhase]),  
        [first_neur, second_neur, first_neur_uncontrolled, second_neur_uncontrolled])

sys = structural_simplify(full_sys)

# Initialise neurons.
init_sys = [first_neur.v => 0, first_neur.vs => 0, 
            first_neur.vus => -1,
            second_neur.v => 0, second_neur.vs => 0,
            second_neur.vus => -0.5,
            first_neur_uncontrolled.v => 0, first_neur_uncontrolled.vs => 0, 
            first_neur_uncontrolled.vus => -1,
            second_neur_uncontrolled.v => 0, second_neur_uncontrolled.vs => 0,
            second_neur_uncontrolled.vus => -0.5,
            pulse => 0,
            ]

prob = ODEProblem(sys, init_sys, (0.0,Tf))
sol = solve(prob,tstops=[phaseControl_switch_time],abstol=1e-11, reltol=1e-9,maxiters=2e6,saveat=5e-3);

## Now, compute the phase of the burst, and the phase delay.
# First, find the times of the bursts.
iz = findfirst(x -> x > 1.5, sol.t) # Remove the initial burst (transient).
signal = sol[first_neur.v][iz:end];
signal_uc = sol[first_neur_uncontrolled.v][iz:end]; # Uncontrolled bursts.

pks, pk_values = findmaxima(signal);
pks_uc, pk_values_uc = findmaxima(signal_uc);
_, pk_proms = peakproms(pks, signal);
_, pk_proms_uc = peakproms(pks_uc, signal_uc);
peak_idxs = pk_proms .> 2 # Select only the spikes, remove the undesired peaks.
peak_idxs_uc = pk_proms_uc .> 2
pks = pks[peak_idxs]; pk_values = pk_values[peak_idxs];
pks_uc = pks_uc[peak_idxs_uc]; pk_values_uc = pk_values_uc[peak_idxs_uc];

pk_times = sol.t[iz:end][pks]; pk_times_uc = sol.t[iz:end][pks_uc]; 
t_start = pk_times[1]; t_burst = pk_times[3] # Assuming two spikes in the first burst.
t_burst_uc = pk_times_uc[3];

println("The first burst starts at $t_start s")
println("The next burst starts at $t_burst s")
pPeaks = plot(sol.t[iz:end], sol[first_neur.v][iz:end])
plot!(pk_times, pk_values, seriestype=:scatter)
pPeaks_uc = plot(sol.t[iz:end], sol[first_neur_uncontrolled.v][iz:end])
plot!(pk_times_uc, pk_values_uc, seriestype=:scatter)

T = t_burst_uc - t_start # Nominal period.
phase_of_pulse = 2*pi*(phaseControl_switch_time - t_start)/T
phase_delay = 2*pi*(t_burst - t_burst_uc)/T
println("Phase of pulse: $phase_of_pulse")
println("Phase delay: $phase_delay")

# iw = findfirst(x -> x > 0.1, sol.t); iend = findfirst(x -> x > 0.2, sol.t)
# p1win = plot(sol[iw:iend], idxs=[torque],legend=:topleft, dpi=300, size=(1200,800),label="Torque")

# i = iw;
# p1z = plot(sol[i:end], idxs=[torque],legend=:topleft,label="Torque", dpi=300, size=(1200,800))
# plot!(twinx(), sol[i:end], idxs=[first_neur.v],color=:firebrick2, xaxis="",legend=:topright,linewidth=2,label="Angle",
#         ylims=(-0.55,0.55),yticks=[-0.5,-0.25,0,0.25,0.5])

p1 = plot(sol, idxs=[first_neur.v],label=false)
plot!(sol, idxs=[second_neur.v],label=false,linestyle=:dash)
plot!(sol, idxs=[first_neur.controlled_pulse, second_neur.controlled_pulse], 
            linewidth=2, label=false)
vline!([phaseControl_switch_time], linestyle=:dash, color=:black, label=false)
# p2 = plot(sol, idxs=[torque],label=false)
# pT = plot(p1, p2, layout=(2,1))

ii = findfirst(x -> x > 2.5, sol.t)
pCompare = plot(sol.t[ii:end]*t_factor/(2*pi), sol[first_neur.v][ii:end],dpi=300, size=(1200,440),
xtickfontsize=fontsize_x,ytickfontsize=fontsize_y,
xguidefontsize=fontsize_guide_large,legendfontsize=fontsize_legend_large+6,linewidth=2,
margin=8Plots.mm, xlabel="t / 2π",
yguidefontsize=fontsize_guide_large,ylabel="Voltage | Current",
label=L"$v_A$",legend=:topleft)
plot!(sol.t[ii:end]*t_factor/(2*pi), sol[first_neur_uncontrolled.v][ii:end],linestyle=:dash,linecolor=1,
    linewidth=2,label=false)#label=L"$v_A$ (no input pulse)")
plot!(sol.t[ii:end]*t_factor/(2*pi), sol[first_neur.controlled_pulse][ii:end], 
            linewidth=2,label="i,pulse")#, label="Input pulse current")
vline!([phaseControl_switch_time*t_factor/(2*pi)], linestyle=:dash, color=:black, label=false)


t = sol.t;  v1 = sol[first_neur.v];  v2 = sol[second_neur.v]
torque = sol[torque]
save_string = "CDC_paper/figs/6_phase_response.jld"
# @save save_string t v1 v2 torque
        
# savefig(p1, "CDC_paper/figs/$figure_title")

## Data
pulse_phases = [1.11 2.92 3.52 4.72 5.32 5.92]'
phase_delays_longpulse = [-1.62 -0.511 -0.150 0.752 1.32 1.95]'
phase_delays_shortpulse = [-0.42 -0.27 -0.15 0.03 0.27 0.63]';

pPRC = plot(pulse_phases, phase_delays_shortpulse,
    xlabel="Phase at which the Pulse Starts [rad]", ylabel="Phase Delay [rad]",
    label=false,
    markershape=:circle,
    #dpi=300, size=(1200,800),
    yticks=pitick(-pi/6, pi/6, 12),
    xticks=pitick(pi/2, 3*pi/2, 2),ylims=(-pi/6,pi/4),
    xtickfontsize=fontsize_x,ytickfontsize=fontsize_y,
xguidefontsize=fontsize_guide+4,margin=5Plots.mm,
yguidefontsize=fontsize_guide_large,
legendfontsize=fontsize_legend_large)

fig_phase_response = plot(pCompare, pPRC, layout=(1,2))
savefig(fig_phase_response, "CDC_paper/figs/6-phasebehaviour")
