using ModelingToolkit, StructArrays, OhMyREPL
using DifferentialEquations: solve
using Plots, ColorSchemes
using IfElse, JLD
using Distributions: Normal
using Statistics: mean
using Peaks

@variables t
D = Differential(t)
Tf = 36.5

q_event = -1 # At what angle to trigger sensory feedback.
pulse_height = -0.3 # Size of (inhibitory) pulses.

config_switch_time = Tf+1 # When to switch the config, and switch on the controller.
phaseControl_switch_time = 0.1

# Mechanical System
rho = 1000; L = 0.36*1.0; r = 0.02;
vol = pi * r^2 * L; m = rho * vol; g = 9.81;
Jperp = (1/12)*m*(L^2 + 3*r^2) + m*(L/2)^2; # Second term is parallel angle theorem.
# J = m * L^2; # Assume point mass at the bottom:
damp = 0.1 # Damping.
ω_n = sqrt( (m*g*(L/2)) / Jperp );
f_n = ω_n / (2*pi) # Resonant peak is slightly different for nonlinear pendulum.

K = 1.0

# Nondimensionalised System
K_ND = K / (m*g*L/2)
α = (damp*L/2)*(1/sqrt(Jperp*m*g*L/2))
t_factor = sqrt(m*g*(L/2)/Jperp) # ̃t = t_factor * t
println("K_ND = $K_ND")
println("α = $α")
println("t_factor = $t_factor")

# Init_motor_config = 0 ?
# If the config_switch_time is 8.2
# I end up with small osc.
# If config_switch_time = 8.5, I end up with large osc.
# So -> phase control.
# (Seems to be approx every xxx seconds that it changes round.)

# Init_motor_config = 1 ?
# Always get small osc for the current settings.

experiment_idx = 1 # Change this when switching experiments.
experiment_titles = ["switching_phasecontrol-short_pulses" "switching_phasecontrol-long_pulses"]
experiment_title = experiment_titles[experiment_idx]


pulse_durations = [0.05 0.3] # For 1.1 Hz or 1.5 Hz, respectively.
pulse_duration = pulse_durations[experiment_idx]
pulse_decay_rate = log(2)/pulse_duration # Half-life expression.

init_motor_config = 1 # 1 for unidirectional, 0 for bidirectional
init_gsyn = 0.04

init_angle =  1
init_d_angle = 0.0

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
a4s = [1.57 1.75 2.49 3.56] # Found using 'hco_pendulum.jl'.
k3s = [0.6 0.7 0.9 1.2]

a2 = 0.8*2;  a3 = k3s[rate_idx]*1.5;  a4 = a4s[rate_idx]

@variables v(t) vs(t) vus(t) v1s(t) v2s(t) v3s(t) v4s(t) input(t) gsyn(t)
@variables pulse(t) rectified_pulse(t) controlled_pulse(t)
@parameters a1=2. τm=0.001 τs=0.05 τus=2.5 a_fback=0
@named first_neur = ODESystem([
    input ~ input_fn1(t),
    τm * D(v) ~ -v + a1*tanh(v) - a2*tanh(vs) + 
        a3*tanh(vs+0.9) - a4*tanh(vus+0.9) +
        Syn(v2s, -0.2) + input + Syn(v3s, gsyn) + controlled_pulse,
    τs * D(vs)  ~ (v - vs),
    τus * D(vus)  ~ (v - vus),
    rectified_pulse ~ IfElse.ifelse(pulse < -0.5, pulse_height, 0),
    controlled_pulse ~ IfElse.ifelse(t > phaseControl_switch_time, rectified_pulse, 0),
])

@named second_neur = ODESystem([
    input ~ input_fn2(t),
    τm * D(v) ~ -v + a1*tanh(v) - a2*tanh(vs) + 
        a3*tanh(vs+0.9) - a4*tanh(vus+0.9) +
        Syn(v1s, -0.2) + input + Syn(v4s, gsyn) + controlled_pulse,
    τs * D(vs)  ~ (v - vs),
    τus * D(vus)  ~ (v - vus),
    rectified_pulse ~ IfElse.ifelse(pulse > 0.5, pulse_height, 0),
    controlled_pulse ~ IfElse.ifelse(t > phaseControl_switch_time, rectified_pulse, 0),
])

@named third_neur = ODESystem([
    input ~ input_fn3(t),
    τm * D(v) ~ -v + a1*tanh(v) - a2*tanh(vs) + 
        a3*tanh(vs+0.9) - a4*tanh(vus+0.9) +
        Syn(v4s, -0.2) + input + Syn(v1s, gsyn) + controlled_pulse,
    τs * D(vs)  ~ (v - vs),
    τus * D(vus)  ~ (v - vus),
    rectified_pulse ~ IfElse.ifelse(pulse < -0.5, pulse_height, 0),
    controlled_pulse ~ IfElse.ifelse(t > phaseControl_switch_time, rectified_pulse, 0),
])

@named fourth_neur = ODESystem([
    input ~ input_fn4(t),
    τm * D(v) ~ -v + a1*tanh(v) - a2*tanh(vs) + 
        a3*tanh(vs+0.9) - a4*tanh(vus+0.9) +
        Syn(v3s, -0.2) + input + Syn(v2s, gsyn) + controlled_pulse,
    τs * D(vs)  ~ (v - vs),
    τus * D(vus)  ~ (v - vus),
    rectified_pulse ~ IfElse.ifelse(pulse > 0.5, pulse_height, 0),
    controlled_pulse ~ IfElse.ifelse(t > phaseControl_switch_time, rectified_pulse, 0),
])


## Mechanical system
@variables angle(t) d_angle(t) torque(t) direction(t) v1(t) v3(t) disturbance(t)
@variables torque_one(t) torque_two(t) direction_two(t)
@variables motor_config(t)

# Switch on sensory feedback
@parameters tswitch=config_switch_time
switchConfig = (t == tswitch) => 
    [motor_config ~ 1-motor_config
    gsyn ~ -gsyn]


# Large oscillations: (based on 'unidirectional_phasecontrol.jl')
root_eqs = [atan(sin(angle), cos(angle)) ~ q_event] # Atan2 function
eps = 0.02
# To get large osc switchover every other oscillation:
# affect = [pulse ~ IfElse.ifelse(asin(sin((angle-pi)/2)) > q_event+eps, pulse, IfElse.ifelse(d_angle < 0, pulse, -sign(pulse)))] 
# To get large osc switchover every one oscillation (second one has no 'dq > 0' rule):
# affect = [pulse ~ IfElse.ifelse(abs(asin(sin((angle-pi)/2))) < abs(q_event)-eps, pulse, IfElse.ifelse(d_angle < 0, pulse, -sign(pulse)))] 
affect = [pulse ~ IfElse.ifelse(abs(asin(sin((angle-pi)/2))) < abs(q_event)-eps, pulse, -sign(pulse))] 
# (note this asin(sin()) logic also avoids the zero crossing from the discontinuity in atan2)

continuous_events =  # equation => affect
    [root_eqs => affect]


@named pendulum = ODESystem([
    Jperp * D(D(angle)) ~ torque - damp*(L/2)*D(angle) - m*g*(L/2)*sin(angle),
    # D(D(angle)) ~ -α*D(angle) - sin(angle) + torque,
    
    torque ~ torque_one + torque_two,

    torque_one ~ IfElse.ifelse(v1 > -0.5, K, 0),

    direction_two ~ IfElse.ifelse(motor_config > 0.5, 1, -1),
    torque_two ~ IfElse.ifelse(v3 > -0.5, direction_two*K, 0),
    
    disturbance ~ 0,
    d_angle ~ D(angle),  # Bug if I use D(angle) in event handling.

    D(pulse) ~ -pulse_decay_rate*pulse,
], t; continuous_events)

full_sys = compose(ODESystem([
        pendulum.v1 ~ first_neur.v,
        pendulum.v3 ~ third_neur.v,

        # Connections within the HCOs
        first_neur.v2s ~ second_neur.vs,
        second_neur.v1s ~ first_neur.vs,
        third_neur.v4s ~ fourth_neur.vs,
        fourth_neur.v3s ~ third_neur.vs,

        # Connections between the HCOs
        first_neur.v3s ~ third_neur.vs,
        third_neur.v1s ~ first_neur.vs,
        second_neur.v4s ~ fourth_neur.vs,
        fourth_neur.v2s ~ second_neur.vs,

        pendulum.motor_config ~ motor_config,
        D(motor_config) ~ 0,
        first_neur.gsyn ~ gsyn,
        second_neur.gsyn ~ gsyn,
        third_neur.gsyn ~ gsyn,
        fourth_neur.gsyn ~ gsyn,
        D(gsyn) ~ 0,

        first_neur.pulse ~ pendulum.pulse,
        second_neur.pulse ~ pendulum.pulse,
        third_neur.pulse ~ pendulum.pulse,
        fourth_neur.pulse ~ pendulum.pulse,
    ], t, [], [tswitch]; name=:full_sys, discrete_events=[switchConfig]), 
        [first_neur, second_neur, third_neur, fourth_neur, pendulum])

sys = structural_simplify(full_sys)

# Initialise neurons.
init_sys = [first_neur.v => 0, first_neur.vs => 0, first_neur.vus => 0,
            second_neur.v => 0, second_neur.vs => 0, second_neur.vus => 0,
            third_neur.v => 0, third_neur.vs => 0, third_neur.vus => 0,
            fourth_neur.v => 0, fourth_neur.vs => 0, fourth_neur.vus => 0,
            pendulum.angle => init_angle,
            pendulum.d_angle => init_d_angle,
            motor_config => init_motor_config,
            gsyn => init_gsyn,
            pendulum.pulse => 1,
            ]

prob = ODEProblem(sys, init_sys, (0.0,Tf))
sol = solve(prob,tstops=[config_switch_time],abstol=1e-11, reltol=1e-9,maxiters=2e6,saveat=2e-4);

KE = 0.5 * Jperp * sol[pendulum.d_angle].^2;
GPE = m * 9.81 * (L/2) * ( 1 .- cos.(sol[pendulum.angle]) ) #( cos.(pi.-sol[angle]).-1 )
energy = GPE .+ KE

# Compute steady-state amplitude and frequency.
i = Int(length(sol.t)-12000)
ampfinal = mod(maximum(sol[pendulum.angle][i:end]), 2*pi)
println("Amp: $ampfinal")
pks, _ = findmaxima(sol[pendulum.angle][i:end])
pk_times = sol.t[i:end][pks]
diffs = diff(pk_times); freqfinal = 1 ./ diffs
println("Latest freqs: $freqfinal")
f = freqfinal / (2*pi)

# Compute steady-state freq for large oscillations.
signal = 2*asin.(sin.(sol[pendulum.angle][i:end]/2))
n = 2 # Number of swings per period.
pks_large, pk_values_large = findmaxima(abs.(signal)) # 'abs' because it's sin(q/2)
pks_large = pks_large[1:n:end]; pk_values_large = pk_values_large[1:n:end]
pk_times_large = sol.t[i:end][pks_large]
diffs_large = diff(pk_times_large)
freqfinal_large = 1 ./ diffs_large
freqfinal_large_rounded = round.(freqfinal_large, digits=3)
println("Large osc freqs: $freqfinal_large_rounded")
# plot(sol.t[i:end], signal)
# plot!(pk_times_large, pk_values_large, seriestype=:scatter)
# plot!(sol.t[i:end], sol[pendulum.torque,i:end])

p1 = plot(sol, idxs=[pendulum.torque,pendulum.angle],legend=:topleft, dpi=300, size=(1200,800),label=["Torque" "Angle"])

iw = findfirst(x -> x > 7, sol.t); iend = findfirst(x -> x > 20, sol.t)
# p1win = plot(sol[iw:iend], idxs=[pendulum.torque,pendulum.angle,
#     10*first_neur.controlled_pulse, 10*second_neur.controlled_pulse],
#     legend=:topleft, dpi=300, size=(1200,800),linewidth=2,
#     label=["Torque" "Angle" "Iapp,1 (scaled)" "Iapp,2 (scaled)"])

# plot(sol[iw:iend], idxs=[pendulum.angle], yticks=[74*pi, 76*pi,78*pi,80*pi,82*pi])
# plot!(twinx(), sol[iw:iend], idxs=[pendulum.torque])

p1win = plot(sol[iw:iend], idxs=[
    # pendulum.angle,
    sin(pendulum.angle), 
    pendulum.torque, 2*first_neur.controlled_pulse,2*second_neur.controlled_pulse,
    #abs(asin(sin((pendulum.angle-pi)/2))),
    #atan(sin(pendulum.angle), cos(pendulum.angle)),
    ],label=false)
    #hline!([-q_event], label=false)

# ptest = plot(sol[iw:iend], idxs=[pendulum.torque,
#     5*first_neur.controlled_pulse,5*second_neur.controlled_pulse, 
#     atan(sin(pendulum.angle), cos(pendulum.angle)),
#     asin(sin((pendulum.angle-pi)/2))
#     ],
#     label=false,size=(1200,800),linewidth=2)
# hline!([q_event], label=false)

# Locate when the bursts start.
dpulses = diff(sol[pendulum.torque,iw:iend])
pks, vals = findmaxima(abs.(dpulses))
tc = sol.t[iw:iend]
#plot(tc, sol[pendulum.torque,iw:iend]) # To test.
#plot!(tc[pks], vals, seriestype=:scatter)
#sol[pendulum.angle,iw:end][pks[1]]

# iE = findfirst(x -> x > 200, sol.t)
# pE = plot(sol[i:end], idxs=[pendulum.d_angle*pendulum.torque,pendulum.torque],label=["ωτ (power transferred)" "Torque"], dpi=300, size=(1200,800))

iz = Int(length(sol.t)-5000)
# p1z = plot(sol[iz:end], idxs=[pendulum.torque,
#     pendulum.pulse,first_neur.controlled_pulse],legend=:topleft,label="Torque",dpi=300, size=(1200,800))
# plot!(twinx(), sol[iz:end], idxs=[pendulum.angle],color=:firebrick2, xaxis="",legend=:topright,linewidth=2,label="Angle",
#         ylims=(-0.55,0.55),yticks=[-0.5,-0.25,0,0.25,0.5])

p1z_wrapped = plot(sol[iz:end], idxs=[
    # pendulum.angle,
    sin(pendulum.angle), 
    pendulum.torque, 2*first_neur.controlled_pulse,2*second_neur.controlled_pulse,
    #abs(asin(sin((pendulum.angle-pi)/2))),
    #atan(sin(pendulum.angle), cos(pendulum.angle)),
    ],label=false)
    #hline!([-q_event], label=false)

# For arrows indicating zero-crossing timings:
# quiver(x,y,quiver=(u,v))

t = sol.t;  torque = sol[pendulum.torque];  angle = sol[pendulum.angle]
d_angle = sol[pendulum.d_angle]; pulse_one = sol[first_neur.controlled_pulse]
pulse_two = sol[second_neur.controlled_pulse]
save_string = "CDC_paper/figs/$experiment_title.jld"
@save save_string t torque angle d_angle pulse_one pulse_two

pi_range = -pi:0.1:pi
p_sensor = plot(pi_range, i -> abs(asin(sin((i-pi)/2)))) # For switch logic
plot!(pi_range, i -> atan(sin(i), cos(i))) # Event

# p1

# p1z_wrapped

p1z_wrapped
