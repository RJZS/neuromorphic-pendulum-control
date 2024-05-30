using ModelingToolkit, StructArrays, OhMyREPL
using DifferentialEquations: solve
using Plots, ColorSchemes
using IfElse, JLD
using Distributions: Normal
using Statistics: mean
using Peaks

@variables t
D = Differential(t)
Tf = 40 # First submission: 34.5

experiment_idx = 2 # 1 for large osc, 2 for small osc.

# Mechanical System
rho = 1000; L = 0.36*1.0; r = 0.02;
vol = pi * r^2 * L; m = rho * vol; g = 9.81;
Jperp = (1/12)*m*(L^2 + 3*r^2) + m*(L/2)^2; # Second term is parallel angle theorem.
# J = m * L^2; # Assume point mass at the bottom:
damp = 0.1 # Damping.
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

init_motor_config = 1 # 1 for unidirectional, 0 for bidirectional
config_switch_time = Tf+1 # When to switch the config

# Init_motor_config = 0 ?
# If the config_switch_time is 155, and init_angle = 4, init_d_angle = 5/t_factor
# I end up with small osc.
# If config_switch_time = 165, I end up with large osc.
# So -> phase control.

# Init_motor_config = 1 ?
# This is always guaranteed I think.

init_angles =  [0 1]
init_angle = init_angles[experiment_idx]
init_d_angle = 0

input_fn1(t) = 0.3 < t < 0.6 ? -1.2 : -1
input_fn2(t) = -1
input_fn3(t) = 0.2 < t < 0.5 ? -1.2 : -1
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
gsyn_val = 0.04

@variables v(t) vs(t) vus(t) v1s(t) v2s(t) v3s(t) v4s(t) input(t) gsyn(t)
@parameters a1=2. τm=0.001 τs=0.05 τus=2.5 a_fback=0
@named first_neur = ODESystem([
    input ~ input_fn1(t),
    τm * D(v) ~ -v + a1*tanh(v) - a2*tanh(vs) + 
        a3*tanh(vs+0.9) - a4*tanh(vus+0.9) +
        Syn(v2s, -0.2) + input + Syn(v3s, gsyn),
    τs * D(vs)  ~ (v - vs),
    τus * D(vus)  ~ (v - vus),
    gsyn ~ gsyn_val, #IfElse.ifelse(t > 6, 0.05, -.05),
])

@named second_neur = ODESystem([
    input ~ input_fn2(t),
    τm * D(v) ~ -v + a1*tanh(v) - a2*tanh(vs) + 
        a3*tanh(vs+0.9) - a4*tanh(vus+0.9) + Syn(v4s, gsyn) +
        Syn(v1s, -0.2) + input,
    τs * D(vs)  ~ (v - vs),
    τus * D(vus)  ~ (v - vus),
    gsyn ~ gsyn_val,
])

@named third_neur = ODESystem([
    input ~ input_fn3(t),
    τm * D(v) ~ -v + a1*tanh(v) - a2*tanh(vs) + 
        a3*tanh(vs+0.9) - a4*tanh(vus+0.9) +
        Syn(v4s, -0.2) + input + Syn(v1s, gsyn),
    τs * D(vs)  ~ (v - vs),
    τus * D(vus)  ~ (v - vus),
    gsyn ~ gsyn_val, #IfElse.ifelse(t > 6, 0.05, -.05),
])

@named fourth_neur = ODESystem([
    input ~ input_fn4(t),
    τm * D(v) ~ -v + a1*tanh(v) - a2*tanh(vs) + 
        a3*tanh(vs+0.9) - a4*tanh(vus+0.9) + Syn(v2s, gsyn) +
        Syn(v3s, -0.2) + input,
    τs * D(vs)  ~ (v - vs),
    τus * D(vus)  ~ (v - vus),
    gsyn ~ gsyn_val,
])




## Mechanical system
@variables angle(t) d_angle(t) torque(t) direction(t) v1(t) v3(t) disturbance(t)
@variables torque_one(t) torque_two(t) direction_two(t)
@variables motor_config(t)

# Switch on sensory feedback
@parameters tswitch=config_switch_time
switchConfig = (t == tswitch) => 
    [motor_config ~ 1-motor_config]

@named pendulum = ODESystem([
    Jperp * D(D(angle)) ~ torque - damp*(L/2)*D(angle) - m*g*(L/2)*sin(angle),
    # D(D(angle)) ~ -α*D(angle) - sin(angle) + torque,
    
    torque ~ torque_one + torque_two,

    torque_one ~ IfElse.ifelse(v1 > -0.5, K, 0),

    direction_two ~ IfElse.ifelse(motor_config > 0.5, 1, -1),
    torque_two ~ IfElse.ifelse(v3 > -0.5, direction_two*K, 0),
    
    disturbance ~ 0,
    d_angle ~ D(angle),  # Bug if I use D(angle) in event handling.
], t)

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
            ]

prob = ODEProblem(sys, init_sys, (0.0,Tf))
sol = solve(prob,tstops=[config_switch_time],abstol=1e-11, reltol=1e-9,maxiters=2e6,saveat=1e-3);

KE = 0.5 * Jperp * sol[pendulum.d_angle].^2;
GPE = m * 9.81 * (L/2) * ( 1 .- cos.(sol[pendulum.angle]) ) #( cos.(pi.-sol[angle]).-1 )
energy = GPE .+ KE

# Compute steady-state amplitude and frequency.
i = Int(length(sol.t)-1000)
ampfinal = mod(maximum(sol[pendulum.angle][i:end]), 2*pi)
println("Amp: $ampfinal")
pks, _ = findmaxima(sol[pendulum.angle][i:end])
pk_times = sol.t[i:end][pks]
diffs = diff(pk_times); freqfinal = 1 ./ diffs
println("Latest freqs: $freqfinal")
f = freqfinal / (2*pi)

# Compute steady-state freq for large oscillations.
signal = 2*asin.(sin.(sol[pendulum.angle][i:end]/2))
pks_large, pk_values_large = findmaxima(signal)
pk_times_large = sol.t[i:end][pks_large]
diffs_large = diff(pk_times_large)
freqfinal_large = 2 ./ diffs_large # Double as goes from pi to -pi to pi.
freqfinal_large_rounded = round.(freqfinal_large, digits=2)
println("Large osc freqs: $freqfinal_large_rounded")
# plot(sol.t[i:end], signal)
# plot!(pk_times_large, pk_values_large, seriestype=:scatter)
# plot!(sol.t[i:end], sol[pendulum.torque,i:end])

plot_titles = ["" "Small Kick" "Mid Kick" "Large Kick"]
plot_title = plot_titles[rate_idx]


p1 = plot(sol, idxs=[pendulum.torque,pendulum.angle],legend=:topleft, title="$plot_title", dpi=300, size=(1200,800),label=["Torque" "Angle"])

iw = findfirst(x -> x > 0.1, sol.t); iend = findfirst(x -> x > 0.2, sol.t)
p1win = plot(sol[iw:iend], idxs=[pendulum.torque],legend=:topleft, dpi=300, size=(1200,800),label="Torque")
plot!(twinx(), sol[iw:iend], idxs=[pendulum.angle],color=:firebrick2, xaxis="",legend=:topright,linewidth=2,label="Angle")

plot(sol[iw:iend], idxs=[pendulum.angle], yticks=[74*pi, 76*pi,78*pi,80*pi,82*pi])
plot!(twinx(), sol[iw:iend], idxs=[pendulum.torque])

iE = findfirst(x -> x > 200, sol.t)
pE = plot(sol[i:end], idxs=[pendulum.d_angle*pendulum.torque,pendulum.torque],label=["ωτ (power transferred)" "Torque"], dpi=300, size=(1200,800))

p1z = plot(sol[i:end], idxs=[pendulum.torque],legend=:topleft,label="Torque",title="$plot_title", dpi=300, size=(1200,800))
plot!(twinx(), sol[i:end], idxs=[pendulum.angle],color=:firebrick2, xaxis="",legend=:topright,linewidth=2,label="Angle",
        ylims=(-0.55,0.55),yticks=[-0.5,-0.25,0,0.25,0.5])

figure_titles = ["3_bistability-large" "3_bistability-small"]
figure_title = figure_titles[experiment_idx]

t = sol.t;  torque = sol[pendulum.torque];  angle = sol[pendulum.angle];
d_angle = sol[pendulum.d_angle]
save_string = "CDC_paper/figs/$figure_title.jld"
@save save_string t torque angle d_angle

# savefig(p1, "CDC_paper/figs/$figure_title")

p1
