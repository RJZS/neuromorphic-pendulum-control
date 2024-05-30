using ModelingToolkit, StructArrays, OhMyREPL
using DifferentialEquations: solve
using Plots, ColorSchemes
using IfElse, JLD
using Distributions: Normal
using Statistics: mean
using Peaks

@variables t
D = Differential(t)
Tf = 6

# Vary 'rate_idx' to vary plot.

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

init_motor_config = 0 # 1 for unidirectional, 0 for bidirectional
config_switch_time = Tf+1 # When to switch the config

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

rate_idx = 3 # For the rate code. 2, 3 or 4 (not using 1).
a4s = [1.57 1.75 2.49 3.56] # Found using 'hco_pendulum.jl'.
k3s = [0.6 0.7 0.9 1.2]

a2 = 0.8*2; # a3 = k3s[rate_idx]*1.5;  
a4 = a4s[rate_idx]

@variables v(t) vs(t) vus(t) v1s(t) v2s(t) v3s(t) v4s(t) input(t) gsyn(t)
@variables a3(t)
@parameters a1=2. τm=0.001 τs=0.05 τus=2.5 a_fback=0
@named first_neur = ODESystem([
    input ~ input_fn1(t),
    τm * D(v) ~ -v + a1*tanh(v) - a2*tanh(vs) + 
        a3*tanh(vs+0.9) - a4*tanh(vus+0.9) +
        Syn(v2s, -0.2) + input,
    τs * D(vs)  ~ (v - vs),
    τus * D(vus)  ~ (v - vus),
    a3 ~ IfElse.ifelse(t > 3, 1.2*1.5, 0.7*1.5),
])

@named second_neur = ODESystem([
    input ~ input_fn2(t),
    τm * D(v) ~ -v + a1*tanh(v) - a2*tanh(vs) + 
        a3*tanh(vs+0.9) - a4*tanh(vus+0.9) +
        Syn(v1s, -0.2) + input,
    τs * D(vs)  ~ (v - vs),
    τus * D(vus)  ~ (v - vus),
    a3 ~ IfElse.ifelse(t > 3, 1.2*1.5, 0.7*1.5),
])


## Mechanical system
@variables angle(t) d_angle(t) torque(t) direction(t) v1(t) v3(t) disturbance(t)
@variables torque_one(t) torque_two(t) direction_two(t)
@variables motor_config(t)

full_sys = compose(ODESystem([
        torque ~ IfElse.ifelse(v1 > -0.5, K, 0),
        v1 ~ first_neur.v,

        first_neur.v2s ~ second_neur.vs,
        second_neur.v1s ~ first_neur.vs,
    ], t; name=:full_sys), 
        [first_neur, second_neur])

sys = structural_simplify(full_sys)

# Initialise neurons.
init_sys = [first_neur.v => 0, first_neur.vs => 0, 
            first_neur.vus => -1,
            second_neur.v => 0, second_neur.vs => 0,
            second_neur.vus => -0.5,
            ]

prob = ODEProblem(sys, init_sys, (0.0,Tf))
sol = solve(prob,abstol=1e-11, reltol=1e-9,maxiters=2e6,saveat=5e-3);

iw = findfirst(x -> x > 0.1, sol.t); iend = findfirst(x -> x > 0.2, sol.t)
p1win = plot(sol[iw:iend], idxs=[torque],legend=:topleft, dpi=300, size=(1200,800),label="Torque")

i = iw;
p1z = plot(sol[i:end], idxs=[torque],legend=:topleft,label="Torque", dpi=300, size=(1200,800))
plot!(twinx(), sol[i:end], idxs=[first_neur.v],color=:firebrick2, xaxis="",legend=:topright,linewidth=2,label="Angle",
        ylims=(-0.55,0.55),yticks=[-0.5,-0.25,0,0.25,0.5])

p1 = plot(sol, idxs=[first_neur.v],label=false)
plot!(sol, idxs=[second_neur.v],label=false,linestyle=:dash)
hline!([-0.5], linestyle=:dash, color=:black, label=false)
p2 = plot(sol, idxs=[torque],label=false)
pT = plot(p1, p2, layout=(2,1))


t = sol.t;  v1 = sol[first_neur.v];  v2 = sol[second_neur.v]
torque = sol[torque]
save_string = "CDC_paper/figs/single_HCO.jld"
@save save_string t v1 v2 torque
        
# savefig(p1, "CDC_paper/figs/$figure_title")

# plot(sol[270:540],idxs=[first_neur.v],label=false,xlabel="",
    #axis=([],false))

pT
