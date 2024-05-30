using ModelingToolkit, StructArrays, OhMyREPL
using DifferentialEquations: solve
using Plots, ColorSchemes
using IfElse, JLD
using Distributions: Normal
using Statistics: mean
using Peaks

@variables t
D = Differential(t)
Tf = 300

# Would probably simulate faster if I used discrete events
# instead of all the 'IfElse' statements to check the time.

with_controller = true # Toggle to switch the adaptive control on or off.

experiment_idx = 6 # Up to 6.

# Mechanical System
rho = 1000; L = 0.36*1.0; r = 0.02;
vol = pi * r^2 * L; m = rho * vol; g = 9.81;
Jperp = (1/12)*m*(L^2 + 3*r^2) + m*(L/2)^2; # Second term is parallel angle theorem.
# J = m * L^2; # Assume point mass at the bottom:
damp = 1.0 # 0.4 # Damping.
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

# From the other fig:
# (though would maybe be nicer to demonstrate this on a lower damping)
Amps = [0.37  0.57  0.68  0.79  0.85  0.90]
a_ref = Amps[experiment_idx] # Reference amplitude
# a_ref = 0.87

adaptive_switch_time = 6 # When to switch on adaptive control

a3_init = 0.95
a4_init = 1.45

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
function inv_sigmoid(y, s, d, imin, height)
    -(1/s)*log((height/(y-imin))-1) + d
end
function sigmoid_a3(u)
    sigmoid(u, 1, 3.25, 0.9, 1.5)
end
function inv_sigmoid_a3(y)
    inv_sigmoid(y, 1, 3.25, 0.9, 1.5)
end
function sigmoid_a4(u)
    sigmoid(u, 1.5, 5.25, 1.4, 2.5)
end
function inv_sigmoid_a4(y)
    inv_sigmoid(y, 1.5, 5.25, 1.4, 2.5)
    end

function Syn(vs, gain)
    gain / ( 1 + exp(-2*(vs+1)) )
end

@variables v(t) vs(t) vus(t) v1s(t) v2s(t) v3s(t) v4s(t) input(t) gsyn(t)
@variables a2(t) a3(t) a3sat(t) a4(t) a4sat(t)
@parameters a1=2. τm=0.001 τs=0.05 τus=2.5 a_fback=0
@named first_neur = ODESystem([
    input ~ input_fn1(t),
    τm * D(v) ~ -v + a1*tanh(v) - a2*tanh(vs) + 
        a3sat*tanh(vs+0.9) - a4sat*tanh(vus+0.9) +
        Syn(v2s, -0.2) + input + Syn(v3s, gsyn),
    τs * D(vs)  ~ (v - vs),
    τus * D(vus)  ~ (v - vus),
    gsyn ~ -0.04, #IfElse.ifelse(t > 6, 0.05, -.05),
    a2 ~ 1.6,
    a3sat ~ sigmoid_a3(a3),
    a4sat ~ sigmoid_a4(a4),
])

@named second_neur = ODESystem([
    input ~ input_fn2(t),
    τm * D(v) ~ -v + a1*tanh(v) - a2*tanh(vs) + 
        a3sat*tanh(vs+0.9) - a4sat*tanh(vus+0.9) + Syn(v4s, gsyn) +
        Syn(v1s, -0.2) + input,
    τs * D(vs)  ~ (v - vs),
    τus * D(vus)  ~ (v - vus),
    gsyn ~ -0.04,
    a2 ~ 1.6,
    a3sat ~ sigmoid_a3(a3),
    a4sat ~ sigmoid_a4(a4),
])

@named third_neur = ODESystem([
    input ~ input_fn3(t),
    τm * D(v) ~ -v + a1*tanh(v) - a2*tanh(vs) + 
        a3sat*tanh(vs+0.9) - a4sat*tanh(vus+0.9) +
        Syn(v4s, -0.2) + input + Syn(v1s, gsyn),
    τs * D(vs)  ~ (v - vs),
    τus * D(vus)  ~ (v - vus),
    gsyn ~ -0.04, #IfElse.ifelse(t > 6, 0.05, -.05),
    a2 ~ 1.6,
    a3sat ~ sigmoid_a3(a3),
    a4sat ~ sigmoid_a4(a4),
])

@named fourth_neur = ODESystem([
    input ~ input_fn4(t),
    τm * D(v) ~ -v + a1*tanh(v) - a2*tanh(vs) + 
        a3sat*tanh(vs+0.9) - a4sat*tanh(vus+0.9) + Syn(v2s, gsyn) +
        Syn(v3s, -0.2) + input,
    τs * D(vs)  ~ (v - vs),
    τus * D(vus)  ~ (v - vus),
    gsyn ~ -0.04,
    a2 ~ 1.6,
    a3sat ~ sigmoid_a3(a3),
    a4sat ~ sigmoid_a4(a4),
])


## Mechanical system
@variables angle(t) d_angle(t) torque(t) direction(t) v1(t) v3(t) disturbance(t)
@variables torque_one(t) torque_two(t)

# Switch motor configuration
@variables adaptive_on(t)
@parameters tswitch=adaptive_switch_time
switchAdaptive = (t == tswitch) => 
    [adaptive_on ~ 1]

# Adaptive Control (currently sensing for small oscillations only)
# To measure freq and amp, using MTK docs on 'Continuous Events'
@variables latest_freq(t) latest_event(t) prev_event(t) latest_amp(t)
@variables amp_event_last_period(t) ampcontrol_intermediate_var(t)
@variables out_time(t) back_time(t) # For measuring amp error.

root_eqs = [angle ~ 0] # pendulum passing through origin indicates a half-period
sat_freq = 8 # Just until received first frequency estimate, to stop the controller blowing up.
affect = [prev_event ~ latest_event,
            latest_event ~ t,
            latest_freq ~ IfElse.ifelse(latest_event > 0.01, 2*pi/((latest_event-prev_event)*2), sat_freq),
            ampcontrol_intermediate_var ~ IfElse.ifelse(ampcontrol_intermediate_var == 0, 0, ampcontrol_intermediate_var - 0.5),
            amp_event_last_period ~ ceil(ampcontrol_intermediate_var),] 
root_eqs_amp = [d_angle ~ 0] # Measure amplitude of signal
affect_amp = [latest_amp ~ abs(angle)]
root_eqs_amp_new = [abs(angle) ~ a_ref]
affect_amp_new = [amp_event_last_period ~ 1,
                ampcontrol_intermediate_var ~ 1,
                out_time ~ IfElse.ifelse(angle*d_angle > 0, t, out_time),
                back_time ~ IfElse.ifelse(angle*d_angle > 0, back_time, t)]


continuous_events =  # equation => affect
    [root_eqs => affect
    root_eqs_amp => affect_amp
    root_eqs_amp_new => affect_amp_new]

@named pendulum = ODESystem([
    Jperp * D(D(angle)) ~ torque - damp*(L/2)*D(angle) - m*g*(L/2)*sin(angle),
    # D(D(angle)) ~ -α*D(angle) - sin(angle) + torque,
    
    torque ~ torque_one + torque_two,

    torque_one ~ IfElse.ifelse(v1 > -0.5, K, 0),
    torque_two ~ IfElse.ifelse(v3 > -0.5, -K, 0),
    
    disturbance ~ 0,
    d_angle ~ D(angle),  # Bug if I use D(angle) in event handling.

    # Freq sensor
    D(latest_event) ~ 0.,
    D(prev_event) ~  0.,
    D(latest_freq) ~ 0.,

    # Amplitude sensor
    D(latest_amp) ~ 0.,
    D(amp_event_last_period) ~ 0.,
    D(ampcontrol_intermediate_var) ~ 0.,
    D(out_time) ~ 0.,
    D(back_time) ~ 0.,
], t; continuous_events)

w_ref = 1.0*2*pi # Desired angular frequency
period_frac = 10 # Sets the 'dead-zone' for amplitude control.
@variables theta(t) error(t) amp_error(t) freq(t) amp(t) d_a3(t)
if with_controller
    @parameters k_w = -0.05 k_A = -0.5
else
    @parameters k_w = 0 k_A = 0
end
@named controller = ODESystem([
    amp_error ~ amp - a_ref,
    error ~ freq - w_ref,
    D(a4) ~ adaptive_on*IfElse.ifelse(abs(error) > 0.05, k_w * error, 0),

    # You're in the dead-zone for amp if the time between the 'out' and 'back' Aref events
    # is small, and (to prevent transients interfering) if the last 'back' event was in the last period.
    d_a3 ~ (2*amp_event_last_period-1)*k_A*0.05,
    D(a3) ~ adaptive_on*IfElse.ifelse(back_time-out_time < 2*pi/(period_frac*freq), 
        IfElse.ifelse(t - back_time < 2*pi/(freq), 0, d_a3), d_a3),
    #IfElse.ifelse(abs(amp_error) > 0.02, k_A*amp_error, 0),
], t)

full_sys = compose(ODESystem([
        pendulum.v1 ~ first_neur.v,
        pendulum.v3 ~ third_neur.v,
        
        first_neur.v2s ~ second_neur.vs,
        second_neur.v1s ~ first_neur.vs,
        third_neur.v4s ~ fourth_neur.vs,
        fourth_neur.v3s ~ third_neur.vs,
        first_neur.v3s ~ third_neur.vs,
        third_neur.v1s ~ first_neur.vs,
        second_neur.v4s ~ fourth_neur.vs,
        fourth_neur.v2s ~ second_neur.vs,

        first_neur.a3 ~ controller.a3,
        second_neur.a3 ~ controller.a3,
        third_neur.a3 ~ controller.a3,
        fourth_neur.a3 ~ controller.a3,

        first_neur.a4 ~ controller.a4,
        second_neur.a4 ~ controller.a4,
        third_neur.a4 ~ controller.a4,
        fourth_neur.a4 ~ controller.a4,

        controller.freq ~ pendulum.latest_freq,
        controller.amp ~ pendulum.latest_amp,
        controller.amp_event_last_period ~ pendulum.amp_event_last_period,
        controller.out_time ~ pendulum.out_time, controller.back_time ~ pendulum.back_time,
        
        D(adaptive_on) ~ 0,
        controller.adaptive_on ~ adaptive_on,
    ], t, [], [tswitch]; name=:full_sys, discrete_events=[switchAdaptive]), 
        [first_neur, second_neur, third_neur, fourth_neur, pendulum, controller])

sys = structural_simplify(full_sys)

# Initialise neurons.
init_sys = [first_neur.v => 0, first_neur.vs => 0, 
            first_neur.vus => -1,
            second_neur.v => 0, second_neur.vs => 0,
            second_neur.vus => -0.5,
            third_neur.v => 0, third_neur.vs => 0, 
            third_neur.vus => -1,
            fourth_neur.v => 0, fourth_neur.vs => 0,
            fourth_neur.vus => -0.5,
            pendulum.angle => init_angle,
            pendulum.d_angle => init_d_angle,

            pendulum.latest_event => 0.,
            pendulum.prev_event => 0.,
            pendulum.latest_freq => 0.,
            pendulum.latest_amp => 0.,
            controller.a3 => inv_sigmoid_a3(a3_init),
            controller.a4 => inv_sigmoid_a4(a4_init),
            adaptive_on => 0.,

            pendulum.amp_event_last_period => 0.,
            pendulum.ampcontrol_intermediate_var => 0.,
            pendulum.out_time => 0., pendulum.back_time => 50., # Start very high.
            ]

prob = ODEProblem(sys, init_sys, (0.0,Tf))
sol = solve(prob,tstops=[adaptive_switch_time],abstol=1e-11, reltol=1e-9,maxiters=2e6,saveat=5e-3);

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
freqfinal_large = 2 * t_factor ./ diffs_large # Double as goes from pi to -pi to pi.
freqfinal_large_rounded = round.(freqfinal_large, digits=2)
println("Large osc freqs: $freqfinal_large_rounded")
# plot(sol.t[i:end], signal)
# plot!(pk_times_large, pk_values_large, seriestype=:scatter)
# plot!(sol.t[i:end], sol[pendulum.torque,i:end])

p1 = plot(sol, idxs=[pendulum.torque,pendulum.angle],legend=:topleft, dpi=300, size=(1200,800),label=["Torque" "Angle"])
hline!([a_ref])

p2 = plot(sol, idxs=[controller.error, controller.amp_error],yrange=[-1,0])

iw = findfirst(x -> x > 2, sol.t); iend = findfirst(x -> x > 8, sol.t)
p1win = plot(sol[iw:iend], idxs=[pendulum.torque],legend=:topleft, dpi=300, size=(1200,800),label="Torque")
plot!(twinx(), sol[iw:iend], idxs=[pendulum.angle],color=:firebrick2, xaxis="",legend=:topright,linewidth=2,label="Angle")

plot(sol[iw:iend], idxs=[pendulum.angle], yticks=[74*pi, 76*pi,78*pi,80*pi,82*pi])
plot!(twinx(), sol[iw:iend], idxs=[pendulum.torque])

iE = findfirst(x -> x > 200, sol.t)
pE = plot(sol[i:end], idxs=[pendulum.d_angle*pendulum.torque,pendulum.torque],label=["ωτ (power transferred)" "Torque"], dpi=300, size=(1200,800))

p1z = plot(sol[i:end], idxs=[pendulum.torque],legend=:topleft,label="Torque", dpi=300, size=(1200,800))
plot!(twinx(), sol[i:end], idxs=[pendulum.angle],color=:firebrick2, xaxis="",legend=:topright,linewidth=2,label="Angle",
)#ylims=(-0.55,0.55),yticks=[-0.5,-0.25,0,0.25,0.5])

pFreq = plot(sol, idxs=[controller.error],legend=:topleft)
plot!(twinx(),sol, idxs=[first_neur.a4sat],color=:red,xticks=:none)

pFreqz = plot(sol[6000:end], idxs=[controller.error],legend=:topleft)
plot!(twinx(),sol, idxs=[first_neur.a4sat],color=:red,xticks=:none,xlabel="",legend=:topright)
        
pAmp = plot(sol, idxs=[controller.amp_error],legend=:topleft)
plot!(twinx(),sol, idxs=[first_neur.a3sat],color=:red,xticks=:none,xlabel="",legend=:bottomright)
        

data_title = "4_adaptivecontrol_tuning_continuous_$experiment_idx"


t = sol.t;  torque = sol[pendulum.torque];  angle = sol[pendulum.angle]
freq_error = sol[controller.error]; a4 = sol[controller.a4]; a4sat = sol[first_neur.a4sat]
amp_error = sol[controller.amp_error]; a3 = sol[controller.a3]; a3sat = sol[first_neur.a3sat]
save_string = "CDC_paper/figs/$data_title.jld"
# @save save_string t torque angle freq_error a4 a4sat amp_error a3 a3sat a_ref


p1