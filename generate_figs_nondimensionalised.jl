using Plots, JLD, LaTeXStrings, Peaks

## Generate the non-dimensionalised figures.

fontsize_x = 14 # Default is 11
fontsize_y = 14
fontsize_guide = 14
fontsize_guide_large = 22
fontsize_legend = 12
fontsize_legend_large = 16

## Set other defaults.
# plot_font = "Computer Modern"
# default(fontfamily=plot_font)

# Code for plotting axes with π.
# Source: https://discourse.julialang.org/t/plot-with-x-axis-in-radians-with-ticks-at-multiples-of-pi/65325/2
function pitick(start, stop, denom; mode=:text)
    a = Int(cld(start, π/denom))
    b = Int(fld(stop, π/denom))
    tick = range(a*π/denom, b*π/denom; step=π/denom)
    ticklabel = piticklabel.((a:b) .// denom, Val(mode))
    tick, ticklabel
end

function piticklabel(x::Rational, ::Val{:text})
    iszero(x) && return "0"
    S = x < 0 ? "-" : ""
    n, d = abs(numerator(x)), denominator(x)
    N = n == 1 ? "" : repr(n)
    d == 1 && return S * N * "π"
    S * N * "π/" * repr(d)
end

function piticklabel(x::Rational, ::Val{:latex})
    iszero(x) && return L"0"
    S = x < 0 ? "-" : ""
    n, d = abs(numerator(x)), denominator(x)
    N = n == 1 ? "" : repr(n)
    d == 1 && return L"%$S%$N\pi"
    L"%$S\frac{%$N\pi}{%$d}"
end

# Torque and time scaling, from the pendulum's parameters.
rho = 1000; L = 0.36*1.0; r = 0.02; vol = pi * r^2 * L; m = rho * vol; g = 9.81;
Jperp = (1/12)*m*(L^2 + 3*r^2) + m*(L/2)^2; # Second term is parallel angle theorem.
Kfactor = 1 / (m*g*L/2)
tfactor = sqrt(m*g*(L/2)/Jperp) # ̃t = t_factor * t


single_HCO = load("figs/single_HCO.jld")
p1 = plot(single_HCO["t"]*tfactor/(2*pi), single_HCO["v1"],label=L"v_A", ylabel="Voltage",
            dpi=300, size=(1200,510),linewidth=2,
            xtickfontsize=fontsize_x,ytickfontsize=fontsize_y,
    guidefontsize=fontsize_guide_large,legendfontsize=fontsize_legend_large,
    margin=8Plots.mm, yticks=[-3,-1.5,0,1.5,3])
plot!(single_HCO["t"]*tfactor/(2*pi), single_HCO["v2"],label=L"v_B",linestyle=:dash,
    legend=:bottomleft, legendfontsize=fontsize_legend_large+4, legend_column = -1,
    linewidth=2)
hline!([-0.5], linestyle=:dash, color=:black, label=false, linewidth=2)
p2 = plot(single_HCO["t"]*tfactor/(2*pi), single_HCO["torque"]*Kfactor,label=false,xlabel="t / 2π",
    ylabel="Torque",
    dpi=300, size=(1200,510),linewidth=2,
    xtickfontsize=fontsize_x,ytickfontsize=fontsize_y,
    guidefontsize=fontsize_guide_large,legendfontsize=fontsize_legend_large,
    margin=8Plots.mm,yticks=[0,0.5,1])
fig_single = plot(p1, p2, layout=(2,1))

savefig(fig_single, "figs/0-single-HCO")



network = load("figs/neuron_network.jld")
switch_time = network["switch_time"]*tfactor/(2*pi)

pA = plot(network["t"]*tfactor/(2*pi), network["v1"], label=L"v_{1A}",
    legend=:bottomright, dpi=300, size=(1200,675),linewidth=2.,
    xtickfontsize=fontsize_x,ytickfontsize=fontsize_y,
    guidefontsize=fontsize_guide_large,legendfontsize=fontsize_legend_large+4,
    ylabel="Voltage")
    plot!(network["t"]*tfactor/(2*pi), network["v3"], linestyle=:dash,
        linewidth=2, label=L"v_{1B}")
    vline!([switch_time], linestyle=:dash, linecolor=:black, label=false, linewidth=2)
pB = plot(network["t"]*tfactor/(2*pi), network["v2"], label=L"v_{2A}",
    legend=:bottomright, dpi=300, size=(1200,675),linewidth=2, xlabel="t / 2π",
    linecolor=3,xtickfontsize=fontsize_x,ytickfontsize=fontsize_y,
    guidefontsize=fontsize_guide_large,legendfontsize=fontsize_legend_large+4,
    margin=5Plots.mm,
    ylabel="Voltage")
    plot!(network["t"]*tfactor/(2*pi), network["v4"], linestyle=:dash,
        linewidth=2, label=L"v_{2B}", linecolor=4)
    vline!([switch_time], linestyle=:dash, linecolor=:black, label=false, linewidth=2)
pC = plot(network["t"]*tfactor/(2*pi), network["torque"]*Kfactor, label=false,
    legend=:bottomright, dpi=300, size=(1200,675),linewidth=2,
    xtickfontsize=fontsize_x,ytickfontsize=fontsize_y,
guidefontsize=fontsize_guide_large,legendfontsize=fontsize_legend,
ylabel="Torque")
    vline!([switch_time], linestyle=:dash, linecolor=:black, label=false, linewidth=2)

fig_network = plot(pC, pA, pB, layout=(3,1))
# pAB = plot(network["t"], [network["v1"] network["v2"]],label=["v1" "v2"],
#     legend=:topleft, dpi=300, size=(1200,800),linewidth=2)
#     vline!([switch_time], linestyle=:dash, linecolor=:black, label=false, linewidth=2)
# fig_network = plot(pC, pAB, layout=(2,1))

savefig(fig_network, "figs/network-simulation")


overdamped_small = load("figs/overdamped_1.jld")
overdamped_med = load("figs/overdamped_3.jld")
overdamped_large = load("figs/overdamped_6.jld")

p1 = plot(overdamped_small["t"]*tfactor/(2*pi), 
 [overdamped_small["torque"]*Kfactor overdamped_small["angle"]],legend=:topleft, dpi=300, size=(1200,675),label=["Torque" "Position"], linewidth=2,
    xtickfontsize=fontsize_x,ytickfontsize=fontsize_y,
    margin=5Plots.mm,yguidefontsize=fontsize_guide,
    xguidefontsize=fontsize_guide_large,legendfontsize=fontsize_legend_large,
    xticks=[0,2,4,6,8,10],ylabel="Torque | Position")
p2 = plot(overdamped_med["t"]*tfactor/(2*pi), 
 [overdamped_med["torque"]*Kfactor overdamped_med["angle"]],legend=false, dpi=300, size=(1200,675), linewidth=2,
 xtickfontsize=fontsize_x,ytickfontsize=fontsize_y,
 xguidefontsize=fontsize_guide_large,legendfontsize=fontsize_legend,
 margin=5Plots.mm,yguidefontsize=fontsize_guide,
 xticks=[0,2,4,6,8,10],ylabel="Torque | Position")
p3 = plot(overdamped_large["t"]*tfactor/(2*pi), 
 [overdamped_large["torque"]*Kfactor overdamped_large["angle"]],legend=false, xlabel="t / 2π", dpi=300, size=(1200,675), linewidth=2,
 xtickfontsize=fontsize_x,ytickfontsize=fontsize_y,
 xguidefontsize=fontsize_guide_large,legendfontsize=fontsize_legend,
 margin=5Plots.mm,yguidefontsize=fontsize_guide,
 xticks=[0,2,4,6,8,10],ylabel="Torque | Position")

fig_overdamped = plot(p1,p2,p3,layout=(3,1))

savefig(fig_overdamped, "figs/1-overdamped")

## Parameter data (collected using 'adaptivecontrol_freqonly.jl')
# gs = [1.05  1.20  1.35  1.50  1.65  1.80]'
# gus =  [1.73  2.36  2.45  3.06  3.08  3.52]'
# As = [0.37  0.57  0.68  0.79  0.85  0.90]'

## Parameter data (collected using '4_adaptivecontrol_tuning_eventbasedcontroller.jl')
gs = [1.06 1.20 1.36 1.57 1.66 1.80]'
gus = [1.75 2.34 2.49 2.87 3.10 3.52]'
As = [0.37 0.57 0.68 0.80 0.85 0.91]'
fig_monotone_a = plot(gs, As, ylabel="Amplitude [rad]", xlabel=L"$g_s^-$",label=false,
                    markershape=:circle,xtickfontsize=fontsize_x,ytickfontsize=fontsize_y,legendfontsize=fontsize_legend_large,
                    guidefontsize=fontsize_guide_large+4, margin=8Plots.mm,
                    xguidefontsize=fontsize_guide_large+8, markersize=6,
                size=(1200,510),dpi=300)
fig_monotone_b = plot(gus, As, xlabel=L"$g_{us}^+$", label=false,
                    markershape=:circle,xtickfontsize=fontsize_x,ytickfontsize=fontsize_y,legendfontsize=fontsize_legend_large,
                    guidefontsize=fontsize_guide_large+4, margin=8Plots.mm,
                    linecolor=2,markercolor=2,
                    xguidefontsize=fontsize_guide_large+8,markersize=6,
                size=(1200,510),dpi=300)
fig_monotone = plot(fig_monotone_a, fig_monotone_b, layout=(1,2))
savefig(fig_monotone, "figs/1-overdamped-monotonicity")



bistable_large = load("figs/3_bistability-large.jld")
bistable_small = load("figs/3_bistability-small.jld")
j = findfirst(x -> x > 30, bistable_large["t"]) # Manually choosing the relevant time.
angle_l = bistable_large["angle"]
angle_s = bistable_small["angle"]

# 'Wrap around' large oscillations. First, find the jumps.
per_angle = atan.(sin.(angle_l), cos.(angle_l));
amplitudelocs_l, amplitudes_l = findmaxima(per_angle) # Use this later, for the amplitudes
per_diffs = diff(per_angle);
per_idxs = BitVector(undef,length(per_angle));
per_idxs[2:end] = per_diffs .< -6;
per_angle[per_idxs .== 1] .= NaN; # Then, eliminate them.

# Finish collecting amplitudes
amplitudelocs_s, amplitudes_s = findmaxima(bistable_small["angle"])
amplitudes_l[amplitudes_l .> 3.1] .= pi # We know that in these cases, there is a large osc (handling numerical imprecision).

# Now, pre-process the phase plots.
# First, find the instant at which each burst begins.
torquepks_l, _ = findmaxima(bistable_large["torque"])
torquepks_l = torquepks_l[1:2:end]
# Then find the corresponding angle.
phases_l = per_angle[torquepks_l]
# Now compute the energy transferred from each of those instants, until right before
# the next one (ie for each period).
dt = bistable_large["t"][2]
energies_l = zeros(length(torquepks_l)-1) # Last period is incomplete, so ignore.
for idx in 2:length(torquepks_l)-1
    torque_tmp = bistable_large["torque"][torquepks_l[idx]:torquepks_l[idx+1]]
    velocity_tmp = bistable_large["d_angle"][torquepks_l[idx]:torquepks_l[idx+1]]
    energies_l[idx] = sum(torque_tmp .* velocity_tmp)*dt
end

# And repeat for the other state.
torquepks_s, _ = findmaxima(bistable_small["torque"])
torquepks_s = torquepks_s[1:2:end]
angle_s_wrapped = atan.(sin.(angle_s), cos.(angle_s))
phases_s = angle_s_wrapped[torquepks_s]

energies_s = zeros(length(torquepks_s)-1) # Last period is incomplete, so ignore.
for idx in 1:length(torquepks_s)-1
    torque_tmp = bistable_small["torque"][torquepks_s[idx]:torquepks_s[idx+1]]
    velocity_tmp = bistable_small["d_angle"][torquepks_s[idx]:torquepks_s[idx+1]]
    energies_s[idx] = sum(torque_tmp .* velocity_tmp)*dt
end


## Test figs.
#plot(bistable_large["t"][j:end], bistable_large["torque"][j:end])
#plot!(bistable_large["t"][torquepks_l[end-5:end]], phases_l[end-5:end], seriestype=:scatter)
#plot!(bistable_large["t"][j:end], per_angle[j:end],ylims=(0.5,1))


## Plots of angle and torque (not used):
# pBistableLarge = plot(bistable_large["t"][j:end]*tfactor/(2*pi), 
# [bistable_large["torque"][j:end]*Kfactor per_angle[j:end]],legend=:topleft, 
#     dpi=300, size=(1200,510),linewidth=2,
#     xlabel="t / 2π", label=["Torque" "Position"],yticks=pitick(-pi, pi, 2),
#     xtickfontsize=fontsize_x,ytickfontsize=fontsize_y,legend_column = -1,
#     guidefontsize=fontsize_guide,legendfontsize=fontsize_legend_large,
#     margin=5Plots.mm,yrange=(-pi, pi))
# pBistableSmall = plot(bistable_small["t"][j:end]*tfactor/(2*pi), 
# [bistable_small["torque"][j:end]*Kfactor angle_s[j:end].-8*pi],legend=:topleft, 
# dpi=300, size=(1200,510),label=false,linewidth=2,yticks=pitick(-pi, pi, 2),
# xlabel="t / 2π",
# xtickfontsize=fontsize_x,ytickfontsize=fontsize_y, yrange=(-pi,pi),
# guidefontsize=fontsize_guide,legendfontsize=fontsize_legend,
# margin=5Plots.mm)

pEnergyLarge = plot(bistable_large["t"][torquepks_l[1:end-1]]*tfactor/(2*pi), energies_l,
    seriestype=:scatter,label="Energy Transferred",xlabel="t / 2π", dpi=300, size=(1200,600),
    margin=8Plots.mm,
    xtickfontsize=fontsize_x,ytickfontsize=fontsize_y,#legend_column = -1,
    xguidefontsize=fontsize_guide_large+2,#legendfontsize=fontsize_legend_large,
    yguidefontsize=fontsize_guide_large-3,legendfontsize=fontsize_legend_large,
    ylabel="Energy Transferred | Amplitude",ylims=(-0.65,3.55),markersize=6,
    legend=:bottomright)
scatter!(bistable_large["t"][amplitudelocs_l[3:end-3]]*tfactor/(2*pi), amplitudes_l[3:end-3],
    label="Amplitude [rad]")

pEnergySmall = plot(bistable_small["t"][torquepks_s[1:end-1]]*tfactor/(2*pi), energies_s,
    seriestype=:scatter,label=false,xlabel="t / 2π", dpi=300, size=(1200,600),
    margin=8Plots.mm,
    xtickfontsize=fontsize_x,ytickfontsize=fontsize_y,legend_column = -1,
    guidefontsize=fontsize_guide_large+2,legendfontsize=fontsize_legend_large,
    ylims=(-0.65,3.55),markersize=6
    )
scatter!(bistable_large["t"][amplitudelocs_s[3:end-1]]*tfactor/(2*pi), amplitudes_s[3:end-1].%(2*pi),
    label=false)


# pBistableOsc = plot(pBistableLarge, pBistableSmall, layout=(1,2))
# savefig(pBistableOsc, "figs/3-bistability")

pBistableEnergy = plot(pEnergyLarge, pEnergySmall, layout=(1,2))
savefig(pBistableEnergy, "figs/3-bistability-energy")


## Adaptive control:
tuning1 = load("figs/4_adaptivecontrol_tuning_pulsebased_1.jld"); tuning2 = load("figs/4_adaptivecontrol_tuning_pulsebased_2.jld");
tuning3 = load("figs/4_adaptivecontrol_tuning_pulsebased_3.jld"); tuning4 = load("figs/4_adaptivecontrol_tuning_pulsebased_4.jld");
tuning5 = load("figs/4_adaptivecontrol_tuning_pulsebased_5.jld"); tuning6 = load("figs/4_adaptivecontrol_tuning_pulsebased_6.jld");

ig = findfirst(x -> x > 34, tuning1["t"]*tfactor/(2*pi)) # Just used in gains plot.

pTuningFreq = plot(tuning1["t"]*tfactor/(2*pi), tuning1["freq_error"]/tfactor,label=false,yrange=[-0.16,0.11],
    ylabel=L"$e_\omega$", xtickfontsize=fontsize_x,ytickfontsize=fontsize_y,
    guidefontsize=fontsize_guide_large+4,legendfontsize=fontsize_legend_large,
    dpi=300, size=(1200,800), margin=5Plots.mm, linewidth=2) 
plot!(tuning2["t"]*tfactor/(2*pi), tuning2["freq_error"]/tfactor,label=false, linewidth=2)
plot!(tuning3["t"]*tfactor/(2*pi), tuning3["freq_error"]/tfactor,label=false, linewidth=2); plot!(tuning4["t"]*tfactor/(2*pi), tuning4["freq_error"]/tfactor,label=false, linewidth=2)
plot!(tuning5["t"]*tfactor/(2*pi), tuning5["freq_error"]/tfactor,label=false, linewidth=2); plot!(tuning6["t"]*tfactor/(2*pi), tuning6["freq_error"]/tfactor,label=false, linewidth=2)
pTuningFreqGain = plot(tuning1["t"][1:ig]*tfactor/(2*pi), tuning1["a4sat"][1:ig],label=tuning1["a_ref"],
    ylabel=L"g_{us}^+", xtickfontsize=fontsize_x,ytickfontsize=fontsize_y,
    guidefontsize=fontsize_guide_large+2,legendfontsize=fontsize_legend_large,
    legend=:topleft,dpi=300, size=(1200,800), margin=5Plots.mm, linewidth=2)
plot!(tuning2["t"][1:ig]*tfactor/(2*pi), tuning2["a4sat"][1:ig],label=tuning2["a_ref"], linewidth=2)
plot!(tuning3["t"][1:ig]*tfactor/(2*pi), tuning3["a4sat"][1:ig],label=tuning3["a_ref"], linewidth=2)
plot!(tuning4["t"][1:ig]*tfactor/(2*pi), tuning4["a4sat"][1:ig],label=tuning4["a_ref"], linewidth=2)
plot!(tuning5["t"][1:ig]*tfactor/(2*pi), tuning5["a4sat"][1:ig],label=tuning5["a_ref"], linewidth=2)
plot!(tuning6["t"][1:ig]*tfactor/(2*pi), tuning6["a4sat"][1:ig],label=tuning6["a_ref"], linewidth=2)

pTuningAmp = plot(tuning1["t"]*tfactor/(2*pi), tuning1["amp_error"], yrange=[-0.6,0.03],label=tuning1["a_ref"],
    ylabel=L"$e_A$",  xtickfontsize=fontsize_x,ytickfontsize=fontsize_y,
    guidefontsize=fontsize_guide_large+4,legendfontsize=fontsize_legend_large,
    dpi=300, size=(1200,800),xlabel="t / 2π", margin=5Plots.mm, linewidth=2,
    legend=:bottomright,
    #xlabel="t [s]",
    #xtickfontsize=fontsize_x,ytickfontsize=fontsize_y,
    #guidefontsize=fontsize_guide,
)#legendfontsize=10)
    #,margin=5Plots.mm) 
plot!(tuning2["t"]*tfactor/(2*pi), tuning2["amp_error"],label=tuning2["a_ref"], linewidth=2)
plot!(tuning3["t"]*tfactor/(2*pi), tuning3["amp_error"],label=tuning3["a_ref"], linewidth=2); plot!(tuning4["t"]*tfactor/(2*pi), tuning4["amp_error"],label=tuning4["a_ref"], linewidth=2)
plot!(tuning5["t"]*tfactor/(2*pi), tuning5["amp_error"],label=tuning5["a_ref"], linewidth=2); plot!(tuning6["t"]*tfactor/(2*pi), tuning6["amp_error"],label=tuning6["a_ref"], linewidth=2)
pTuningAmpGain = plot(tuning1["t"][1:ig]*tfactor/(2*pi), tuning1["a3sat"][1:ig],label=false,
    xlabel="t / 2π",ylabel=L"g_s^-", xtickfontsize=fontsize_x,ytickfontsize=fontsize_y,
    guidefontsize=fontsize_guide_large,legendfontsize=fontsize_legend_large,
    dpi=300, size=(1200,800), margin=5Plots.mm, linewidth=2)
plot!(tuning2["t"][1:ig]*tfactor/(2*pi), tuning2["a3sat"][1:ig],label=false, linewidth=2)
plot!(tuning3["t"][1:ig]*tfactor/(2*pi), tuning3["a3sat"][1:ig],label=false, linewidth=2)
plot!(tuning4["t"][1:ig]*tfactor/(2*pi), tuning4["a3sat"][1:ig],label=false, linewidth=2)
plot!(tuning5["t"][1:ig]*tfactor/(2*pi), tuning5["a3sat"][1:ig],label=false, linewidth=2)
plot!(tuning6["t"][1:ig]*tfactor/(2*pi), tuning6["a3sat"][1:ig],label=false, linewidth=2)

tuningFig_errors = plot(pTuningFreq, pTuningAmp, layout=(2,1))
tuningFig_gains = plot(pTuningFreqGain, pTuningAmpGain, layout=(2,1))
savefig(tuningFig_errors, "figs/tuning-errors-pulsebased")
savefig(tuningFig_gains, "figs/tuning-gains-pulsebased")




## Pulse response 

## (See script `6_phase_response.jl`)


## New phase control figs
pc_short = load("figs/switching_phasecontrol-short_pulses.jld")

# 'Wrap around' large oscillations. First, find the jumps.
per_angle_short = atan.(sin.(pc_short["angle"]),
                cos.(pc_short["angle"]))
per_diffs = diff(per_angle_short)
per_idxs = BitVector(undef,length(per_angle_short))
per_idxs[2:end] = per_diffs .< -6
per_angle_short[per_idxs .== 1] .= NaN

# Find the phase.
torquepks_pc, _ = findmaxima(pc_short["torque"])
# Need to manually select torquepks, as initially there are some bursts with >2 spikes.
torquepks_pc = [torquepks_pc[[1,3,5,7,9,11,14,16,18,21,23,26,28,31,34,36,39]]; torquepks_pc[41:2:end]]
# Then find the corresponding angle.
phases_pc = per_angle_short[torquepks_pc]
# Now compute the energy transferred from each of those instants, until right before
# the next one (ie for each period).
dt_pc = pc_short["t"][2]
energies_pc = zeros(length(torquepks_pc)-1) # Last period is incomplete, so ignore.
for idx in 2:length(torquepks_pc)-1
    torque_tmp = pc_short["torque"][torquepks_pc[idx]:torquepks_pc[idx+1]]
    velocity_tmp = pc_short["d_angle"][torquepks_pc[idx]:torquepks_pc[idx+1]]
    energies_pc[idx] = sum(torque_tmp .* velocity_tmp)*dt_pc
end

nb = 8 # Number of bursts shown in transient plot.
iy = findfirst(x -> x > 10, pc_short["t"])
iz = findfirst(x -> x > 26, pc_short["t"])
## Separately plot the transient and the steady-state of the short-pulse case.
fig_pc_short_transient = plot(pc_short["t"][1:iy]*tfactor/(2*pi), [pc_short["torque"][1:iy]*Kfactor, per_angle_short[1:iy]],
    legend=:bottomleft, dpi=300, size=(1200,510),linewidth=2, xlabel="t / 2π",
    #label=false, 
    label=["Torque" "Position"],
    xtickfontsize=fontsize_x,ytickfontsize=fontsize_y,
    xguidefontsize=fontsize_guide_large,legendfontsize=fontsize_legend_large,margin=8Plots.mm,
    yguidefontsize=fontsize_guide+6,ylabel="Torque | Position | Current",
    yticks=pitick(-pi, pi, 2),xticks=[0,2,4,6,8,10]
)
plot!(pc_short["t"][1:iy]*tfactor/(2*pi), 2*pc_short["pulse_one"][1:iy],
    linewidth=3, label="ipulse,A (scaled)",linecolor=4)
scatter!(pc_short["t"][torquepks_pc[1:nb]]*tfactor/(2*pi), 2*ones(nb), label=false, color=7,
    markersize=6)

# fig_pc_short = plot(pc_short["t"][iz:end]*tfactor/(2*pi), [pc_short["torque"][iz:end]*Kfactor, per_angle_short[iz:end]],
#     legend=:bottomleft, dpi=300, size=(1200,510),linewidth=2, xlabel="t / 2π",
#     label=false, 
#     #label=["u [Nm]" "q [rad]"],
#     xtickfontsize=fontsize_x,ytickfontsize=fontsize_y,
#     guidefontsize=fontsize_guide_large,legendfontsize=fontsize_legend_large,margin=8Plots.mm,
#     yticks=pitick(-pi, pi, 2)
# )
# plot!(pc_short["t"][iz:end]*tfactor/(2*pi), [2*pc_short["pulse_one"][iz:end]],#, 2*pc_short["pulse_two"][iz:end]],
#     linewidth=3, label=false)#["ipulse,A (scaled)" "ipulse,B (scaled)"])


fig_pc_energy = plot(pc_short["t"][torquepks_pc[1:end-1]][nb+1:end]*tfactor/(2*pi), energies_pc[nb+1:end],
seriestype=:scatter,label=false,xlabel="t / 2π", dpi=300, size=(1200,510),
margin=8Plots.mm,ylabel="Energy Transferred",
xtickfontsize=fontsize_x,ytickfontsize=fontsize_y,legend_column = -1,
guidefontsize=fontsize_guide_large,legendfontsize=fontsize_legend_large,
markersize=6
)
scatter!(pc_short["t"][torquepks_pc[1:nb]]*tfactor/(2*pi), energies_pc[1:nb], label=false,
color=7, markersize=6)


# fig_pc = plot(fig_pc_short_transient, fig_pc_short, layout=(1,2))
# savefig(fig_pc, "figs/6-phasecontrol")

fig_pc_alternative = plot(fig_pc_short_transient, fig_pc_energy, layout=(1,2))
savefig(fig_pc_alternative, "figs/6-phasecontrol-new")
