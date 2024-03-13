# Neuromorphic Control of a Pendulum

This readme provides the non-dimensionalised simulation parameters for the L-CSS/CDC submission 'Neuromorphic Control of a Pendulum'. The code itself will be uploaded in due course.

All simulations have $g_f^- = 2$ and $g_s^+ = 1.6$. The intra-HCO synapes have gain $g_\rm{syn,ij} = -0.2$ and the inter-HCO synapses have gain $g_\rm{syn,ij} = \pm 0.04$.

## Figure 2 - An HCO and its motor

$g_{us}^+ = 2.36$

We set $g_s^-$ to 1.05 for the lower burst size, and to 1.8 for the higher burst size.

## Fig 4 - Network Simulation

$g_s^- = 1.05$ and $g_{us}^+ = 1.75$

## Fig 5 - Overdamped Regime (Small Oscillations)

Top:       $g_s^- = 1.05$ and $g_{us}^+ = 1.73$

Middle:  $g_s^- = 1.35$ and $g_{us}^+ = 2.45$

Bottom: $g_s^- = 1.80$ and $g_{us}^+ = 3.52$

## Fig 7 - Underdamped Regime (Bistability)

 $g_s^- = 1.05$ and $g_{us}^+ = 1.75$

Initial conditions: $q(0) = 0$, $\dot{q}(0) = 0$ for large oscillations and $q(0) = 1, \dot{q}(0) = 0$ for small oscillations.

## Fig 8 - Adaptive Control

$k_\omega = 0.05$ and $k_A = 0.5$

$\delta_\omega = 0.078$ and $\delta_A = 0.02$.

The saturation functions are

$\sigma_\omega(x) = 1.4 + 2.5/(1+\exp(-1.5(x-5.25)))$

$\sigma_A(x) = 0.9 + 1.5/(1+\exp(-(x-3.25)))$

## Fig 11 - Phase Control

 $g_s^- = 1.05$ and $g_{us}^+ = 1.75$

