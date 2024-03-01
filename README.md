# Neuromorphic Control of a Pendulum

This readme provides the simulation parameters for the L-CSS/CDC submission 'Neuromorphic Control of a Pendulum'. The code itself will be uploaded in due course.

All simulations have $g_f^- = 2$ and $g_s^+ = 1.6$. The intra-HCO synapes have gain $g_\rm{syn,ij} = -0.2$ and the inter-HCO synapses have gain $g_\rm{syn,ij} = \pm 0.04$.

## Pendulum

The pendulum is modelled as a uniform solid cylinder. Its moment of inertia is therefore $J = (m/12)(L^2 + 3r^2) + m(L/2)^2$, where $r$ is the radius of the cylinder.

## Figure 4 - An HCO and its motor

$g_{us}^+ = 2.36$

We set $g_s^-$ to 1.05 for the lower duty cycle, and to 1.8 for the higher duty cycle.

## Fig 6 - Network Simulation

$g_s^- = 1.05$ and $g_{us}^+ = 1.75$

## Fig 7 - Overdamped Regime (Small Oscillations)

Top:       $g_s^- = 1.05$ and $g_{us}^+ = 1.73$

Middle:  $g_s^- = 1.35$ and $g_{us}^+ = 2.45$

Bottom: $g_s^- = 1.80$ and $g_{us}^+ = 3.52$

## Fig 9 - Underdamped Regime (Large Oscillations)

 $g_s^- = 1.05$ and $g_{us}^+ = 1.75$

Initial conditions: The angle $q$, the velocity $\dot{q}$ and all voltages were initialised at 0.

## Fig 10 - Bistability in the Underdamped Regime

 $g_s^- = 1.05$ and $g_{us}^+ = 1.75$

The switching times were 8.2 seconds and 8.5 seconds for the top and bottoms rows respectively.

Initial conditions: $q(0) = 0$, $\dot{q}(0) = 0$. For all $i$, we set $v_i(0) = 0$ and $v_{s,i}(0)=0$. To reduce the length of the transient at the start of the simulation, we set $v_{us,1A}(0) = v_{us,2A}(0) = -1$ and $v_{us,1A}(0) = v_{us,2A}(0) = -0.5$ (these values were hand-tuned).

## Figs 11-12 - Adaptive Control

$k_\omega = 0.05$ and $k_A = 0.5$

$\delta_\omega = 0.05$ rad/s and $\delta_A = 0.02$ rad.

## Fig 13 - Adaptive Control with a Perturbation

$k_\omega = 0.02$ and $k_A = 0.1$

$\delta_\omega = 0.1$ rad/s and $\delta_A = 0.02$ rad.

## Figs 15-16 - Phase Control

 $g_s^- = 1.05$ and $g_{us}^+ = 1.75$

