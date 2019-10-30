# ECDC_part_Traj
Encoder-decoder of particle trajectory

## Environment
* Keras 2.2.4 (tensorflow backended) 
* Cuda 9.2

## Goals
1. Learn features from DNS-generated particle trajectories 
1. Apply the feature to a simple trajecory to have the DNS features.

## Simplifications
1. Since the DNS flow is horizontally homogeneous ==> only $z_$ direction matters, so drop all horizontal information in the first place.
2. There are many ways to model inertial particles in turbulence, e.g. Mueller and Veron (2014), but I start with the simplest case (using parabola+noise) and leave everything different to the NN.


## Particle trajectories:
Time series of spray droplets are inherently related, so the structure is 4xN_t, where N_t is the length. All time series is resampled for a fixed time step, while it is not neccessarily to have fixed time steps in DNS.
  * vertical location: $z_p(t)$ ==> DNS vs. simplified stochastic model (starting with the simplest way: parabola + noise)
  * flow temprature $T_f(t)$ ==> DNS vs. table look-up from vertical profile given estimated $z_p(t)$
  * flow humidity $q_f(t)$ ==> DNS vs. table look-up from vertical profile based on estimated $z_p(t)$
  * particle temperature $T_p(t)$ ==> DNS vs. equilibrium temperature using Kepert (1996).


## Directories
1. The Encoder-Decoder is stored in "code"
1. Data processing code is stored in the root directory
1. Functions is stored in "func"

## Examples of initial data ($z_p(t)$)
![short trajectories](figs/cls2aMp1_short_zp.png?raw=true "Title")
