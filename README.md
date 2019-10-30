# ECDC_part_Traj
Encoder-decoder of particle trajectory

Goal: Learn features from DNS-generated particle trajectories 
Particle trajectories:
-- vertical location: z_p(t) ==> DNS vs. simplified stochastic model (starting with a simplest parabalo + noise)
-- flow temprature T_f(t) ==> DNS vs. table look-up from vertical profile given estimated z_p(t)
-- flow humidity q_f(t) ==> DNS vs. table look-up from vertical profile based on estimated z_p(t)
-- particle temperature T_p(t) ==> DNS vs. equilibrium temperature using Kepert (1996).


## code
1. The Encoder-Decoder is stored in directory code
2. Data processing code is stored in the root directory
3. Functions is stored in func directory
```
