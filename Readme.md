This is the code for the paper "Directed Functional Connectivity by Variational Cross-mapping of Psychophysiological Variables", 
which there will be a preprint out shortly.

The proposed method, in its purest sense, attempts to model pairwise coupling between regions of interests evoked by a psychological
event by nonparametrically modeling a generative function of event responses using Gaussian processes with inputs being 
states on reconstructed manifolds. Bayesian evidence comparison for a
"cross-mapping" phenomenon calculated with approximate posterior distributions 
for hyperparameters is used to infer whether directional coupling is significant.

We validate the proposed method using simulations of neurovascular systems emerging from neural state equation dynamics with varying conditions 
and inter-event time intervals (simulations 3 and 4 short-time intervals, simulations 1 and 2 long-time intervals)

We also validate the proposed method with highly oscillatory chaotic signals that force other highly oscillatory signals (Lorenz Rossler simuls)

Finally, we reproduce DCM connectivity results of low-level face processing networks with the proposed method using an open dataset provided
by the classic Haxby dataset contrasting faces and objects

We provide an operational jupyter notebook example in the jupyter folder
