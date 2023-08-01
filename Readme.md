This is the code for the paper "Directed Functional Connectivity by Variational Cross-mapping of Psychophysiological Variables".

The proposed method, in its purest sense, attempts to model pairwise coupling between regions of interests evoked by a psychological
event by nonparametrically modeling a generative function of event responses using Gaussian processes with inputs being 
states on reconstructed manifolds. Bayesian evidence comparison for a
"cross-mapping" phenomenon calculated with approximate posterior distributions 
for hyperparameters is used to infer whether directional coupling is significant.

We validate the proposed method using simulations of neurovascular systems emerging from neural state equation dynamics with varying conditions 
and inter-event time intervals (simulations 3 and 4 short-time intervals, simulations 1 and 2 long-time intervals). 

We furthermore simulate various network topolgies: an acyclic one (SimulationAcyclic/) and a cyclic one (SimulationRing/)

Finally, we reproduce DCM connectivity results of low-level face processing networks with the proposed method using an open dataset provided
by the Wu-Minn Human Connectome Project S900 release contrasting emotional faces
and arbitrary shapes. The script _doCCM.py_ runs the PPI-CCM algorithm while  _analyzeResults.py_ runs the statistical analysis of the PPI-CCM results.

We provide an operational jupyter notebook example in the jupyter folder

To cite this work, we have a preprint you can use, which at publication should be associated with the journal DOI:

```
@article{Ghouse2022PPI,
	author = {Ghouse, Ameer and Schultz, Johannes and Valenza, Gaetano},
	doi = {10.1101/2022.10.21.513137},
	elocation-id = {2022.10.21.513137},
	eprint = {https://www.biorxiv.org/content/early/2022/10/21/2022.10.21.513137.full.pdf},
	journal = {bioRxiv},
	publisher = {Cold Spring Harbor Laboratory},
	title = {Directed Functional Connectivity by Variational Cross-mapping of Psychophysiological Variables},
	url = {https://www.biorxiv.org/content/early/2022/10/21/2022.10.21.513137},
	year = {2022},
	bdsk-url-1 = {https://www.biorxiv.org/content/early/2022/10/21/2022.10.21.513137},
	bdsk-url-2 = {https://doi.org/10.1101/2022.10.21.513137}
}
```
