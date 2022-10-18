# A posteriori time series aggregation for energy system planning models with storage




## Summary

This repository contains data, model files and example code for the paper *Reducing climate risk in energy system planning: a posteriori time series aggregation for models with storage*.

<!-- This repository contains data, model files and example code for the paper [*Importance subsampling for power system planning under multi-year demand and weather variability*](https://ieeexplore.ieee.org/abstract/document/9183591) (2020) (publicly available version [here](https://arxiv.org/abs/2008.10300)). -->




## Entry point

A single bash script runs all simulations in the paper, post-processes the results and generates
the figures. To run it, call

```
sh scripts/main.sh
```

from a command line in the main directory (**not** in the `scripts/` directory). This, in turn,
runs three scripts:
- `run_validation.sh`: the *validation* experiment
- `run_example.sh`: the *example* experiment
- `make_figures.sh`: collate and clean the data, and create the figures. These appear in the directory `outputs/plots_post/`.

In this repo, this code is structured to run all simulations in series. However, each of the 40 replications can also be run in parallel -- you can do this for your machine by running each `REPLICATION` separately (this variable appears in the `.sh` files).




## Contains

- `models/`: power system model generating files, for `Calliope` (see acknowledgements)
- `data/`: demand and weather time series data
- `model_files/`: power system model generating files, for `Calliope` (see acknowledgements)
- `models/`: python code to run the models
- `outputs/`: where simulation outputs and figures are stored
- `scripts/`: bash shell to run experiments and create figures
- various `.py` functions to run the simulations and create figures




## Requirements & Installation

Running the code in this repo requires two things: some `python` packages and a solver for the optimisation problem. For a very quick way to install these, follow the `Requirements & installation` instructions for [this repo](https://github.com/ahilbers/renewable_test_PSMs/#requirements-&-installation), as it has the same dependencies. Otherwise, install the following:
- Python modules:
  - `Calliope 0.6.7`: A (fully open-source) energy system model generator. See [this link](https://calliope.readthedocs.io/en/stable/user/installation.html) for installation. **See note below for additional requirement**.
  - `numpy` (`pip install numpy`)
  - `pandas` (`pip install pandas`)
  - `yaml` (`pip install pyyaml`)
  - `sklearn` (`pip install scikit-learn`)
- Other:
  - `cbc`: open-source optimiser: see [this link](https://projects.coin-or.org/Cbc) for installation. Other solvers (e.g. `gurobi`) are also possible -- the solver can be specified in `model_files/model.yaml`.
- **Extra**: The current Calliope implementation can lead to errors due to floating point storage levels around zero, as detailed [here](https://github.com/calliope-project/calliope/issues/379). To avoid this change, implement [this pull request](https://github.com/calliope-project/calliope/pull/380). You can do this yourself by changing [this line](https://github.com/calliope-project/calliope/pull/380/files) in the file `{YOUR_PATH_TO_CALLIOPE_PACKAGE}/calliope/backend/run.py`.




## How to cite

If you use this repository for further research, please cite the following papers:

- AP Hilbers, DJ Brayshaw, A Gandy. A posteriori time series aggregation for energy system planning models with storage (to appear).



## Contact

[Adriaan Hilbers](https://ahilbers.github.io/). Department of Mathematics, Imperial College London. [a.hilbers17@imperial.ac.uk](mailto:a.hilbers17@imperial.ac.uk).




## Acknowledgements

Models are constructed in the modelling framework `Calliope`, created by Stefan Pfenninger and Bryn Pickering. See [callio.pe](https://callio.pe) or the following paper for details:

- Pfenninger, S. and Pickering, B. (2018). Calliope: a multi-scale energy systems modelling framework. Journal of Open Source Software, 3(29), 825, doi:[10.21105/joss.00825](https://doi.org/10.21105/joss.00825).

The demand and wind dataset is based on work by Hannah Bloomfield et al. Details can be found in the following paper and dataset:

- Bloomfield, H. C., Brayshaw, D. J. and Charlton-Perez, A. (2019) Characterising the winter meteorological drivers of the European electricity system using Targeted Circulation Types. Meteorological Applications. ISSN 1469-8080 (In Press). doi:[10.1002/met.1858](https://doi.org/10.1002/met.1858)

- HC Bloomfield, DJ Brayshaw, A Charlton-Perez (2020). MERRA2 derived time series of European country-aggregate electricity demand, wind power generation and solar power generation. University of Reading. Dataset. doi:[10.17864/1947.239](https://doi.org/10.17864/1947.239)
