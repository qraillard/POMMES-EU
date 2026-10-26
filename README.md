# POMMES-EU
This repository presents running code and input data for a European electricity and hydrogen model. It runs on [POMMES-KUL](https://github.com/qraillard/pommes-kuleuven), the KU Leuven version of [POMMES](https://git.persee.minesparis.psl.eu/energy-alternatives/pommes), an open source framework to model investment pathways in multi-energy systems.

The framework enables to minimise the system costs to meet the energy services demand by modelling the investment
and operating costs of energy conversion, storage and transport technologies.

## Model and data
POMMES-EU is based on data from [ERAA2024](https://www.entsoe.eu/eraa/2024/) and [TYNDP2024](https://www.entsoe.eu/outlooks/tyndp/2024/), integrating 35 countries (each represented by a single modeling node), four investment horizons, and three operational years—each corresponding to a distinct weather scenario—between each investment phase. A description of the input data is available in [`doc/data_description.pdf`](doc/data_description.pdf).

![image](readme_images/POMMES-EU-Horizons.png)
[CY=Climate Year, WS=Weather Scenario]

## Scenarios
The repository was developed to study electricity and hydrogen capacity expansion in production and storage technologies under different hydrogen infrastructure settings. The dataset includes 3 scenarios, each with its own configuration file:

| Scenario | Configuration file |
|---|---|
| ISO - Isolated Hydrogen Systems | `config.yaml` |
| EHB - European Hydrogen Backbone | `config_ehb.yaml` |
| EHB+I - European Hydrogen Backbone allowing for extra-european imports | `config_ehb+i.yaml` |

With a focus on Belgium, the dataset provides as well another case redispatching hydrogen demand in Belgium among different hydrogen consuming industry clusters to assess the impact of an increase in spatial granularity.

![image](readme_images/merged_maps.png)

## Running a study
Install POMMES-KUL following [its instructions](https://github.com/qraillard/pommes-kuleuven#installation), then run from the root of this repository:

```bash
python run_study.py -c config_ehb.yaml -t 4
```

| Option | Effect |
|---|---|
| `-c <file>` | Configuration file (default `config.yaml`, i.e. the ISO scenario) |
| `-s <solver>` | Solver: `gurobi` (default), `cplex`, `highs`, ... |
| `-t <n>` | Number of threads (default 4) |
| `-n <n>` | Keep only the first `n` countries of the default list of 27 |
| `-m` | Hydrogen must-run setting |
| `-d` | Daily hydrogen demand |
| `-h` | Add the Belgian industry hubs |
| `-b <0\|1\|2>` | Belgian hydrogen backbone setting (requires `-h`) |
| `-r <0\|1>` | Methane-based hydrogen ramping off/on (default `1`) |

At most one of `-m`, `-d` and `-h` can be used in a single run.

The default solver, Gurobi, is commercial (free academic licences are available). HiGHS is open source, but according to the authors it cannot solve this type of problem in a reasonable time.

## Input data
The inputs are semicolon-separated CSV files in `data/`, listed in the configuration files. They are stored with [Git LFS](https://git-lfs.com) and weigh about 3.8 GB in total (`availability.csv` alone is 2 GB). Install `git-lfs` before cloning. GitHub's "Download ZIP" button does not include them: it only provides small pointer files.

## License
MIT, see [`LICENSE.txt`](LICENSE.txt). Contributors are listed in [`AUTHORS`](AUTHORS).
