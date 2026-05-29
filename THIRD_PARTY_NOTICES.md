# Third-party notices

## SUMO-RL

This repository contains code copied or adapted from SUMO-RL.

- Project: SUMO-RL
- Upstream repository: https://github.com/LucasAlegre/sumo-rl
- Original author: Lucas Alegre and contributors
- Original license: MIT License
- Upstream commit used as adaptation base: `596b7c63c6d98e2a38785146b75f40ed0f7f456e`
- Preserved license text: `LICENSES/MIT-SUMO-RL.txt`

The original SUMO-RL copyright and permission notice must be preserved in all copies or substantial portions of the software.

## SUMO / Eclipse SUMO

This repository depends on Eclipse SUMO packages such as `eclipse-sumo`, `sumolib`, `traci`, and/or `libsumo`. Treat these as external dependencies. Do not copy SUMO source code into this repository unless its license notices are separately preserved.

## IntersectionZoo

This repository includes IntersectionZoo as a pinned Git submodule for reproducibility.

- Project: IntersectionZoo
- Upstream repository: https://github.com/mit-wu-lab/IntersectionZoo
- Original license: MIT License
- Pinned revision used for the paper: `912d10262cad8ce100d6582459d138e3b34efa4b` (`912d102`)
- Location: `third_party/IntersectionZoo/`
- Preserved license text: `LICENSES/MIT-INTERSECTIONZOO.txt`

IntersectionZoo remains under its original MIT License. This repository does not relicense unmodified IntersectionZoo code under AGPL.

If any IntersectionZoo files are modified in this repository or in a forked submodule, preserve the original MIT notice and clearly mark the modifications.

## Other dependencies

Python package dependencies are listed in `pyproject.toml`. Their inclusion as dependencies does not change the license of this repository.
