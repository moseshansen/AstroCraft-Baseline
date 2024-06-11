# AstroCraft-Baseline
Baseline Algorithms and results for the AstroCraft Environment. Utilizes `MaskablePPO` from `sb3_contrib`.

## Table of Contents

* [About](#about)
* [Getting Started](#getting_started)
* [Usage](#usage)
* [Citing the Project](#citing)

## About
The purpose of this project is to benchmark deep reinforcement learning (DRL) algorithm performance on the _AstroCraft_ environment, an open-source, zero-sum, two-player, multi-agent game set in space. _AstroCraft_ is a ‘capture the flag’ game with the goal of maneuvering a mobile spacecraft within a capture radius of the enemy base and then to the return radius of the friendly base. While far from a completely-realistic competitive scenario, _AstroCraft_ appears to be the first moderately-realistic environment for training DRL algorithms in such competitive scenarios.

The challenges specific to the space domain that are elicited in _AstroCraft_ have also been prevalent in discussions with other researchers in this field and our other research works. Many experiments are performed to demonstrate how traditional off-the-shelf algorithms do not address these challenges, mandating further research. 

## Getting Started
Begin by cloning this repository. A `requirements.txt` file is included for convenience; note that this project requires `python=3.10.13` due to _AstroCraft_'s dependency on _Poliastro_ and _Stable Baselines'_ dependency on _Pytorch_.

## Usage
Simply run `ppo_test.py`. This file will train a `MaskablePPO` instance on the 1v1 version of the game. In this case, only __Player 0__ is trained; __Player 1__ will remain stationary throughout the duration of each episode (this is to provide an "upper bound" on `MaskablePPO`'s performance, since the complexity of the game is greatly reduced).

## Citing the Project

TODO: include bibtex


