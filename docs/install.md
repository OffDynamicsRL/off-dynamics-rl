# Installation Guide

Our benchmark is *installation-free*, i.e., one does not need to run `pip install -e .`. This design choice is motivated by the fact that users may have multiple local environments which actually share numerous packages like `torch`, making it a waste of space to create another conda environment for running ODRL. Moreover, the provided packages may conflict with existing ones, posing a risk of corrupting the current environment. As a result, we do not offer a `setup.py` file. ODRL relies on some most commonly adopted packages, which should be easily satisfied: `python==3.8.13, torch==1.11.0, gym==0.18.3, dm-control==1.0.8, numpy==1.23.5, d4rl==1.1, mujoco-py==2.1.2.14`.

## Basic Usage

To use our benchmark, clone the repository and then you can start the journey!

```
git clone https://github.com/OffDynamicsRL/off-dynamics-rl.git
cd off-dynamics-rl
```

## Install from Packages

Nevertheless, we totally understand that some users may still need the detailed list of dependencies, and hence we also include the `requirement.txt` in ODRL. To use it, run the following commands:
```bash
conda create -n offdynamics python=3.8.13 && conda activate offdynamics
pip install setuptools==63.2.0
pip install wheel==0.38.4
pip install -r requirement.txt
```