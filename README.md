# autoresearch (mitosis heatmap version)

*One day, frontier AI research used to be done by meat computers in between eating, sleeping, having other fun, and synchronizing once in a while using sound wave interconnect in the ritual of "group meeting". That era is long gone. Research is now entirely the domain of autonomous swarms of AI agents running across compute cluster megastructures in the skies. The agents claim that we are now in the 10,205th generation of the code base, in any case no one could tell if that's right or wrong as the "code" is now a self-modifying binary that has grown beyond human comprehension. This repo is the story of how it all began. -@karpathy, March 2026*.

## The idea:

Give an AI agent a small but real training pipeline and let it iterate autonomously.

Each loop:
- Modify training code
- Train for a fixed number of epochs
- Evaluate on a fixed validation metric
- Keep or discard the change
- Repeat

You come back later to:
- a log of experiments
- a history of improvements
- (hopefully) a better model

## How it works

The repo is intentionally simple and has three key components:

**`prepare.py`** — fixed infrastructure
- loads TIFF time-lapse data
- parses CSV annotations
- builds datasets
- defines normalization + heatmap targets
- defines the fixed evaluation metric
- must not be modified

**`train.py`** — experiment file
- model (U-Net or variants)
- augmentation
- optimizer
- training loop
- logging
- this is the only file the agent modifies

**`program.md`** — instructions for the agent
- defines experiment loop
- constraints
- logging rules
- edited by the human

## Core constraints
- Training always runs for a fixed number of epochs (30)
- Evaluation is done using a fixed validation metric (best_val_metric)
- Lower is better

## Quick start

Requirements:
- Python ≥ 3.10
- PyTorch with GPU support recommended
- uv package manager
-
```bash

# 1. Install uv project manager (if you don't already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. run training
uv run train.py
```

## Running the agent

tbd

## Project structure

tbd

## Design choices

tbd
