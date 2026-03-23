# autoresearch

This is an experiment to have the LLM do its own research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch_celltracking/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch_celltracking/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed data pipeline: TIFF loading, CSV parsing, dataset construction, normalization, and heatmap target generation. Do not modify.
   - `train.py` — the file you modify. Augmentation, model architecture, optimizer, and training loop.
4. **Verify data exists**: Check that the dataset directory (`./002_nuc_data`) contains the TIFF sequence (`t%03d.tif`) and that the corresponding CSV files (`pos.csv`, `neg.csv`) exist in (`./002_labels`) and are correctly referenced in `train.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a fixed number of epochs (configured inside train.py). You launch it simply as: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: augmentation, model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed data loading (TIFF sequence),CSV parsing, dataset construction, normalization + target generation
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- - Modify the evaluation harness. The `evaluate_fixed_metric` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the lowest best_val_metric.** The training always runs for a fixed number of epochs (30), so you don't need to worry about early stopping or training duration. Everything is fair game: change the architecture, optimizer, hyperparameters, augmentation, batch size, or model size. The only constraint is that the code runs without crashing and completes all epochs successfully.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful improvements in best_val_loss, but it should not grow excessively.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 in best_val_metric improvement that adds 20 lines of hacky code? Probably not worth it. A small val loss improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
best_val_metric: 0.123456
best_epoch: 18
final_val_metric: 0.130000
training_seconds: 120.5
total_seconds: 135.2
peak_vram_mb: 8200.4
num_params_M: 3.2
epochs: 30
batch_size: 8
patch: 256
temporal: True
augmentation: True
```

Note that the script is configured to always stop after 30 epochs, so depending on the computing platform of this computer the numbers might look different. You can extract the key metric from the log file:

```
grep "^best_val_metric:" run.log
```


## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	val_bpb	memory_gb	status	description
commit   best_val_metric   memory_gb   status   description
```

1. git commit hash (short, 7 chars)
2. best validation metric achieved (e.g. 1.234567) — use 0.000000 for crashes
3. peak memory in GB, round to .1f (e.g. 12.3 — divide peak_vram_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit best_val_metric memory_gb status description
a1b2c3d 0.123456 8.0 keep baseline
b2c3d4e 0.118900 8.2 keep increase LR
c3d4e5f 0.125000 8.0 discard switch to larger model
d4e5f6g 0.000000 0.0 crash too large batch (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch_celltracking/mar5` or `autoresearch_celltracking/mar5-gpu0`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^best_val_metric:\|^peak_vram_mb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If best_val_metric improved (lower), you "advance" the branch, keeping the git commit
9. If best_val_metric is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take 30 epochs total. If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!
