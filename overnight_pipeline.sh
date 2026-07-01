#!/bin/bash
# Overnight pipeline: evaluate run5 → train run6
# Run from repo root with:
#   bash overnight_pipeline.sh > pipeline.log 2>&1 &
set -e

REPO=/c/Users/alin2/GAN-PH
RUN6_WORKTREE=/c/Users/alin2/GAN-PH-run6

log() { echo "[pipeline $(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# ── PHASE 1: Wait for run5 epoch 50 ──────────────────────────────────────────
log "=== PHASE 1: Waiting for run5 epoch 050 ==="
log "Checking $REPO/1_GAN/log.dat for [050/050][026/026]...tpb_loss"

until grep -q "\[050/050\]\[026/026\].*tpb_loss" "$REPO/1_GAN/log.dat" 2>/dev/null; do
    # Report progress every 5 minutes
    latest=$(grep "tpb_loss" "$REPO/1_GAN/log.dat" 2>/dev/null | tail -1 || echo "(no entries yet)")
    log "Still waiting... latest: $latest"
    sleep 300
done

log "run5 epoch 050 detected in log.dat"
log "Sleeping 120s for checkpoint save and image render..."
sleep 120

# Confirm checkpoint exists
CKPT="$REPO/1_GAN/save_model/Generator_050epoch.pth"
if [ ! -f "$CKPT" ]; then
    log "ERROR: checkpoint not found at $CKPT — aborting"
    exit 1
fi
log "Checkpoint confirmed: $CKPT"

# ── PHASE 2: Evaluate run5 ────────────────────────────────────────────────────
log "=== PHASE 2: Evaluating run5 ==="

log "Generating 50 structures from run5 checkpoint..."
cd "$REPO/4_CNNCT"
conda run -n ganph --no-capture-output python -u generate_structures.py \
    --training-data ../real_data \
    --output ../generated_data \
    --n 50

log "Running S-value analysis..."
conda run -n ganph --no-capture-output python -u analyze.py \
    --input ../real_data \
    --compare ../generated_data \
    --output s_values_run5.csv 2>&1 | tee "$REPO/1_GAN/run5_svalues.log"

log "=== run5 EVALUATION DONE ==="
log "Results CSV : $REPO/4_CNNCT/s_values_run5.csv"
log "S-value CSV : $REPO/4_CNNCT/s_values_run5_svalues.csv"
log "Summary log : $REPO/1_GAN/run5_svalues.log"

# ── PHASE 3: Set up run6 worktree ─────────────────────────────────────────────
log "=== PHASE 3: Setting up run6 worktree ==="

# Remove stale worktree if it exists from a previous attempt
if [ -d "$RUN6_WORKTREE" ]; then
    log "Removing stale worktree at $RUN6_WORKTREE"
    git -C "$REPO" worktree remove --force "$RUN6_WORKTREE" 2>/dev/null || rm -rf "$RUN6_WORKTREE"
fi

log "Creating worktree for run6-weighted-tau-ysz at $RUN6_WORKTREE"
git -C "$REPO" worktree add "$RUN6_WORKTREE" run6-weighted-tau-ysz

log "Worktree created."

# ── PHASE 4: Train run6 ───────────────────────────────────────────────────────
log "=== PHASE 4: Starting run6 training (weighted tau YSZ 3x) ==="

cd "$RUN6_WORKTREE/1_GAN"
conda run -n ganph --no-capture-output python -u main.py \
    --data "$REPO/real_data" \
    --lr 0.00005 \
    --epochs 50 \
    --tau-estimator "$REPO/5_TAU/save_model/tau_net.pth" \
    --tau-targets "$REPO/5_TAU/tau_targets.json" 2>&1 | tee "$REPO/1_GAN/run6_output.log"

log "=== PHASE 4: run6 training DONE ==="
log "Checkpoint: $RUN6_WORKTREE/1_GAN/save_model/Generator_050epoch.pth"
log "Training log: $REPO/1_GAN/run6_output.log"

# ── PHASE 5: Evaluate run6 ────────────────────────────────────────────────────
log "=== PHASE 5: Evaluating run6 ==="

# Copy run6 checkpoint to a known location for generate_structures.py
mkdir -p "$REPO/1_GAN/save_model_run6"
cp "$RUN6_WORKTREE/1_GAN/save_model/Generator_050epoch.pth" \
   "$REPO/1_GAN/save_model_run6/Generator_050epoch.pth"

log "Generating 50 structures from run6 checkpoint..."
cd "$REPO/4_CNNCT"
conda run -n ganph --no-capture-output python -u generate_structures.py \
    --weights ../1_GAN/save_model_run6/Generator_050epoch.pth \
    --training-data ../real_data \
    --output ../generated_data_run6 \
    --n 50

log "Running S-value analysis for run6..."
conda run -n ganph --no-capture-output python -u analyze.py \
    --input ../real_data \
    --compare ../generated_data_run6 \
    --output s_values_run6.csv 2>&1 | tee "$REPO/1_GAN/run6_svalues.log"

log "=== run6 EVALUATION DONE ==="
log "Results CSV : $REPO/4_CNNCT/s_values_run6.csv"
log "S-value CSV : $REPO/4_CNNCT/s_values_run6_svalues.csv"
log "Summary log : $REPO/1_GAN/run6_svalues.log"

log "=== OVERNIGHT PIPELINE COMPLETE ==="
log "All done. Review run5 and run6 S-values, then update RESULTS.md."
