# cathode_run0_pipeline.ps1
# ==========================
# Overnight training + evaluation chain for cathode-run0.
#
# Run from repo root:
#   .\cathode_run0_pipeline.ps1
#
# All output is appended to 1_GAN/cathode_run0_output.log
# Grep tag for this run: [cathode-run0]
#
# Step budget: 24 crops / batch=4 = 6 batches/epoch; 216 epochs = 1296 G-steps
# (~anode run5 reference: 26 batches * 50 epochs = 1300 G-steps).
# Checkpoints saved at epochs 54 / 108 / 162 / 216 (25/50/75/100%).

$ErrorActionPreference = "Continue"
$LOG = "1_GAN\cathode_run0_output.log"

function Log {
    param([string]$msg)
    $ts   = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[cathode-run0][$ts] $msg"
    Write-Host $line
    Add-Content -Path $LOG -Value $line -Encoding UTF8
}

# ── Config ────────────────────────────────────────────────────────────────────
$TRAIN_DATA   = "cathode_crops_s1_str32"   # 24 training crops (stride-32)
$REF_DATA     = "cathode_crops_s1_str64"   # 6 reference crops (stride-64, non-overlapping)
$EPOCHS       = 216
$SAVE_EVERY   = 54
$CKPTS        = @(54, 108, 162)            # intermediate checkpoints
$FINAL_EP     = 216

Log "==================================================================="
Log "  cathode-run0: WGAN-GP on S1 Supercrop (LSCF/GDC/Pore, 64^3)"
Log "  G_loss = WGAN + 1000*vf + 50*conn(pore)  [no tau, no tpb-proxy, no ysz-density]"
Log "  $EPOCHS epochs, save every $SAVE_EVERY, ~1296 G-steps"
Log "==================================================================="

# ── PHASE 1: Train ────────────────────────────────────────────────────────────
Log "PHASE 1: Training ($EPOCHS epochs) ..."
conda run -n ganph --no-capture-output python -u 1_GAN/main.py `
    --data        $TRAIN_DATA `
    --epochs      $EPOCHS `
    --lr          0.00005 `
    --save-every  $SAVE_EVERY `
    --no-tpb-proxy `
    --no-ysz-density 2>&1 | Tee-Object -FilePath $LOG -Append
Log "PHASE 1 done."

# ── PHASE 2: Generate from intermediate checkpoints (20 each, no-tau for speed)
foreach ($ep in $CKPTS) {
    $OUT = "generated_cathode_run0_ep${ep}"
    Log "PHASE 2: Generate 20 structures @ epoch $ep -> $OUT ..."
    conda run -n ganph --no-capture-output python -u 4_CNNCT/generate_structures.py `
        --training-data $TRAIN_DATA `
        --epoch $ep `
        --output $OUT `
        --n 20 2>&1 | Tee-Object -FilePath $LOG -Append
    Log "PHASE 2 ep${ep}: S-value analysis (no-tau) ..."
    conda run -n ganph --no-capture-output python -u 4_CNNCT/analyze.py `
        --dataset-config cathode_s1_supercrop `
        --input   $REF_DATA `
        --compare $OUT `
        --output  "4_CNNCT\s_values_cathode_run0_ep${ep}.csv" `
        --no-tau 2>&1 | Tee-Object -FilePath $LOG -Append
}

# ── PHASE 3: Generate 50 from final checkpoint ────────────────────────────────
$OUT_FINAL = "generated_cathode_run0_final"
Log "PHASE 3: Generate 50 structures @ epoch $FINAL_EP -> $OUT_FINAL ..."
conda run -n ganph --no-capture-output python -u 4_CNNCT/generate_structures.py `
    --training-data $TRAIN_DATA `
    --epoch $FINAL_EP `
    --output $OUT_FINAL `
    --n 50 2>&1 | Tee-Object -FilePath $LOG -Append

# ── PHASE 4: Full S-value analysis of final (with tau — informational) ────────
Log "PHASE 4: Full S-value analysis (final, with tau) ..."
conda run -n ganph --no-capture-output python -u 4_CNNCT/analyze.py `
    --dataset-config cathode_s1_supercrop `
    --input   $REF_DATA `
    --compare $OUT_FINAL `
    --output  "4_CNNCT\s_values_cathode_run0_final.csv" 2>&1 | Tee-Object -FilePath $LOG -Append

# ── PHASE 5: Memorization check ───────────────────────────────────────────────
Log "PHASE 5: Memorization check (generated-final vs training crops) ..."
conda run -n ganph --no-capture-output python -u 4_CNNCT/memo_check.py `
    --train     $TRAIN_DATA `
    --generated $OUT_FINAL `
    --output    "4_CNNCT\cathode_run0_memo.csv" 2>&1 | Tee-Object -FilePath $LOG -Append

Log "==================================================================="
Log "ALL DONE. Results in:"
Log "  4_CNNCT/s_values_cathode_run0_ep{54,108,162}.csv"
Log "  4_CNNCT/s_values_cathode_run0_final.csv"
Log "  4_CNNCT/cathode_run0_memo.csv"
Log "  -> Fill in 1_GAN/CATHODE_RUN0_REPORT.md"
Log "==================================================================="
