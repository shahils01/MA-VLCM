# smolvlm-local: local checkpoints, reward and progress

The screenshot's orange curve matches `outputs/zero_margin_900` (11 epochs, 28,499 training clips). The magenta curve matches `outputs/smolvlm-value` (30 epochs, 54,188 training clips). Both used local episode datasets and a frozen SmolVLM backbone, with local trainable-head checkpoints. Several runs share the W&B name `smolvlm-local`.

| Run | Target | Best epoch | Best logged validation MSE | Final training MSE | Final validation MSE |
| --- | --- | ---: | ---: | ---: | ---: |
| zero_margin_900 (orange) | Full 20 Hz discounted control rewards | 5 | 0.09136 | 0.03726 | 0.11518 |
| smolvlm-value (magenta) | Sampled-reward proxy | 6 | 0.14096 | 0.04160 | 0.18322 |

Both curves show overfitting after an early validation minimum. Use the best-validation checkpoint, not the latest epoch. These runs use different datasets, image sizes, model widths, and return definitions; their raw MSE values are not a controlled model comparison. Training metrics are accumulated with dropout and updating parameters, while validation uses evaluation mode.

For the orange run, the local checkpoint is `outputs/zero_margin_900/best.pt`; snapshots `epoch_010.pt` and `epoch_011.pt` are also present. The original run has `best.pt`, `epoch_010.pt`, `epoch_020.pt`, and `epoch_030.pt` under `outputs/smolvlm-value`.

## Best-checkpoint reward evaluation

The following metrics were recomputed from existing best-checkpoint prediction CSVs. No fresh model inference was performed in this session.

| Split | Episodes | Nonterminal clips | MSE | MAE | R² |
| --- | ---: | ---: | ---: | ---: | ---: |
| Validation | 87 | 3,238 | 0.09136 | 0.17926 | 0.70398 |
| Test | 87 | 3,213 | 0.11097 | 0.20039 | 0.66653 |

The saved training-mean baseline test MSE is 0.33278; model test MSE is about 66.7% lower. Test prediction/return Pearson correlation is 0.83290. Successful test clips have MSE 0.20139 and within-outcome R² -0.36664; failed clips have MSE 0.07307 and R² 0.22289. Aggregate performance therefore hides weaker value accuracy within successful episodes. The dataset includes initial and intermediate policies only; these measurements do not establish performance on trained or fine-tuned policies.

Validation and test targets were independently recomputed from each episode's complete 20 Hz reward stream, gamma=0.99 per control step, at the saved 5 Hz image anchors. Maximum absolute difference from CSV labels: 5.95e-8 validation, 1.09e-7 test. This verifies label alignment and return construction, not the model weights or a fresh forward pass.

## Progress

The model has one signed scalar value output: discounted future reward. It has no progress head and no immediate-reward head. Recorded team progress and frame rewards were aligned to each saved prediction and exported in the split CSVs. Value versus team-progress Pearson correlation is 0.3593 validation and 0.3922 test. These are descriptive associations, not progress prediction accuracy; do not interpret the value as a progress percentage. Future return need not rise monotonically as a successful task approaches completion.

The trajectory figures show predicted/recorded returns, recorded endpoint reward, and recorded team progress in separate panels. Examples are median-MSE episodes within each outcome. No synthetic terminal-zero predictions were appended. Endpoint reward curves do not display every control-step reward. Clip correlations weight longer episodes more and are not independent-sample significance tests.

## Artifacts and reproduction

- [Training curves matching the screenshot](training_comparison.png)
- [Validation loss and return scatter](validation/loss_and_return.png)
- [Validation reward and progress trajectories](validation/reward_progress_trajectories.png)
- [Test reward and progress trajectories](test/reward_progress_trajectories.png)
- [Validation aligned CSV](validation/validation_reward_progress.csv)
- [Test aligned CSV](test/test_reward_progress.csv)
- [Validation summary](validation/summary.json)
- [Test summary](test/summary.json)

From the MA-VLCM directory, regenerate these analyses without PyTorch:

```bash
python3 analyze_saved_run.py --output-dir reports/smolvlm-local/validation
python3 analyze_saved_run.py --split test --output-dir reports/smolvlm-local/test
```

To run fresh inference for each saved checkpoint on validation, activate the training environment with PyTorch and run:

```bash
python analyze_saved_run.py --run-dir outputs/zero_margin_900 --output-dir reports/smolvlm-local/fresh --reevaluate
```

This writes per-checkpoint validation predictions and `checkpoint_metrics.json`. It uses the checkpoint's saved episode split and feature cache, and evaluates nonterminal anchors consistently. Use validation to compare checkpoints; retain test for the selected checkpoint.

Fresh inference could not run in this session: `python` resolves to Python 2, `python3` has no PyTorch, the listed VLCM conda environment has no Python executable, and `nvidia-smi` cannot communicate with the driver. Existing predictions and local archives remain readable. No training, checkpoint modification, or W&B write was performed.
