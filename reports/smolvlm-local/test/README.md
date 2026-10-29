# zero_margin_900: saved-checkpoint analysis

Best logged validation epoch: 5; MSE 0.091362.
Final logged validation MSE: 0.115185.

Analyzed 3213 test clips from 87 episodes.
Prediction source: `outputs/zero_margin_900/test_predictions.csv`.
Fresh inference requested: False.

The critic predicts discounted future reward, not instantaneous reward or progress.
Progress curves are recorded labels; value/progress correlation is descriptive, not progress prediction accuracy.
The recorded frame reward plot shows endpoint rewards, not every 20 Hz control reward.
Example trajectories use the median-MSE episode within each outcome, without appending synthetic terminal predictions.
Clip correlations weight longer episodes more and are not independent-sample significance tests.

Use `best.pt` for deployment/evaluation; epoch snapshots are for comparison.
Training metrics include dropout and parameter updates; validation uses evaluation mode.
Full-control and sampled-proxy runs have different labels and cannot be ranked by raw MSE.

Reevaluate all saved checkpoints on validation with the training Python environment:

```bash
python analyze_saved_run.py --run-dir outputs/zero_margin_900 --output-dir reports/smolvlm-local/test --reevaluate
```
