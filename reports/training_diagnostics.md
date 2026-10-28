# Formation critic training comparison

This is one controlled comparison on the existing 2,000-episode, 5 Hz dataset.
Both models use the same episode train/validation/test splits, sampled-reward
proxy targets and frozen SmolVLM features. The test comparison excludes terminal
anchors for both models. It is not a measurement of full 20 Hz policy return.

The original run's best checkpoint was epoch 6; its logged validation MSE,
including terminal frames, was 0.1410. Training MSE continued from 0.1325 at
epoch 1 to 0.0416 at epoch 30 while logged validation MSE rose to 0.1832. This
is consistent with overfitting. Direct train/validation numbers are not measured
under identical dropout conditions because training metrics are collected while
parameters are updating.

The second run used width 128, one GAT layer, one temporal layer, dropout 0.2,
learning rate 0.0001, batch size 64, learning-rate reduction on validation
plateau, terminal-anchor exclusion, and six-epoch early stopping. It selected
epoch 4 using validation and stopped after epoch 10. These changes were tested
together, so this comparison cannot isolate which one caused the improvement.

| Nonterminal clip metric | Original best | Regularized best | Constant baseline |
| --- | ---: | ---: | ---: |
| Validation MSE | 0.13239 | 0.11158 | 0.2801 |
| Validation MAE | 0.24086 | 0.21541 | — |
| Test MSE | 0.17570 | 0.14864 | 0.2703 |
| Test MAE | 0.27707 | 0.24897 | — |
| Test R² | 0.3499 | 0.4500 | — |
| Test episode-averaged MSE | 0.16922 | 0.14179 | — |

Each split has 200 episodes; the comparison uses 6,526 validation clips and
6,943 test clips. A paired bootstrap over test episodes gave a 95% interval of
0.0074–0.0468 for the MSE reduction; this treats episodes as independent and
does not account for shared collection streams. Four-robot test MSE changed
from 0.2496 to 0.2516, and fine-tuned-policy-stage test MSE changed from 0.2790
to 0.2905. Improvements are therefore not uniform. Repeated seeds and genuinely
unseen environments/policies are still needed before making broader claims.

In the original test predictions, terminal anchors alone had MSE 0.401 even
though their future return is known to be zero. The critic did not receive the
episode-done flag. The revised trainer excludes terminal anchors from learned
regression and the single-episode prediction command returns zero when the
supplied frame is explicitly known to be terminal.

The separate 10 Hz `formation_vlcm_polygon_3to6_10hz_inner00m` collection has
903 episodes, including 901 complete episodes with aligned 20 Hz control reward
streams. Its image dimensions are 256×256 versus 128×128 in the original 5 Hz
collection. It contains only initial and intermediate policies and has a
different workspace boundary, so it does not replace the old data's coverage.
On a random sample of 30 archives from each collection, PNG data represented
approximately 61% of old archive bytes and 85% of the newer archive bytes;
`control_rewards.json` accounted for about 0.05% of newer archive bytes. Saving
fewer/lower-resolution images while retaining all control rewards is therefore
a practical way to reduce size without discarding control-step reward labels.

The full-control and sampled-proxy returns on the 901 complete new episodes
have a mean absolute difference of 0.157 across the selected nonterminal clips.
Their mean full return is -0.430 versus -0.272 for the sampled proxy. This
comparison uses each episode's actual 20 Hz team reward sequence; it does not
estimate the original dataset's missing rewards.
