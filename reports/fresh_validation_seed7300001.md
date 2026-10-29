# Fresh validation: 15 episodes per formation

Collected 60 new Isaac Lab episodes with the frozen intermediate MAPPO actor as the behavior policy. Value estimates use the separate MA-VLCM `outputs/zero_margin_900/best.pt` (epoch 5, frozen SmolVLM). This was fresh VLM feature extraction and value inference on new observations, not replay of an existing prediction CSV. The MAPPO critic was not used.

Archive root: `Ground_Robot_MARL/datasets/formation_validation_seed7300001` (relative to the Postdoc workspace). Each archive has a `.summary.json` sidecar. Exact paths and outcomes are listed in `validation_inventory.csv` at that root.

| Formation | Episodes | Success | Failure | New stream seed | Value MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| triangle | 15 | 9 | 6 | 7300001 | 0.04823 |
| square | 15 | 9 | 6 | 7400001 | 0.04939 |
| pentagon | 15 | 4 | 11 | 7500001 | 0.07400 |
| hexagon | 15 | 1 | 14 | 7600001 | 0.05548 |

Overall: 2498 nonterminal clips; MSE 0.05710, MAE 0.16104, R² 0.81601, Pearson correlation 0.91179. Training-mean constant baseline MSE 0.35326. Episode-averaged MSE 0.05298.

All episodes ended in success or out_of_bounds, so none were excluded for timeout. Images are captured at 10 Hz and sampled at 5 Hz for this checkpoint; every 20 Hz control reward is retained. Return uses gamma=0.99 per control step. No outcome filtering or success quota was used. Each formation shares a fresh seeded simulator stream, not independent per-episode seeds.

Validation checks: 60 unique initial scenarios, no scenario or stream-seed overlap with the two previous collections (4,000 unique prior scenarios), complete control rewards, and timestamp/reward interval alignment. The detailed archive audit found zero errors and zero warnings for all 60 archives.

Results are for the intermediate behavior policy and the existing scene/task. They do not demonstrate generalization to unseen policies/tasks. The one successful hexagon episode is insufficient to characterize hexagon-success accuracy. The dataset is external validation once used for model selection, not a permanently untouched final test. Lower MSE than an older split does not indicate training improvement: the weights did not change and the episodes/policy mixture differ.

Outputs: `MA-VLCM/outputs/validation_seed7300001_evaluation/external_metrics.json` and `external_predictions.csv`.

The completed full-control training collection now has 2,000 episodes, 865 successes, and 14 timeouts. Timeouts are excluded from complete-return training. See `formation_10hz_current.md` for the new audit and `smolvlm-local/square_training_notes.md` for training recommendations. The 60 new episodes remain separate from training.
