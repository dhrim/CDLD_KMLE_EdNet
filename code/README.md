# KMLE analysis inputs and order

Set `CDLD_WORKSPACE` to the root containing these retained training artifacts:

```text
20260914_cdld_reviewer/cache/KMLE/full/responses.npy
20260917_cdld_finder_5x10/results/KMLE/full/warm_seed{42..46}/
20260917_cdld_finder_5x10/results/KMLE/full/cold_seed{42..46}/
20260920_representation_study/results/
20260922_cdld_kmle_review_52025292/review_analysis/
```

The training notebook writes `split.npz`, `predictions.npz`, CDLD latent arrays and Rasch/2PL parameter files. Place the generated results and cache in the locations above, preserving row-index alignment. Historical folder names are retained by the analysis implementations.

Run `run_static_probes.py`, then `analyze_static_probes.py`. Next run `run_review.py` (which checks the original IRT probe results before the exploratory comparisons), followed by `bootstrap_review.py`. These scripts now process KMLE only. No model was retrained when preparing this revision.

The quick root notebook reconstructs final tables and figures from compact saved summaries. It does not run these expensive stages. Exact refits may vary with hardware and numerical environment; the original dependency versions are documented in the training notebook.
