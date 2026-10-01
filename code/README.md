# Analysis inputs and order

These are the final scientific implementations. Filesystem roots are configurable through `CDLD_WORKSPACE` (default: repository `work/`); algorithms and numerical settings are retained.

Expected layout below that root:

```text
20260914_cdld_reviewer/
  data/questions.csv
  data/EdNet-KT1.zip
  cache/{KMLE,EdNet}/full/responses.npy
20260917_cdld_finder_5x10/
  results/{KMLE,EdNet}/full/warm_seed{42..46}/
  results/{KMLE,EdNet}/full/cold_seed{42..46}/
20260920_representation_study/
  aux_cache/
  results/
  feature_augmented/results/EdNet/full/
20260922_cdld_kmle_review_52025292/review_analysis/
```

Training notebook outputs include split.npz, predictions.npz, CDLD student/item latent arrays, and Rasch/2PL item parameter files. Place their results and cache directories in the locations above (directory symlinks are sufficient). This preserves row-index alignment.

Run `run_static_probes.py` then `analyze_static_probes.py`; `run_review.py` uses those original probe results for its reproduction check, followed by `bootstrap_review.py`. Run `prepare_ednet_aux_cache.py` before `run_rt_probes.py`. Run `run_tag_supplement.py` for tag results. `run_feature_augmented.py` uses a dedicated `REPRESENTATION_WORK_DIR` and the original EdNet notebook named in its configuration, then `analyze_feature_augmentation.py` compares its results with the base runs.

The quick public notebook uses final metric summaries and does not launch these expensive stages. The scripts were not rerun from raw data during repository preparation. Exact CDLD refits may vary across hardware and numerical environments; use the original dependency settings in the training notebooks when refitting.
