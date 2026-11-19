# Motion World Model for Latency-Compensated MR-Linac Beam Gating

Code for *"Forecasting Tumor Motion Ahead of the Beam: A Label-Free Continuous-Time Motion World Model
with Conformal Gating"* (AAAI-27 Workshop on Medical World Models submission).

On an MR-Linac, the gating decision acts on anatomy a few hundred milliseconds after the image it is
based on. We forecast the motion over this latency and gate the beam on the forecast:

- **Label-free state.** DIS optical-flow registration fields of cine MRI; no annotations needed, so
  unlabeled scans become training data.
- **Continuous-time world model.** A space-time transformer forecasts the change of the field at any
  horizon from irregularly timed frames; five independently initialized members form an ensemble.
- **Conformal gating.** The ensemble spread scales a split-conformal margin around the forecast
  target; the beam is enabled only if the dilated forecast stays within the treatment boundary.

    ├── common.py  data.py  flows.py  metrics.py
    ├── motion_wm.py          # model + training
    ├── evaluate_motion.py    # baselines, forecasts, gating quantities
    ├── results.py            # statistics, conformal calibration, gating
    ├── splits.json  mwm_full_scans.json
    └── models.py  train_head.py  evaluate.py  analyze.py   # DINOv3 baseline
