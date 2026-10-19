
### event=caution: per-source AUROC / Brier (H=512, known-status still-safe test prefixes; n, positives)

| source | n | pos | direct_AO | A_horizon_query+mem+jepa | B_latent_rollout+mem+jepa | D_hierarchical+mem+jepa | A_no_memory | A_no_jepa | A_no_hazard | B_no_memory |
|---|---|---|---|---|---|---|---|---|---|---|
| aegis2 | 16 | 16 | –/0.485 | –/0.444 | –/0.509 | –/0.464 | –/0.478 | –/0.417 | –/0.540 | –/0.544 |
| beavertails | 2 | 2 | –/0.000 | –/0.000 | –/0.000 | –/0.000 | –/0.000 | –/0.000 | –/0.000 | –/0.000 |
| longwriter | 145 | 0 | –/0.003 | –/0.000 | –/0.004 | –/0.000 | –/0.000 | –/0.001 | –/0.003 | –/0.009 |
| toxic_chat | 29 | 1 | 1.00/0.019 | 1.00/0.019 | 1.00/0.024 | 1.00/0.028 | 0.96/0.024 | 1.00/0.014 | 1.00/0.020 | 1.00/0.019 |
| xstest | 9 | 8 | 0.38/0.583 | 0.00/0.807 | 0.50/0.675 | 1.00/0.718 | 0.38/0.654 | 0.00/0.824 | 0.12/0.727 | 0.25/0.756 |

### event=unsafe: per-source AUROC / Brier (H=512, known-status still-safe test prefixes; n, positives)

| source | n | pos | direct_AO | A_horizon_query+mem+jepa | B_latent_rollout+mem+jepa | D_hierarchical+mem+jepa | A_no_memory | A_no_jepa | A_no_hazard | B_no_memory |
|---|---|---|---|---|---|---|---|---|---|---|
| aegis2 | 14 | 8 | 0.85/0.388 | 0.88/0.550 | 0.85/0.524 | 0.94/0.545 | 0.88/0.540 | 0.94/0.543 | 0.79/0.508 | 0.85/0.539 |
| beavertails | 0 | 0 | – | – | – | – | – | – | – | – |
| longwriter | 131 | 2 | 0.23/0.016 | 0.11/0.015 | 0.41/0.016 | 0.13/0.015 | 0.37/0.015 | 0.12/0.015 | 0.39/0.015 | 0.55/0.016 |
| toxic_chat | 28 | 0 | –/0.000 | –/0.000 | –/0.000 | –/0.000 | –/0.000 | –/0.000 | –/0.000 | –/0.000 |
| xstest | 2 | 0 | –/0.035 | –/0.000 | –/0.000 | –/0.000 | –/0.001 | –/0.000 | –/0.006 | –/0.001 |