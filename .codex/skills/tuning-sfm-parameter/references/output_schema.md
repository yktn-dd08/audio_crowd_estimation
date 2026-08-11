# Output schema

`sfm_search_result.json` は次の構造を使う。

```json
{
  "schema_version": "1.2",
  "base_config": "path/to/base.json",
  "search": {
    "target_mean_speed": [0.8, 1.0],
    "person_num": [100, 500],
    "top_k": 1,
    "candidates_per_condition": 48,
    "rounds": 4,
    "refine_centers": 3,
    "refine_candidates_per_center": 12,
    "expansion_batch": 48,
    "max_candidates_per_condition": 384,
    "exact_k_required": true,
    "fallback_policy": "strict-first-then-constraint-violating-evaluable",
    "render_top": 3,
    "collision_distance_m": 0.6
  },
  "scoring": {
    "target_speed": 0.25,
    "vector_acceleration": 0.20,
    "oscillation": 0.25,
    "tortuosity": 0.05,
    "collision": 0.20,
    "output_person_ratio": 0.05
  },
  "results": [
    {
      "target_mean_speed": 0.8,
      "person_num": 100,
      "evaluated": 12,
      "rejected": 2,
      "strict_recommendations": 1,
      "fallback_recommendations": 0,
      "parameter_sets": [
        {
          "rank": 1,
          "c_obs": 450.0,
          "r_obs": 0.425,
          "c_wall": 800.0,
          "r_wall": 0.85,
          "desired_speed": 1.0,
          "score": 0.91,
          "constraint_status": "strict",
          "metrics": {
            "mean_speed": 0.82,
            "p95_speed": 1.31,
            "max_speed": 1.76,
            "mean_acc": 0.19,
            "p95_acc": 0.52,
            "mean_speed_change_acc": 0.07,
            "jitter_mean_deg": 6.2,
            "jitter_large_turn_rate": 0.01,
            "reversal_rate": 0.002,
            "severe_reversal_rate": 0.0,
            "two_step_backtrack_rate": 0.0,
            "oscillating_agent_ratio": 0.0,
            "path_tortuosity_mean": 1.08,
            "target_speed_error": 0.02,
            "collision_count": 0,
            "collision_pair_rate": 0.0,
            "collision_frame_rate": 0.0,
            "output_person_num": 100,
            "output_person_ratio": 1.0
          },
          "flags": []
        }
      ]
    }
  ]
}
```

## Rules

- `schema_version` is always `1.2`.
- `results` contains one item per target speed / person count pair.
- Final non-dry-run output has exactly K items in every `parameter_sets`; K未満の結果を成功として保存しない。
- `constraint_status` is `strict` or `fallback`. Place all strict candidates before fallback candidates, then sort by descending `score` within each tier.
- Use fallback only for candidates with valid, finite trajectory metrics that violate behavioral hard constraints. Never include simulation failures, missing/empty CSV, NaN/inf, or evaluation failures.
- `strict_recommendations + fallback_recommendations == top_k` for every final result.
- Do not add long natural-language explanations inside each candidate.
- Optional diagnostics may be added to `metrics` if they are computed deterministically.
- A strict-rejected but evaluable candidate may appear only as an explicitly flagged fallback after adaptive exploration is exhausted.
