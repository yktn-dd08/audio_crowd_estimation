# Current simulator contract

Use this reference when generating configs or interpreting SFM CSV output.

## Config precedence

`simulation.crowd_trajectory` resolves SFM values in this order:

1. task-level value: `task_list.<task>.<key>`
2. common value: `param.<key>`
3. built-in default

For search-controlled keys, write the value to both task and common levels so a stale task override cannot shadow the requested condition.

Search-controlled keys:

- `person_num`
- `desired_speed`
- `c_obs`
- `r_obs`
- `c_wall`
- `r_wall`

## Simulation invocation

Use:

```bash
python -m simulation.crowd_trajectory -c <config.json>
```

The config must use `option: social_force`.

## Video generation

The simulator generates a video whenever task-level `output_mp4` exists. Therefore remove `output_mp4` during coarse/refine search. Only rerun selected top candidates with `output_mp4` restored.

## CSV schema

`SocialForceSimulation.to_csv()` writes one row per agent trajectory, not one row per timestep.

Expected columns:

- `id`: person ID
- `start_time`: timestamp of the first recorded trajectory point
- `geom`: WKT `LINESTRING` containing successive positions
- `goal`: optional WKT point

Successive `LINESTRING` coordinates are separated by the configured simulation `dt`. Reconstruct speed as segment length divided by `dt`.

## Collision proxy

`SocialForcePerson.radius` defaults to `0.3 m`. Use center distance `< 0.6 m` as the standard near-collision/overlap threshold unless the simulator radius is changed.

Synchronize agents using `start_time` plus coordinate index times `dt` before counting collisions.

## Important output limitation

The CSV does not preserve `finish_reason`, wall projection count, stuck events, or wall-oscillation events. Therefore those events cannot be exactly reconstructed from the saved CSV alone.

Use trajectory metrics and output-person ratio for batch ranking. Use top-candidate video inspection for wall/stuck behavior. If exact counts become necessary, modify the simulator to export event counters rather than asking the LLM to infer them from video.
