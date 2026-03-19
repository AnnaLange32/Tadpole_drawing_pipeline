# Nao robot - Attention mode
---
`nao_attention_capture.py` runs on your local computer and connects to the
robot remotely over TCP.

## Different Attention Modes

| Mode | Behaviour | LED colour (Eyes)|
|---|---|---|
| `face` | Turns toward detected faces | Red |
| `motion` | Turns toward movement clusters | Blue |
| `sound` | Turns toward sound source (azimuth/elevation) | Green |
| `idle` | Random head movements at a fixed interval | Dim white |
| `all` | All three simultaneously; head priority: face > motion > sound | Winning source |
| `all_nosound` | Face and motion only | Winning source |

## Output
**Detection CSV columns:**

| Column | Description |
|---|---|
| `wall_time` / `elapsed_s` | Absolute and session-relative timestamp |
| `source` | `face`, `motion`, or `sound` |
| `detected` | `yes` / `no` |
| `confidence` | Detection confidence (0–1) |
| `bbox_x/y/w/h` | Bounding box in pixels |
| `head_yaw` / `head_pitch` | Head joint angles at log time (radians) |
| `azimuth` / `elevation` | Angular position of detection source (radians) |
| `energy (sound)` | Acoustic energy proxy (sound mode only) |
| `head_driver` | Which source drove the head (`all` mode only) |

#### CLI Parameters

| Argument | Description | Default |
|---|---|---|
| `--output_folder` | Extraction pipeline output directory | *(required)* |
| `--csv` | CSV ratings file for good visible parts| *(required)* |
| `--csv_all` | CSV ratings  file for all frames (used for frequency calculation)| *(required)* |
| `--count` | Number of figures to generate | `20` |
| `--size` | Canvas size in pixels (square) | `512` |
| `--overlap` | Joint overlap as fraction of part height | `0.06` |
| `--seed` | Random seed for reproducibility | `None` |
