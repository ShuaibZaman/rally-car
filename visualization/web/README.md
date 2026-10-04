# Web lab

Spec §41. Not built.

The interactive lab is the pygame window:

```powershell
.\.venv\Scripts\python.exe -m visualization.dashboard --mode classical --until-lap
.\.venv\Scripts\python.exe -m visualization.dashboard --mode manual
.\.venv\Scripts\python.exe -m visualization.dashboard --mode policy --checkpoint training\checkpoints\ppo_smoke.zip
.\.venv\Scripts\python.exe -m visualization.dashboard --mode ghosts
.\.venv\Scripts\python.exe -m visualization.dashboard --mode evolution
.\.venv\Scripts\python.exe -m visualization.dashboard --mode compare
```

A later React frontend can read the same checkpoint JSON, eval JSON, and trajectory samples. It should not be the first interface. The pygame panels already show the live track, model card, reward and loss sparklines, and steering / throttle / brake.
