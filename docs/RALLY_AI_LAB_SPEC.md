# Autonomous Rally AI Lab — canonical brief

This file is the reference for every child plan under `docs/plans/`. Section numbers match the original brief. Later plans cite these sections instead of restating them.

## 1. The core idea

Frame the project as:

**Autonomous Rally AI Lab**

A 2D rally car is placed on procedurally generated tracks. An AI agent controls the car and attempts to complete laps as quickly and safely as possible.

Expose the underlying ML system as an experiment:

```
                    AUTONOMOUS RALLY AI
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
   Environment        Neural Network      Training
        │                  │                  │
  Track generation      MLP/CNN/RNN      PPO/SAC/TD3
  Car physics           2/3/4 layers     Loss functions
  Sensors               32/64/128        Optimizers
  Rewards               Residual         Learning rates
        │                  │                  │
        └──────────────────┼──────────────────┘
                           ↓
                   Driving behavior
                           ↓
             Completion / lap time / crashes
```

The showcase is not merely "I trained an RL agent to drive."

It is: "I built a controlled environment for studying how reinforcement-learning algorithms, objectives, and neural-network architectures affect autonomous driving behavior."

## 2. What the game should look like

Use a top-down 2D rally game rather than a 3D racing simulator. A top-down view makes the AI behavior easier to understand and keeps time on the ML instead of a large game engine.

Visual target: a clean rally track with the car, track boundaries, trajectory, and sensor information visible. Expose raycast sensors, checkpoint progress, reward, speed, and steering.

Prefer less game-like clutter and more emphasis on the AI. The lab layout:

```
┌───────────────────────────────────────────────────────────────┐
│                   AUTONOMOUS RALLY AI                        │
├──────────────────────────────────┬────────────────────────────┤
│                                  │ MODEL                       │
│              track + car         │ PPO                         │
│                                  │ MLP 3 × 128                │
│                                  │ Parameters                  │
│                                  │ EPISODE                    │
│                                  │ REWARD                     │
│    raycast sensors               │ LAP TIME                   │
│                                  │ PROGRESS                   │
├──────────────────────────────────┼────────────────────────────┤
│ TRAINING                         │ ACTION                      │
│ Reward sparkline                 │ Steering                    │
│ Loss sparkline                   │ Throttle                    │
│                                  │ Brake                       │
└──────────────────────────────────┴────────────────────────────┘
```

## 3. The car itself

Physics does not need to reproduce a real rally car. Avoid extreme realism initially.

State:

- Position: `x`, `y`
- Orientation: `θ`
- Velocity: `vx`, `vy`
- Angular velocity: `ω`

Controls: steering, throttle, brake.

Steering must not instantly point the car in the desired direction. The car needs momentum. That creates the learning problem: a poor agent reacts inside the corner and crashes; a learning agent turns in; a very good agent positions itself before the corner.

## 4. Continuous controls

Use continuous actions.

- steering ∈ [-1, 1] — -1 full left, 0 straight, +1 full right
- throttle ∈ [0, 1] — 0 no acceleration, 1 maximum acceleration
- brake ∈ [0, 1]

Keep throttle and brake separate so behavior stays interpretable. Do not fold brake into negative throttle.

## 5. What the AI sees — version A, sensor-based driving

The network receives numerical observations. Use 11, 15, or 21 raycasts. Each ray tells the AI how far the track boundary is, as a normalized distance.

Additional state:

- speed
- angular velocity
- heading error
- distance from center
- track curvature
- current progress
- previous steering
- previous throttle

Target about 20–40 numerical features. This is the MLP input.

## 6. Version B, vision-based driving

Later, replace distances with an image, for example 84 × 84 × 3, and let a CNN learn the track. Pipeline: image → CNN → MLP → steering, throttle, brake.

Architectural comparison:

- Engineered sensors → MLP → actions
- Pixels → CNN → MLP → actions

The sensor version should learn faster because useful information is given directly. The vision version is harder because the network must learn the visual representation.

## 7. Procedural track generation

Procedural tracks prevent "the model memorized one track."

Start from control points, interpolate a smooth closed curve. Parameters:

- track width
- number of turns
- maximum curvature
- straight length
- corner sharpness
- random seed

Difficulties: easy, medium, hard, extreme (hairpins, S-curves, varying radii, short straights).

Every track has a seed and a metadata card, for example:

- Track id
- Seed
- Difficulty
- Turns
- Length

That makes experiments reproducible.

## 8. Separate training and testing tracks

Train on many tracks. Evaluate on completely unseen tracks. Otherwise it is unclear whether the model learned to drive or learned specific tracks.

Dataset target:

- Training: 1,000 procedurally generated tracks
- Validation: 100 tracks
- Test: 100 tracks

Test tracks are never used during training. Report completion on training tracks and on unseen tracks as a generalization experiment.

## 9. What the car's objective means

The reward is not "+1 for every checkpoint." Use several components:

```
Reward =
    progress reward
  - off-track penalty
  - collision penalty
  - excessive steering penalty
  - excessive braking penalty
  - time penalty
  + lap completion bonus
```

Conceptually: make progress, don't crash, don't waste time, stay on the track, finish the lap.

Reward shaping changes behavior. That is part of the point of the lab.

## 10. Reward-function experiments

Explicit variants:

- Reward A, progress only: `R = Δprogress`. The car may drive recklessly.
- Reward B, progress + safety: `R = Δprogress - collision_penalty`. More conservative.
- Reward C, racing: progress − collision − time − off-track. Maximize speed while staying alive.
- Reward D, racing line: add a term toward a desirable trajectory.

Compare wide corners, cut corners, and an efficient racing line.

## 11. Reward functions are not loss functions

```
Environment → Reward → RL algorithm → Training objective / losses → Neural network
```

Separate experiment categories:

1. Reward/objective: progress-only, progress + safety, progress + speed, progress + speed + racing line
2. RL algorithm: PPO, SAC, TD3
3. Architecture: 2-layer MLP, 3-layer MLP, 4-layer MLP, residual MLP, CNN, CNN + LSTM
4. Optimization: Adam, AdamW, learning rates, learning-rate schedules

## 12. PPO is the starting point

PPO is the baseline, not because it is necessarily best, but because it is a good first algorithm. Then SAC, then TD3.

Question: given the same environment, reward, network, and training budget, which algorithm produces the best driver?

Measure more than final reward:

- lap completion
- lap time
- collision count
- off-track time
- average speed
- distance traveled
- fuel consumption
- training stability
- generalization to unseen tracks

## 13. Model architecture experiments

Architectures:

- MLP-2
- MLP-3
- MLP-4
- Wide MLP
- Residual MLP
- CNN
- CNN + LSTM

Question: does a deeper network actually produce a better driver? The answer may be no.

## 14. Control parameter count

Two comparison types:

- Depth comparison: hold approximate parameter count constant. Vary 2, 4, and 6 layers and shrink width.
- Capacity comparison: intentionally increase model size (about 10k, 50k, 100k, 500k, 1M parameters) and plot size vs performance.

A larger model winning does not, by itself, say anything about depth.

## 15. Memory

Compare MLP against MLP + LSTM. Question: does temporal memory help the car anticipate turns and recover from ambiguous situations?

## 16. Curriculum

Do not start on the hardest track.

1. Straight — throttle and basic steering
2. Gentle curves — turning
3. Sharp corners — braking and corner entry
4. S-curves — rapid direction changes
5. Hairpins — aggressive braking and acceleration
6. Random tracks — generalization
7. Disturbances — random starting speed, random starting position, temporary loss of grip, lateral force, uneven surfaces. The car must learn recovery.

## 17. Surfaces

Rally surfaces, not just a racing car. Physics changes; graphics can stay simple.

| surface | grip | acceleration |
| --- | --- | --- |
| asphalt | high | high |
| gravel | medium | medium |
| mud | low | low |
| snow | very low | low |
| dirt | medium-low | medium |

A corner that is safe at high speed on asphalt may require a much lower speed on gravel. The agent learns that from experience.

## 18. Weather

Later: dry, rain, snow, altering physics.

Compare:

- Model A trained only on dry tracks
- Model B trained on dry + wet
- Model C trained with domain randomization

Test all three on wet tracks. This is a robustness and generalization question.

## 19. Trajectory history

Whenever the car drives, save `(x, y, t)` and draw the trajectory. Overlay several models (PPO, SAC, TD3, expert) so policy differences are visible.

## 20. Ghost cars

Give every model the same starting conditions and run them simultaneously. Show lap times next to the cars. The comparison should be understandable without reading a paper.

## 21. Make the car visibly think

Draw sensor rays. Distance can change color or thickness. Show speed, heading error, track distance, curvature, and the outputs steering, throttle, and brake. The chain is input → network → action → behavior.

## 22. Reward decomposition

Do not show only `Reward = 4.7`. Show the step breakdown: progress, speed, centering, off-track, steering, collision, and the total. A crash should make the collision term obvious.

## 23. Training improvement over time

A timeline of checkpoints (0, 10k, 25k, 50k, 100k episodes or the closest saved steps). Loading an earlier checkpoint shows earlier behavior. Loading a later one shows a more trained policy. Do not fake intermediate skill if those checkpoints were not trained.

## 24. Evolution of the driver

A screen with six replay windows: Random, Beginner, Competent, Advanced, Racing line, Unseen. This is a landing view for how the policy changes. Empty slots stay empty until a checkpoint exists for that stage.

## 25. Metrics

Primary:

- Lap completion rate (completed laps / attempted laps)
- Lap time (lower is better)
- Progress (how far along the track before termination)
- Collision rate

Secondary:

- Average speed
- Off-track percentage
- Centerline deviation
- Steering smoothness
- Throttle efficiency
- Number of recoveries
- Training steps to reach a target performance
- Generalization score on unseen tracks

## 26. A composite score, plus the components

```
Rally score = completion + speed + smoothness - crashes - off-track time
```

Always show the individual metrics too. One number can be gamed, and which model is "better" depends on the objective. That tradeoff is itself an ML question.

## 27. Reward hacking

Progress-only reward can produce non-human strategies: leave the track, cut, still collect progress. Keep that variant as an explicit demonstration of how reward design changes emergent behavior. It is not the default racing objective.

## 28. Obstacles

After basic driving works, add static obstacles. The agent must drive, avoid the obstacle, remain on track, and maintain speed. Randomize placement. Measure obstacle-avoidance success.

## 29. Moving obstacles

Later, an opponent that changes lanes or speed. Do this only after the basic problem works.

## 30. Classical controller baseline

Before treating the neural policy as the only driver, build a traditional controller:

track centerline → pure pursuit / steering → throttle controller → car.

Compare the classical controller with the RL agent. The classical controller may win on simple tracks. The learned policy has room to matter on unknown tracks, disturbances, low grip, obstacles, and partial observation.

## 31. Imitation learning

Use the classical controller as an expert. Generate `observation → expert action` pairs. Train with supervised learning (behavior cloning).

Compare losses MSE, MAE, and Huber, and optimizers Adam, AdamW, and SGD.

## 32. From imitation to reinforcement learning

Pipeline: expert controller → demonstrations → behavior cloning → initial neural policy → reinforcement learning → improved driver.

The model first learns what competent driving looks like, then RL can try to drive better. Two learning paradigms in one project.

## 33. Experimental matrix

The space is large: imitation (MSE / MAE / Huber, Adam / AdamW / SGD) and RL (PPO / SAC / TD3, different objectives), crossed with architecture (MLP, ResNet, LSTM; shallow, deep, memory), scored by lap time, success, and generalization.

## 34. Do not train 50 models immediately

Staged experiments, in order:

1. PPO + a modest MLP. Get the car working.
2. PPO + shallow vs deep MLP. Architecture.
3. PPO + different reward functions. Reward design.
4. PPO vs SAC vs TD3. Algorithms.
5. State input vs pixel input. Representations.
6. Train vs unseen tracks. Generalization.

Implement the harness for these. Do not launch the full matrix in the first build.

## 35. Dashboard

Four sections:

- A. Live simulation: track, car, sensor rays, trajectory
- B. Model panel: algorithm, architecture, hidden size, parameter count, observation mode, learning rate, entropy coefficient
- C. Training panel: episode reward and success-rate curves
- D. Experiment comparison: completion, lap time, collisions, off-track, parameters across PPO, SAC, and TD3

## 36. Learning curves and driving behavior together

Pair a training checkpoint with the behavior it produces. A reward number without the corresponding driving trace is not enough. Use saved checkpoints and their eval traces. Do not invent behavior for checkpoints that were not run.

## 37. Checkpoint saving

Save models periodically. Also save track seed, environment config, model config, training config, and metrics. The dashboard can load a checkpoint and replay it.

## 38. Reproducible experiments

Every experiment has an id of the form `PPO_MLP3_REWARD2_SEED42`, plus algorithm, architecture, observation, reward, optimizer, learning rate, training seed, and environment seed.

For serious comparisons, train each configuration with multiple random seeds and report average and variability. Implement aggregation. Do not run the multi-seed study until asked.

## 39. Project architecture

```
environment/     car, physics, track, generator, sensors, rewards
agents/          policies (mlp, residual_mlp, cnn, cnn_lstm), ppo, sac, td3
training/        train.py, evaluate.py, configs/, checkpoints/
experiments/     architecture, rewards, algorithms, generalization
visualization/   renderer, dashboard, plots
data/            trajectories, results
```

## 40. Technology stack

- Environment: Python + Pygame
- ML: PyTorch
- RL: Stable-Baselines3 first, so the environment and experiments come before a from-scratch PPO. A custom PPO module comes second, for loss-level experiments.
- Dashboard: Pygame + Matplotlib first. A React/TypeScript frontend is a later polished version, not the starting point.

On this machine: Python 3.13 virtualenv, CUDA torch from the PyTorch index, interpreter `.\.venv\Scripts\python.exe`. Do not create the venv with system Python 3.14.

## 41. What a final website could look like

A later portfolio front page: watch a network learn to drive, then open training progress, compare models, compare architectures, explore reward functions, and test on a new track. Do not build this first.

## 42. Compare models mode

Choose PPO, SAC, TD3. Same track and start. Show cars and lap times, plus training stability, generalization, collision rate, and average speed when those eval files exist.

## 43. Compare architectures mode

Same algorithm, reward, and tracks. Different architecture. Show completion bars and a trajectory for the selected model when a result file exists.

## 44. What did the AI learn

Qualitative notes derived from a recorded trace, for example: does not brake before corners; brakes but exits slowly; brakes before the corner, turns smoothly, accelerates at the exit. Generate the notes from the trace. Do not hard-code a story the trace does not support.

## 45. Visualize the learned policy

For sensor models, vary speed and curvature while holding other inputs fixed. Plot the resulting brake (and steering) so high speed and high curvature can be inspected directly.

## 46. Heat maps

Bin crashes and off-track events along the centerline. Show where the policy fails, and compare two checkpoints when both exist.

## 47. Generalization, visually

Train-pool performance vs an unseen test track. Report both numbers. Curriculum or domain randomization can be compared later, when those runs exist.

## 48. Research questions

- Architecture: how does network depth affect sample efficiency and driving performance under fixed model capacity?
- RL algorithm: how do PPO, SAC, and TD3 differ in learning speed, stability, and final driving performance in continuous-control racing?
- Reward design: how does reward shaping influence the tradeoff between lap completion, speed, and safety?
- Generalization: does training on procedurally generated tracks produce a policy that generalizes to unseen track geometries?
- Representation: how does learning from raw pixels compare with learning from engineered sensor observations?
- Curriculum: does progressive difficulty reduce the number of interactions required to reach competent driving?
- Memory: does a recurrent architecture improve driving under partial observability?

The first three the lab is set up to answer, before any full sweep is launched:

1. Reward shaping
2. Architecture at fixed capacity
3. Unseen-track generalization

## 49. A compelling final experiment

"Can a small neural network learn to race?" Start from an untrained 2-layer MLP on raycasts, then show later checkpoints on a training track and on an unseen track. Only show a checkpoint that was actually trained. The claim "it learned a driving policy" waits on a real training run, which is out of scope until explicitly requested.

## 50. Build order

1. Car simulator: physics, track, collision, lap checkpoints, camera. Manual driving should already be possible.
2. Sensors: 11–15 raycasts, speed, heading, track position, rendered rays, then a rule-based driver.
3. PPO baseline on a simple track. Do not compare models yet.
4. Procedural generation: easy, medium, hard, then train on many tracks (full training only when asked).
5. Training visualization: reward, completion, lap time, trajectory, rays, actions.
6. Experiment harness: algorithms, depth, rewards, optimizers, state vs vision. Configs first; full runs later.
7. Polished presentation: checkpoints, ghost cars, comparison, unseen tracks, trajectory overlays, leaderboard, interactive dashboard.

## 51. MVP

2D top-down car, procedural tracks, 15 raycast sensors, speed and heading, PPO, more than one MLP width available as configs, 3+ reward functions, training replay via checkpoints, lap time / completion / collision metrics, unseen test tracks.

The second version, after the MVP is real: SAC / TD3 runs, CNN vision, LSTM, curriculum, surfaces, obstacles.

## 52. The driving objective

Winning is not the only goal. The policy optimizes a measurable objective with a tradeoff among complete the lap, go fast, and don't crash. Always report the components behind a single score.

## 53. Preferred final picture

```
                    RALLY AI LAB
                         │
       ┌─────────────────┼──────────────────┐
   PROCEDURAL         SENSOR              VISION
    TRACKS            INPUTS               INPUTS
       └─────────────────┼──────────────────┘
                         ↓
              POLICY NETWORK
              MLP / ResNet / LSTM
                         ↓
                  PPO / SAC / TD3
                         ↓
                     CAR
                         ↓
              COMPLETE / FAST / SAFE
                         ↓
                  TRAINING HISTORY
                         ↓
         TRAJECTORY / METRICS / MODEL COMPARE
```
