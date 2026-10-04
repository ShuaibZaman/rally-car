"""Four-panel pygame lab.

Modes: manual, classical, policy, ghosts, evolution, compare.
Missing checkpoints stay empty. Nothing on screen is invented from a run that
did not happen.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pygame

from agents.classical import PurePursuit
from environment.env import RallyEnv
from environment.generator import EASY_FIXTURE_SEED, TEST_FIXTURE_SEED
from environment.types import Action, EnvConfig, VehicleState
from training.metrics import corner_markers, crash_histogram, describe_trace
from visualization.leaderboard import load_board, record_lap, reference_lap
from visualization.plots import save_comparison_bars, save_policy_slice
from visualization.renderer import (
    BG,
    BRAKE,
    CLASSICAL,
    GOLD,
    LEARNED,
    MUTED,
    RANDOM,
    TEXT,
    THROTTLE,
    WorldView,
    draw_car,
    draw_channels,
    draw_heatmap_band,
    draw_markers,
    draw_polyline,
    draw_rays,
    draw_results_table,
    draw_sparkline,
    draw_speed_line,
    draw_track,
    mono,
    panel,
)

WIDTH = 1280
HEIGHT = 800
MODES = ("manual", "classical", "policy", "ghosts", "evolution", "compare")
LEGEND = "1-6 mode   F follow   L rays   P pause   arrows scrub   [ ] checkpoint   E export"
EVOLUTION = (
    ("Random", None, False),
    ("Early", "early.zip", False),
    ("Mid", "mid.zip", False),
    ("Late", "late.zip", False),
    ("Racing line", None, False),
    ("Unseen", "late.zip", True),
)


def action_from_flags(left: bool, right: bool, throttle: bool, brake: bool) -> Action:
    steering = float(right) - float(left)
    return Action(steering, 1.0 if throttle else 0.0, 1.0 if brake else 0.0)


@dataclass
class Ghost:
    name: str
    color: tuple[int, int, int]
    trace: list[tuple[float, float, float]] = field(default_factory=list)
    state: VehicleState | None = None
    done: bool = False
    label: str = ""
    lap_time: float | None = None


def _format_time(metrics) -> str:
    if metrics.completed and metrics.lap_time is not None:
        return f"{metrics.lap_time:.2f}s"
    return f"dnf {metrics.progress * 100:.0f}%"


def _ghost_delta(ghosts: list[Ghost]) -> str:
    by_name = {ghost.name: ghost for ghost in ghosts}
    classical = by_name.get("classical")
    learned = by_name.get("ppo")
    if classical and learned and classical.lap_time is not None and learned.lap_time is not None:
        return f"delta {learned.lap_time - classical.lap_time:+.2f}s"
    return "delta —"


class LabDashboard:
    def __init__(self, headless_size: tuple[int, int] | None = None) -> None:
        pygame.init()
        pygame.display.set_caption("Autonomous Rally AI")
        self.screen = pygame.display.set_mode(headless_size or (WIDTH, HEIGHT))
        self.font = mono(16)
        self.small = mono(14)
        self.title_font = mono(18)
        self.clock = pygame.time.Clock()

    def close(self) -> None:
        pygame.quit()

    def layout(self):
        live = pygame.Rect(16, 52, 760, 448)
        model = pygame.Rect(788, 52, 476, 448)
        training = pygame.Rect(16, 512, 760, 250)
        action = pygame.Rect(788, 512, 476, 250)
        return live, model, training, action

    def frame(
        self,
        env: RallyEnv,
        *,
        title: str,
        algorithm: str,
        architecture: str,
        parameters: int | None,
        episode: int,
        reward_history: list[float],
        loss_history: list[float],
        learning_rate: float,
        entropy: float,
        observation: str,
        notes: list[str],
        ghosts: list[Ghost] | None = None,
        hist: list[float] | None = None,
        extra_lines: list[str] | None = None,
        follow: bool = False,
        show_rays: bool = True,
        markers: list[dict] | None = None,
        timing: dict | None = None,
        mode: str = "classical",
        samples: list | None = None,
        car: VehicleState | None = None,
        table_rows: list[dict] | None = None,
        leaderboard: list[dict] | None = None,
    ) -> None:
        self.screen.fill(BG)
        live, model, training, action = self.layout()
        for rect in (live, model, training, action):
            panel(self.screen, rect)
        self._rail(mode)
        self.screen.blit(self.small.render(title, True, MUTED), (980, 14))
        self.screen.blit(self.small.render(LEGEND, True, MUTED), (16, HEIGHT - 22))

        shown = samples if samples is not None else env.trace
        focus = None
        pose = car or env.car.state
        if follow:
            focus = (pose.x, pose.y)
        if table_rows is None:
            view = WorldView(live.inflate(-8, -8), env.track, focus=focus)
            self.screen.set_clip(live)
            draw_track(self.screen, env.track, view)
            if hist:
                draw_heatmap_band(self.screen, env.track, hist, view)
            if ghosts:
                for ghost in ghosts:
                    draw_polyline(self.screen, [(point[0], point[1]) for point in ghost.trace], view, ghost.color, 2)
                    if ghost.state is not None:
                        draw_car(self.screen, ghost.state, view, ghost.color)
            else:
                draw_speed_line(self.screen, shown, view, 2)
                draw_car(self.screen, pose, view, LEARNED)
            if show_rays:
                draw_rays(self.screen, env.rays, view)
            self.screen.set_clip(None)
            if markers:
                draw_markers(self.screen, markers, view, self.small)
        else:
            draw_results_table(self.screen, live.inflate(-12, -12), table_rows, self.small)

        self._model_panel(model, env, algorithm, architecture, parameters, episode, observation, learning_rate, entropy, timing, leaderboard)
        self.screen.blit(self.font.render("TRAINING", True, GOLD), (training.x + 12, training.y + 8))
        draw_sparkline(self.screen, pygame.Rect(training.x + 12, training.y + 36, training.width - 24, 96), reward_history, THROTTLE, "reward", self.small)
        draw_sparkline(self.screen, pygame.Rect(training.x + 12, training.y + 140, training.width - 24, 96), loss_history, BRAKE, "loss", self.small)
        self.screen.blit(self.font.render("TELEMETRY", True, GOLD), (action.x + 12, action.y + 8))
        draw_channels(self.screen, pygame.Rect(action.x + 12, action.y + 32, action.width - 24, 150), [sample for sample in shown if isinstance(sample, dict)], self.small)
        footer = list(extra_lines or [])[:3] or notes[:1]
        for index, line in enumerate(footer):
            self.screen.blit(self.small.render(line[:78], True, MUTED), (action.x + 12, action.y + 188 + index * 16))
        pygame.display.flip()

    def _rail(self, mode: str) -> None:
        x = 16
        self.screen.blit(self.title_font.render("RALLY AI", True, GOLD), (x, 12))
        x = 150
        for name in MODES:
            color = GOLD if name == mode else MUTED
            label = self.small.render(name.upper(), True, color)
            self.screen.blit(label, (x, 16))
            x += label.get_width() + 16

    def _model_panel(self, rect, env, algorithm, architecture, parameters, episode, observation, learning_rate, entropy, timing, leaderboard) -> None:
        meta = env.track.meta
        timing = timing or {}
        best = timing.get("best")
        delta = timing.get("delta")
        lines = [
            "MODEL",
            str(algorithm),
            str(architecture),
            f"params {parameters if parameters is not None else '—'}",
            f"{observation}  lr {learning_rate:g}  ent {entropy:g}",
            "",
            "TIMING",
            f"time {float(timing.get('time', env.time)):.2f}s",
            f"best {best:.2f}s" if best is not None else "best —",
            f"ref {timing['reference']:.2f}s" if timing.get("reference") is not None else "ref —",
            f"delta {delta:+.2f}s" if delta is not None else "delta —",
            "",
            "TRACK",
            f"#{meta.track_id}  seed {meta.seed}",
            f"{meta.difficulty_name}  {meta.difficulty:.2f}  {meta.turns} turns  {meta.length_m:.0f} m",
            "",
            "SENSORS",
            f"step {episode}  progress {env.progress_fraction * 100:.0f}%",
            f"speed {env.car.state.speed * 3.6:.0f} km/h",
            f"heading {np.degrees(env.trace[-1]['heading_error']) if env.trace else 0:+.1f} deg",
            f"center {env.projection.lateral:+.2f} m  curv {env.projection.curvature:+.3f}",
        ]
        if leaderboard:
            lines.append("")
            lines.append("BOARD")
            for entry in leaderboard[:3]:
                lines.append(f"{entry['driver']}  {entry['lap_time']:.2f}s")
        self._text_block(rect, lines)

    def _text_block(self, rect: pygame.Rect, lines: list[str]) -> None:
        y = rect.y + 10
        for line in lines:
            if y > rect.bottom - 18:
                break
            color = GOLD if line in {"MODEL", "TIMING", "TRACK", "SENSORS", "BOARD"} else TEXT
            self.screen.blit(self.small.render(line, True, color), (rect.x + 14, y))
            y += 18

    def poll(self) -> set[int]:
        pressed: set[int] = set()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pressed.add(pygame.QUIT)
            elif event.type == pygame.KEYDOWN:
                pressed.add(event.key)
        return pressed

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        pygame.image.save(self.screen, str(path))


def _load_bundle(checkpoint_dir: Path) -> dict:
    for name in ("ladder.json", "ppo_smoke.json"):
        path = checkpoint_dir / name
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _load_policy(path: Path):
    from stable_baselines3 import PPO

    model = PPO.load(str(path), device="cpu")
    parameters = int(sum(param.numel() for param in model.policy.parameters()))

    def predict(obs, _env):
        action, _state = model.predict(obs, deterministic=True)
        return np.asarray(action, dtype=np.float32)

    return predict, parameters, model


def _resolve_checkpoint(args: argparse.Namespace) -> Path:
    late = Path(args.checkpoint_dir) / "late.zip"
    requested = Path(args.checkpoint)
    if requested.name == "ppo_smoke.zip" and late.exists():
        return late
    return requested


def _comparison_rows() -> list[dict]:
    rows = []
    for path in sorted(Path("data/results").glob("eval_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        summary = payload.get("summary", {})
        progress = summary.get("progress", {}).get("mean")
        if progress is None:
            continue
        rows.append(
            {
                "label": payload.get("label", path.stem),
                "completion": float(summary.get("completed_rate", progress)),
                "collisions": float(summary.get("collisions", {}).get("mean") or 0.0),
                "lap_time": summary.get("lap_time", {}).get("mean"),
                "off_track": float(summary.get("off_track_fraction", {}).get("mean") or 0.0),
                "parameters": payload.get("parameters"),
            }
        )
    return rows


def _save_trace(env: RallyEnv, name: str) -> None:
    path = Path("data/trajectories") / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = ("x", "y", "theta", "t", "s", "speed", "steer", "throttle", "brake", "curvature_ahead", "collision")
    payload = []
    for sample in env.trace:
        payload.append({key: float(sample[key]) if key != "collision" else bool(sample[key]) for key in keys if key in sample})
    path.write_text(json.dumps(payload), encoding="utf-8")


def _state_from_sample(sample: dict, fallback: VehicleState) -> VehicleState:
    return VehicleState(
        x=float(sample.get("x", fallback.x)),
        y=float(sample.get("y", fallback.y)),
        theta=float(sample.get("theta", fallback.theta)),
        vx=float(sample.get("speed", 0.0)),
    )


class LabSession:
    def __init__(self, dashboard: LabDashboard, args: argparse.Namespace, bundle: dict) -> None:
        self.dashboard = dashboard
        self.args = args
        self.bundle = bundle
        self.mode = args.mode if args.mode in MODES else "classical"
        self.follow = bool(getattr(args, "follow", False))
        self.show_rays = True
        self.paused = False
        self.scrub: int | None = None
        self.best_lap: float | None = None
        self.reference = reference_lap(args.track_seed)
        self.leaderboard = load_board()
        self.env: RallyEnv | None = None
        self.obs = None
        self.policy = None
        self.parameters = bundle.get("parameters")
        self.algorithm = bundle.get("experiment", {}).get("algorithm", "classical")
        self.architecture = bundle.get("experiment", {}).get("architecture", "—")
        self.learning_rate = float(bundle.get("experiment", {}).get("learning_rate", 0.0))
        self.entropy = float(bundle.get("experiment", {}).get("entropy_coef", 0.0))
        self.observation = bundle.get("experiment", {}).get("observation", "sensors")
        self.checkpoints: list[Path] = []
        self.ckpt_index = 0
        self.ghosts: list[Ghost] = []
        self.ghost_envs: dict = {}
        self.slots: list[dict] = []
        self.rows: list[dict] = []
        self.replay: list[dict] | None = None
        self.steps = 0
        self.heat: list[float] = []
        self._stop = False
        if args.replay:
            self.replay = json.loads(Path(args.replay).read_text(encoding="utf-8"))
            self.paused = True
            self.scrub = max(0, len(self.replay) // 2)
        self.set_mode(self.mode)

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        self.steps = 0
        if self.replay is None:
            self.paused = False
            self.scrub = None
        if mode == "compare":
            self.rows = _comparison_rows()
            if self.rows:
                save_comparison_bars(self.rows, Path("data/results/comparison.png"))
            self._open_env()
            return
        if mode == "evolution":
            self._open_evolution()
            return
        if mode == "ghosts":
            self._open_ghosts()
            return
        self._open_env()
        self._bind_single_policy()

    def _open_env(self) -> None:
        self.env = RallyEnv(
            EnvConfig(
                track_seed=self.args.track_seed,
                preset=self.args.preset,
                reward=self.args.reward,
                max_steps=max(self.args.steps, 8000),
            )
        )
        self.obs, _info = self.env.reset(seed=self.args.seed)

    def _bind_single_policy(self) -> None:
        self.checkpoints = sorted(Path(self.args.checkpoint_dir).glob("*.zip")) if self.mode == "policy" else []
        if self.mode == "classical":
            expert = PurePursuit()
            self.policy = lambda _obs, env: expert.act(env)
            self.algorithm = "classical"
            self.architecture = "pure pursuit"
            self.parameters = None
            self.learning_rate = 0.0
            self.entropy = 0.0
        elif self.mode == "policy":
            path = _resolve_checkpoint(self.args)
            if self.checkpoints:
                self.ckpt_index = _index_of(self.checkpoints, path)
                path = self.checkpoints[self.ckpt_index]
            if path.exists():
                self.policy, self.parameters, _model = _load_policy(path)
                self.algorithm = path.stem
            else:
                self.policy = lambda _obs, env: env.action_space.sample()
                self.algorithm = "no checkpoint"
        else:
            self.policy = None
            self.algorithm = "human"
            self.architecture = "keyboard"
            self.parameters = None

    def _open_ghosts(self) -> None:
        colors = {"classical": CLASSICAL, "random": RANDOM, "ppo": LEARNED}
        expert = PurePursuit()
        agents = {
            "classical": lambda _obs, env: expert.act(env),
            "random": lambda _obs, env: env.action_space.sample(),
        }
        learned = _resolve_checkpoint(self.args)
        if learned.exists():
            predict, self.parameters, _model = _load_policy(learned)
            agents["ppo"] = predict
            self.algorithm = learned.stem
        self.ghosts = []
        self.ghost_envs = {}
        for name, policy in agents.items():
            env = RallyEnv(EnvConfig(track_seed=self.args.track_seed, preset=self.args.preset, reward="C", max_steps=max(self.args.steps, 8000)))
            obs, _info = env.reset(seed=self.args.seed)
            self.ghost_envs[name] = {"env": env, "policy": policy, "obs": obs}
            self.ghosts.append(Ghost(name=name, color=colors[name], state=env.car.state.copy()))
        self.env = self.ghost_envs["classical"]["env"]

    def _open_evolution(self) -> None:
        self.slots = []
        directory = Path(self.args.checkpoint_dir)
        for name, filename, unseen in EVOLUTION:
            seed = TEST_FIXTURE_SEED if unseen else self.args.track_seed
            env = RallyEnv(EnvConfig(track_seed=seed, preset=self.args.preset, max_steps=max(self.args.steps, 400)))
            obs, _info = env.reset(seed=0)
            policy = None
            status = "no checkpoint"
            if name == "Random":
                policy = lambda _obs, live: live.action_space.sample()
                status = "random"
            elif filename and (directory / filename).exists():
                policy, _count, _model = _load_policy(directory / filename)
                status = "loaded"
            self.slots.append({"name": name, "env": env, "obs": obs, "policy": policy, "status": status, "done": policy is None})
        self.env = self.slots[0]["env"]

    def handle(self, pressed: set[int]) -> None:
        for index, key in enumerate((pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5, pygame.K_6)):
            if key in pressed:
                self.set_mode(MODES[index])
                return
        if pygame.K_f in pressed:
            self.follow = not self.follow
        if pygame.K_l in pressed:
            self.show_rays = not self.show_rays
        if pygame.K_p in pressed:
            self.paused = not self.paused
            samples = self._samples()
            self.scrub = max(0, len(samples) - 1) if self.paused and samples else None
        if pygame.K_e in pressed:
            self.export()
        if pygame.K_r in pressed and self.mode in {"manual", "classical", "policy"} and self.env is not None:
            self.obs, _info = self.env.reset(seed=self.args.seed)
        if self.paused:
            samples = self._samples()
            if samples and pygame.K_LEFT in pressed:
                self.scrub = max(0, (self.scrub or 0) - 1)
            if samples and pygame.K_RIGHT in pressed:
                self.scrub = min(len(samples) - 1, (self.scrub or 0) + 1)
        if self.mode == "policy" and self.checkpoints:
            if pygame.K_LEFTBRACKET in pressed or pygame.K_RIGHTBRACKET in pressed:
                if pygame.K_LEFTBRACKET in pressed:
                    self.ckpt_index = max(0, self.ckpt_index - 1)
                else:
                    self.ckpt_index = min(len(self.checkpoints) - 1, self.ckpt_index + 1)
                path = self.checkpoints[self.ckpt_index]
                self.policy, self.parameters, _model = _load_policy(path)
                self.algorithm = path.stem
                self.obs, _info = self.env.reset(seed=self.args.seed)

    def advance(self) -> bool:
        if self.paused or self.replay is not None or self.mode == "compare":
            return self.mode == "compare" or self.replay is not None
        if self.mode == "manual":
            return self._advance_manual()
        if self.mode in {"classical", "policy"}:
            return self._advance_single()
        if self.mode == "ghosts":
            return self._advance_ghosts()
        if self.mode == "evolution":
            return self._advance_evolution()
        return False

    def _advance_manual(self) -> bool:
        held = pygame.key.get_pressed()
        action = action_from_flags(
            bool(held[pygame.K_LEFT] or held[pygame.K_a]),
            bool(held[pygame.K_RIGHT] or held[pygame.K_d]),
            bool(held[pygame.K_UP] or held[pygame.K_w] or self.args.hold_throttle),
            bool(held[pygame.K_DOWN] or held[pygame.K_s]),
        )
        self.obs, _reward, terminated, truncated, _info = self.env.step(action.as_array())
        self._note_lap(self.env, "human")
        if terminated or truncated:
            self.obs, _info = self.env.reset(seed=self.args.seed)
        return False

    def _advance_single(self) -> bool:
        action = self.policy(self.obs, self.env)
        self.obs, _reward, terminated, truncated, _info = self.env.step(action)
        finished = self._note_lap(self.env, self.algorithm)
        if terminated or truncated:
            self._keep_heat()
            if finished and self.args.until_lap:
                return True
            if self.args.until_lap:
                return True
            self.obs, _info = self.env.reset(seed=self.args.seed)
        return False

    def _advance_ghosts(self) -> bool:
        for ghost in self.ghosts:
            slot = self.ghost_envs[ghost.name]
            env = slot["env"]
            if ghost.done:
                continue
            slot["obs"], _reward, terminated, truncated, _info = env.step(slot["policy"](slot["obs"], env))
            ghost.trace.append((env.car.state.x, env.car.state.y, env.car.state.speed))
            ghost.state = env.car.state.copy()
            metrics = env.episode_metrics()
            ghost.label = _format_time(metrics)
            if metrics.completed and metrics.lap_time is not None:
                ghost.lap_time = metrics.lap_time
                self._remember(ghost.name, metrics.lap_time, env)
            if terminated or truncated:
                ghost.done = True
        self.env = self.ghost_envs["classical"]["env"]
        return all(ghost.done for ghost in self.ghosts)

    def _advance_evolution(self) -> bool:
        for slot in self.slots:
            if slot["done"] or slot["policy"] is None:
                continue
            slot["obs"], _reward, terminated, truncated, _info = slot["env"].step(slot["policy"](slot["obs"], slot["env"]))
            if terminated or truncated:
                slot["done"] = True
        return all(slot["done"] or slot["policy"] is None for slot in self.slots)

    def _keep_heat(self) -> None:
        if self.env is None or not self.env.trace:
            return
        hist = crash_histogram(self.env.trace, self.env.track.length)
        if not self.heat:
            self.heat = hist
            return
        self.heat = [left + right for left, right in zip(self.heat, hist)]

    def _note_lap(self, env: RallyEnv, driver: str) -> bool:
        metrics = env.episode_metrics()
        if not metrics.completed or metrics.lap_time is None:
            return False
        self._remember(driver, metrics.lap_time, env)
        _save_trace(env, f"{self.mode}_{self.args.track_seed}")
        return True

    def _remember(self, driver: str, lap_time: float, env: RallyEnv) -> None:
        if self.best_lap is None or lap_time < self.best_lap:
            self.best_lap = lap_time
        self.leaderboard = record_lap(driver, lap_time, env.track.meta.seed, True)

    def _samples(self) -> list:
        if self.replay is not None:
            return self.replay
        if self.env is None:
            return []
        return self.env.trace

    def _timing(self) -> dict:
        current = self.env.time if self.env is not None else 0.0
        metrics = self.env.episode_metrics() if self.env is not None else None
        if metrics and metrics.completed and metrics.lap_time is not None:
            current = metrics.lap_time
        delta = None
        if self.best_lap is not None and self.reference is not None:
            delta = self.best_lap - self.reference
        return {"time": current, "best": self.best_lap, "delta": delta, "reference": self.reference}

    def _car(self) -> VehicleState | None:
        samples = self._samples()
        if self.scrub is None or not samples:
            return None
        sample = samples[max(0, min(self.scrub, len(samples) - 1))]
        if not isinstance(sample, dict):
            return None
        fallback = self.env.car.state if self.env is not None else VehicleState()
        return _state_from_sample(sample, fallback)

    def export(self) -> None:
        stamp = f"{self.mode}_{self.args.track_seed}"
        self.dashboard.save(Path("data/results") / f"frame_{stamp}.png")

    def draw(self) -> None:
        if self.mode == "evolution":
            self._draw_evolution()
            return
        samples = self._samples()
        window = samples if self.scrub is None else samples[: self.scrub + 1]
        notes = describe_trace([sample for sample in window if isinstance(sample, dict)])
        markers = corner_markers([sample for sample in window if isinstance(sample, dict)])
        extra = list(notes[:2])
        if self.mode == "ghosts":
            extra = [
                _ghost_delta(self.ghosts),
                "   ".join(f"{ghost.name} {ghost.label}" for ghost in self.ghosts),
            ]
        title = self.algorithm
        if self.mode == "policy" and self.checkpoints:
            title = f"{self.algorithm}  {self.ckpt_index + 1}/{len(self.checkpoints)}"
        if self.paused and self.scrub is not None:
            title = f"{title}  scrub {self.scrub + 1}/{max(len(samples), 1)}"
        hist = None
        if self.args.heatmap and self.env is not None:
            current = crash_histogram(self.env.trace, self.env.track.length)
            hist = [left + right for left, right in zip(self.heat, current)] if self.heat else current
        self.dashboard.frame(
            self.env,
            title=title,
            algorithm=self.algorithm,
            architecture=self.architecture,
            parameters=self.parameters,
            episode=self.steps,
            reward_history=[] if self.mode in {"manual", "classical", "evolution"} else [float(v) for v in self.bundle.get("episode_rewards", [])],
            loss_history=[] if self.mode in {"manual", "classical", "evolution"} else [float(v) for v in self.bundle.get("loss_history", [])],
            learning_rate=self.learning_rate,
            entropy=self.entropy,
            observation=self.observation,
            notes=notes[:1],
            ghosts=self.ghosts if self.mode == "ghosts" else None,
            hist=hist,
            extra_lines=extra,
            follow=self.follow,
            show_rays=self.show_rays,
            markers=markers,
            timing=self._timing(),
            mode=self.mode,
            samples=window,
            car=self._car(),
            table_rows=self.rows if self.mode == "compare" else None,
            leaderboard=[entry for entry in self.leaderboard if int(entry["track_seed"]) == int(self.args.track_seed)],
        )

    def _draw_evolution(self) -> None:
        screen = self.dashboard.screen
        screen.fill(BG)
        screen.blit(self.dashboard.title_font.render("HOW THE AI LEARNED", True, GOLD), (16, 12))
        screen.blit(self.dashboard.small.render(LEGEND, True, MUTED), (16, HEIGHT - 22))
        for index, slot in enumerate(self.slots):
            col = index % 3
            row = index // 3
            rect = pygame.Rect(16 + col * 422, 48 + row * 252, 410, 240)
            panel(screen, rect)
            env = slot["env"]
            view = WorldView(rect.inflate(-8, -28), env.track)
            screen.set_clip(rect)
            draw_track(screen, env.track, view)
            if slot["status"] != "no checkpoint":
                draw_speed_line(screen, env.trace, view, 2)
                draw_car(screen, env.car.state, view, LEARNED if slot["name"] != "Random" else RANDOM)
            screen.set_clip(None)
            screen.blit(self.dashboard.small.render(f"{slot['name']}  {slot['status']}", True, TEXT), (rect.x + 8, rect.bottom - 22))
        pygame.display.flip()

    def run(self) -> None:
        limit = 1 if self.replay is not None else self.args.steps
        while self.steps < limit and not self._stop:
            pressed = self.dashboard.poll()
            if pygame.QUIT in pressed or pygame.K_ESCAPE in pressed:
                break
            self.handle(pressed)
            finished = self.advance()
            self.steps += 1
            screenshot = bool(self.args.screenshot)
            if (not screenshot) or finished or self.steps >= limit:
                self.draw()
            if not screenshot:
                self.dashboard.clock.tick(60)
            if screenshot and (finished or self.steps >= limit):
                break
        if self.args.screenshot:
            self.draw()
            self.dashboard.save(Path(self.args.screenshot))


def _index_of(paths: list[Path], target: Path) -> int:
    for index, path in enumerate(paths):
        if path.name == target.name or (target.exists() and path.resolve() == target.resolve()):
            return index
    return 0


def _run_slice(dashboard: LabDashboard, args: argparse.Namespace, bundle: dict) -> None:
    path = _resolve_checkpoint(args)
    predict, parameters, _model = _load_policy(path)
    out = Path(args.slice_out)
    save_policy_slice(lambda obs: predict(obs, None), out)
    env = RallyEnv(EnvConfig(track_seed=args.track_seed, preset=args.preset, max_steps=10))
    env.reset(seed=args.seed)
    experiment = bundle.get("experiment", {})
    dashboard.frame(
        env,
        title="policy slice",
        algorithm=path.stem,
        architecture=experiment.get("architecture", "MLP2"),
        parameters=parameters,
        episode=0,
        reward_history=[float(v) for v in bundle.get("episode_rewards", [])],
        loss_history=[float(v) for v in bundle.get("loss_history", [])],
        learning_rate=float(experiment.get("learning_rate", 0.0)),
        entropy=float(experiment.get("entropy_coef", 0.0)),
        observation=experiment.get("observation", "sensors"),
        notes=[f"wrote {out}"],
        mode="policy",
        timing={"time": 0.0, "best": None, "delta": None},
    )
    if args.screenshot:
        dashboard.save(Path(args.screenshot))


def run_mode(args: argparse.Namespace) -> Path | None:
    bundle = _load_bundle(Path(args.checkpoint_dir))
    dashboard = LabDashboard()
    screenshot = Path(args.screenshot) if args.screenshot else None
    try:
        if args.mode == "slice":
            _run_slice(dashboard, args, bundle)
        else:
            LabSession(dashboard, args, bundle).run()
    finally:
        dashboard.close()
    return screenshot


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autonomous Rally AI lab window.")
    parser.add_argument("--mode", default="classical", choices=(*MODES, "slice"))
    parser.add_argument("--preset", default="easy")
    parser.add_argument("--track-seed", type=int, default=EASY_FIXTURE_SEED)
    parser.add_argument("--reward", default="C")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=2400)
    parser.add_argument("--until-lap", action="store_true")
    parser.add_argument("--heatmap", action="store_true")
    parser.add_argument("--screenshot", default="")
    parser.add_argument("--checkpoint", default="training/checkpoints/ppo_smoke.zip")
    parser.add_argument("--checkpoint-dir", default="training/checkpoints")
    parser.add_argument("--slice-out", default="data/results/policy_slice.png")
    parser.add_argument("--hold-throttle", action="store_true")
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--replay", default="")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_mode(args)


if __name__ == "__main__":
    main()
