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
from environment.types import Action, EnvConfig
from training.metrics import crash_histogram, describe_trace
from visualization.plots import save_comparison_bars, save_policy_slice
from visualization.renderer import (
    WorldView,
    draw_bar,
    draw_car,
    draw_heatmap,
    draw_polyline,
    draw_rays,
    draw_sparkline,
    draw_steering,
    draw_track,
)

WIDTH = 1280
HEIGHT = 800
BG = (14, 16, 22)
PANEL = (24, 28, 38)
TEXT = (220, 224, 232)
MUTED = (150, 156, 170)
GOLD = (232, 196, 72)

EVOLUTION = (
    ("Random", None),
    ("Beginner", "training/checkpoints/beginner.zip"),
    ("Competent", "training/checkpoints/competent.zip"),
    ("Advanced", "training/checkpoints/advanced.zip"),
    ("Racing line", "training/checkpoints/racing_line.zip"),
    ("Unseen", "training/checkpoints/ppo_smoke.zip"),
)


def action_from_flags(left: bool, right: bool, throttle: bool, brake: bool) -> Action:
    steering = float(right) - float(left)
    return Action(steering, 1.0 if throttle else 0.0, 1.0 if brake else 0.0)


@dataclass
class Ghost:
    name: str
    color: tuple[int, int, int]
    trace: list[tuple[float, float]] = field(default_factory=list)
    state: object | None = None
    done: bool = False
    label: str = ""


def _format_time(metrics) -> str:
    if metrics.completed and metrics.lap_time is not None:
        return f"{metrics.lap_time:.2f}s"
    return f"dnf {metrics.progress * 100:.0f}%"


def simulate_agent(env: RallyEnv, policy, steps: int, until_lap: bool = False):
    obs, _info = env.reset()
    for _ in range(steps):
        action = policy(obs, env)
        obs, _reward, terminated, truncated, _info = env.step(action)
        if until_lap and env.lap_completed:
            break
        if terminated or truncated:
            break
    return env


class LabDashboard:
    def __init__(self, headless_size: tuple[int, int] | None = None) -> None:
        pygame.init()
        pygame.display.set_caption("Autonomous Rally AI")
        if headless_size is None:
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        else:
            self.screen = pygame.display.set_mode(headless_size)
        self.font = pygame.font.SysFont("consolas", 16)
        self.small = pygame.font.SysFont("consolas", 14)
        self.title_font = pygame.font.SysFont("consolas", 20)
        self.clock = pygame.time.Clock()

    def close(self) -> None:
        pygame.quit()

    def layout(self):
        live = pygame.Rect(16, 48, 760, 470)
        model = pygame.Rect(788, 48, 476, 470)
        training = pygame.Rect(16, 530, 760, 254)
        action = pygame.Rect(788, 530, 476, 254)
        return live, model, training, action

    def frame(self, env: RallyEnv, *, title: str, algorithm: str, architecture: str, parameters: int | None, episode: int, reward_history: list[float], loss_history: list[float], learning_rate: float, entropy: float, observation: str, notes: list[str], ghosts: list[Ghost] | None = None, hist: list[float] | None = None, extra_lines: list[str] | None = None) -> None:
        self.screen.fill(BG)
        live, model, training, action = self.layout()
        for rect in (live, model, training, action):
            pygame.draw.rect(self.screen, PANEL, rect, border_radius=6)
        self.screen.blit(self.title_font.render("AUTONOMOUS RALLY AI", True, GOLD), (16, 12))
        self.screen.blit(self.small.render(title, True, MUTED), (320, 16))

        view = WorldView(live.inflate(-12, -12), env.track)
        draw_track(self.screen, env.track, view)
        if hist:
            draw_heatmap(self.screen, env.track, hist, view)
        if ghosts:
            for ghost in ghosts:
                draw_polyline(self.screen, ghost.trace, view, ghost.color, 2)
                if ghost.state is not None:
                    draw_car(self.screen, ghost.state, view, ghost.color)
            draw_rays(self.screen, env.rays, view)
        else:
            draw_polyline(self.screen, [(s["x"], s["y"]) for s in env.trace], view, (255, 120, 96), 2)
            draw_rays(self.screen, env.rays, view)
            draw_car(self.screen, env.car.state, view)

        meta = env.track.meta
        metrics = env.episode_metrics()
        lap_line = f"lap {metrics.lap_time:.2f}s" if env.lap_completed and metrics.lap_time is not None else "lap —"
        lines = [
            "MODEL",
            f"{algorithm}",
            f"{architecture}",
            f"Parameters  {parameters if parameters is not None else '—'}",
            f"Observation {observation}",
            f"lr {learning_rate:g}   entropy {entropy:g}",
            "",
            "TRACK",
            f"#{meta.track_id}  seed {meta.seed}",
            f"{meta.difficulty_name}  {meta.difficulty:.2f}",
            f"turns {meta.turns}   {meta.length_m:.0f} m",
            "",
            "EPISODE",
            f"step {episode}",
            f"reward {env.last_reward:+.2f}",
            lap_line,
            f"progress {env.progress_fraction * 100:.0f}%",
            "",
            "SENSORS",
            f"speed {env.car.state.speed * 3.6:.0f} km/h",
            f"heading err {np.degrees(env.trace[-1]['heading_error']) if env.trace else 0:+.1f} deg",
            f"center {env.projection.lateral:+.2f} m",
            f"curvature {env.projection.curvature:+.3f}",
        ]
        if extra_lines:
            lines.extend([""] + extra_lines)
        self._text_block(model, lines)

        reward_rect = pygame.Rect(training.x + 12, training.y + 36, training.width - 24, 90)
        loss_rect = pygame.Rect(training.x + 12, training.y + 136, training.width - 24, 90)
        self.screen.blit(self.font.render("TRAINING", True, TEXT), (training.x + 12, training.y + 8))
        draw_sparkline(self.screen, reward_rect, reward_history, (120, 200, 140), "reward", self.small)
        draw_sparkline(self.screen, loss_rect, loss_history, (220, 140, 120), "loss", self.small)

        control = env.car.prev_action
        self.screen.blit(self.font.render("ACTION", True, TEXT), (action.x + 12, action.y + 8))
        self.screen.blit(self.small.render(f"steering {control.steering:+.2f}", True, TEXT), (action.x + 12, action.y + 40))
        draw_steering(self.screen, (action.x + 12, action.y + 64), action.width - 24, control.steering)
        self.screen.blit(self.small.render(f"throttle {control.throttle:.2f}", True, TEXT), (action.x + 12, action.y + 88))
        draw_bar(self.screen, (action.x + 12, action.y + 110), action.width - 24, control.throttle, (96, 180, 120))
        self.screen.blit(self.small.render(f"brake {control.brake:.2f}", True, TEXT), (action.x + 12, action.y + 132))
        draw_bar(self.screen, (action.x + 12, action.y + 154), action.width - 24, control.brake, (200, 96, 96))
        breakdown = env.last_breakdown or {}
        if breakdown:
            labels = (("progress", "progress"), ("off_track", "off-track"), ("collision", "collision"), ("time", "time"))
            bits = [f"{label} {breakdown[key]:+.2f}" for key, label in labels if key in breakdown]
            self.screen.blit(self.small.render("  ".join(bits), True, MUTED), (action.x + 12, action.y + 180))
        if notes:
            self.screen.blit(self.small.render(notes[0][:70], True, MUTED), (action.x + 12, action.y + 206))
        pygame.display.flip()

    def _text_block(self, rect: pygame.Rect, lines: list[str]) -> None:
        y = rect.y + 10
        for line in lines:
            if y > rect.bottom - 18:
                break
            color = GOLD if line in {"MODEL", "TRACK", "EPISODE", "SENSORS"} else TEXT
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
    path = checkpoint_dir / "ppo_smoke.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_policy(path: Path):
    from stable_baselines3 import PPO

    model = PPO.load(str(path), device="cpu")
    parameters = int(sum(param.numel() for param in model.policy.parameters()))

    def predict(obs, _env):
        action, _state = model.predict(obs, deterministic=True)
        return np.asarray(action, dtype=np.float32)

    return predict, parameters, model


def run_mode(args: argparse.Namespace) -> Path | None:
    bundle = _load_bundle(Path(args.checkpoint_dir))
    experiment = bundle.get("experiment", {})
    reward_history = [float(v) for v in bundle.get("episode_rewards", [])]
    loss_history = [float(v) for v in bundle.get("loss_history", [])]
    if args.mode in {"manual", "classical", "evolution"}:
        reward_history = []
        loss_history = []
    parameters = bundle.get("parameters")
    algorithm = experiment.get("algorithm", "manual" if args.mode == "manual" else "classical")
    architecture = experiment.get("architecture", "—")
    learning_rate = float(experiment.get("learning_rate", 0.0))
    entropy = float(experiment.get("entropy_coef", 0.0))
    observation = experiment.get("observation", "sensors")

    dashboard = LabDashboard()
    screenshot = Path(args.screenshot) if args.screenshot else None
    try:
        if args.mode == "manual":
            _run_manual(dashboard, args, reward_history, loss_history, parameters)
        elif args.mode == "classical":
            expert = PurePursuit()
            _run_single(
                dashboard,
                args,
                lambda _obs, env: expert.act(env),
                "classical",
                "pure pursuit",
                None,
                0.0,
                0.0,
                "sensors",
                reward_history,
                loss_history,
            )
        elif args.mode == "policy":
            predict, parameters, _model = _load_policy(Path(args.checkpoint))
            _run_single(
                dashboard,
                args,
                predict,
                algorithm,
                architecture,
                parameters,
                learning_rate,
                entropy,
                observation,
                reward_history,
                loss_history,
            )
        elif args.mode == "ghosts":
            _run_ghosts(dashboard, args, bundle, reward_history, loss_history)
        elif args.mode == "evolution":
            _run_evolution(dashboard, args, reward_history, loss_history)
        elif args.mode == "compare":
            _run_compare(dashboard, args, reward_history, loss_history)
        elif args.mode == "slice":
            predict, parameters, model = _load_policy(Path(args.checkpoint))
            out = Path(args.slice_out)
            save_policy_slice(lambda obs: predict(obs, None), out)
            env = RallyEnv(EnvConfig(track_seed=args.track_seed, preset=args.preset, max_steps=args.steps))
            env.reset()
            notes = [f"policy slice written to {out}"]
            dashboard.frame(
                env,
                title="policy slice",
                algorithm=algorithm,
                architecture=architecture,
                parameters=parameters,
                episode=0,
                reward_history=reward_history,
                loss_history=loss_history,
                learning_rate=learning_rate,
                entropy=entropy,
                observation=observation,
                notes=notes,
            )
            if screenshot:
                dashboard.save(screenshot)
        else:
            raise SystemExit(f"unknown mode {args.mode}")
        if screenshot and args.mode != "slice":
            dashboard.save(screenshot)
    finally:
        dashboard.close()
    return screenshot


def _pace(dashboard, args, fps: int = 60) -> None:
    if not getattr(args, "screenshot", ""):
        dashboard.clock.tick(fps)


def _alive(pressed: set[int]) -> bool:
    return pygame.QUIT not in pressed and pygame.K_ESCAPE not in pressed


def _run_manual(dashboard, args, reward_history, loss_history, parameters) -> None:
    env = RallyEnv(EnvConfig(track_seed=args.track_seed, preset=args.preset, max_steps=args.steps))
    env.reset()
    for step in range(args.steps):
        if not _alive(dashboard.poll()):
            break
        pressed = pygame.key.get_pressed()
        action = action_from_flags(
            bool(pressed[pygame.K_LEFT] or pressed[pygame.K_a]),
            bool(pressed[pygame.K_RIGHT] or pressed[pygame.K_d]),
            bool(pressed[pygame.K_UP] or pressed[pygame.K_w] or args.hold_throttle),
            bool(pressed[pygame.K_DOWN] or pressed[pygame.K_s]),
        )
        if pressed[pygame.K_r]:
            env.reset()
        env.step(action.as_array())
        dashboard.frame(
            env,
            title="manual",
            algorithm="human",
            architecture="keyboard",
            parameters=parameters,
            episode=step,
            reward_history=reward_history,
            loss_history=loss_history,
            learning_rate=0.0,
            entropy=0.0,
            observation="sensors",
            notes=["arrows or WASD, R reset, esc quit"],
        )
        _pace(dashboard, args)


def _run_single(dashboard, args, policy, algorithm, architecture, parameters, learning_rate, entropy, observation, reward_history, loss_history) -> None:
    checkpoints = sorted(Path(args.checkpoint_dir).glob("*.zip")) if args.mode == "policy" else []
    loaded = []
    for path in checkpoints:
        predict, count, _model = _load_policy(path)
        loaded.append((path, predict, count))
    index = 0
    active = policy
    active_parameters = parameters
    active_name = algorithm
    if loaded and args.mode == "policy":
        index = _index_of(checkpoints, Path(args.checkpoint))
        active = loaded[index][1]
        active_parameters = loaded[index][2]
        active_name = loaded[index][0].stem

    env = RallyEnv(EnvConfig(track_seed=args.track_seed, preset=args.preset, reward=args.reward, max_steps=max(args.steps, 8000)))
    obs, _info = env.reset(seed=args.seed)
    for step in range(args.steps):
        pressed = dashboard.poll()
        if not _alive(pressed):
            break
        if loaded and pygame.K_LEFTBRACKET in pressed:
            index = max(0, index - 1)
            active = loaded[index][1]
            active_parameters = loaded[index][2]
            active_name = loaded[index][0].stem
            obs, _info = env.reset(seed=args.seed)
        if loaded and pygame.K_RIGHTBRACKET in pressed:
            index = min(len(loaded) - 1, index + 1)
            active = loaded[index][1]
            active_parameters = loaded[index][2]
            active_name = loaded[index][0].stem
            obs, _info = env.reset(seed=args.seed)
        action = active(obs, env)
        obs, _reward, _terminated, _truncated, _info = env.step(action)
        hist = crash_histogram(env.trace, env.track.length) if args.heatmap else None
        notes = describe_trace(env.trace)
        slider = f"checkpoint {index + 1}/{len(loaded)}  [ and ]" if loaded else "no checkpoint files"
        finished = bool(env.lap_completed or _terminated or _truncated or step + 1 == args.steps)
        if (not args.screenshot) or finished:
            dashboard.frame(
                env,
                title=active_name if args.mode != "policy" else f"{active_name}  {slider}",
                algorithm=active_name,
                architecture=architecture,
                parameters=active_parameters,
                episode=step,
                reward_history=reward_history,
                loss_history=loss_history,
                learning_rate=learning_rate,
                entropy=entropy,
                observation=observation,
                notes=notes[:1],
                hist=hist,
                extra_lines=notes[:3],
            )
        _pace(dashboard, args)
        if env.lap_completed:
            _save_trace(env, f"{args.mode}_{args.track_seed}")
        if args.until_lap and env.lap_completed:
            break
        if _terminated or _truncated:
            if args.until_lap:
                break
            obs, _info = env.reset(seed=args.seed)


def _save_trace(env: RallyEnv, name: str) -> None:
    path = Path("data/trajectories") / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [{"x": sample["x"], "y": sample["y"], "t": sample["t"]} for sample in env.trace]
    path.write_text(json.dumps(payload), encoding="utf-8")


def _index_of(paths: list[Path], target: Path) -> int:
    resolved = target.resolve() if target.exists() else target
    for i, path in enumerate(paths):
        if path.resolve() == resolved or path.name == target.name:
            return i
    return 0


def _run_ghosts(dashboard, args, bundle, reward_history, loss_history) -> None:
    seed = args.track_seed
    preset = args.preset
    colors = {"classical": (120, 200, 255), "random": (170, 170, 170), "ppo": (232, 196, 72)}
    expert = PurePursuit()
    agents = {
        "classical": lambda _obs, env: expert.act(env),
        "random": lambda _obs, env: env.action_space.sample(),
    }
    parameters = bundle.get("parameters")
    checkpoint = Path(args.checkpoint)
    if checkpoint.exists():
        predict, parameters, _model = _load_policy(checkpoint)
        agents["ppo"] = predict
    envs = {}
    ghosts = []
    for name, policy in agents.items():
        env = RallyEnv(EnvConfig(track_seed=seed, preset=preset, reward="C", max_steps=args.steps))
        obs, _info = env.reset(seed=args.seed)
        envs[name] = {"env": env, "policy": policy, "obs": obs}
        ghosts.append(Ghost(name=name, color=colors[name], state=env.car.state.copy()))
    for _step in range(args.steps):
        if not _alive(dashboard.poll()):
            break
        for ghost in ghosts:
            slot = envs[ghost.name]
            env = slot["env"]
            if ghost.done:
                continue
            action = slot["policy"](slot["obs"], env)
            slot["obs"], _reward, terminated, truncated, _info = env.step(action)
            ghost.trace.append((env.car.state.x, env.car.state.y))
            ghost.state = env.car.state.copy()
            ghost.label = _format_time(env.episode_metrics())
            if terminated or truncated:
                ghost.done = True
        host = envs["classical"]["env"]
        finished = all(ghost.done for ghost in ghosts) or _step + 1 == args.steps
        if (not args.screenshot) or finished:
            dashboard.frame(
                host,
                title="ghosts",
                algorithm="compare",
                architecture="same start",
                parameters=parameters,
                episode=host.steps,
                reward_history=reward_history,
                loss_history=loss_history,
                learning_rate=float(bundle.get("experiment", {}).get("learning_rate", 0.0)),
                entropy=float(bundle.get("experiment", {}).get("entropy_coef", 0.0)),
                observation="sensors",
                notes=[],
                ghosts=ghosts,
                extra_lines=[f"{ghost.name}  {ghost.label}" for ghost in ghosts],
            )
        _pace(dashboard, args)
        if finished:
            break


def _run_evolution(dashboard, args, reward_history, loss_history) -> None:
    slots = []
    for name, path in EVOLUTION:
        seed = TEST_FIXTURE_SEED if name == "Unseen" else args.track_seed
        env = RallyEnv(EnvConfig(track_seed=seed, preset=args.preset, max_steps=max(args.steps, 30)))
        obs, _info = env.reset(seed=0)
        policy = None
        status = "no checkpoint"
        if path is None:
            policy = lambda _obs, env: env.action_space.sample()
            status = "random"
        elif Path(path).exists():
            policy, _count, _model = _load_policy(Path(path))
            status = "loaded"
        slots.append({"name": name, "env": env, "obs": obs, "policy": policy, "status": status, "done": policy is None})
    for _step in range(min(args.steps, 240)):
        if not _alive(dashboard.poll()):
            break
        for slot in slots:
            if slot["done"] or slot["policy"] is None:
                continue
            slot["obs"], _reward, terminated, truncated, _info = slot["env"].step(slot["policy"](slot["obs"], slot["env"]))
            if terminated or truncated:
                slot["done"] = True
        last = _step + 1 == min(args.steps, 240) or all(slot["done"] or slot["policy"] is None for slot in slots)
        if (not args.screenshot) or last:
            _draw_evolution(dashboard, slots)
        _pace(dashboard, args, 30)
        if last and args.screenshot:
            break


def _draw_evolution(dashboard, slots) -> None:
    dashboard.screen.fill(BG)
    dashboard.screen.blit(dashboard.title_font.render("HOW THE AI LEARNED", True, GOLD), (16, 12))
    origin_x, origin_y = 16, 48
    cell_w, cell_h = 410, 240
    for index, slot in enumerate(slots):
        col = index % 3
        row = index // 3
        rect = pygame.Rect(origin_x + col * (cell_w + 12), origin_y + row * (cell_h + 12), cell_w, cell_h)
        pygame.draw.rect(dashboard.screen, PANEL, rect, border_radius=6)
        env = slot["env"]
        view = WorldView(rect.inflate(-8, -28), env.track)
        draw_track(dashboard.screen, env.track, view)
        if slot["status"] != "no checkpoint":
            draw_polyline(dashboard.screen, [(sample["x"], sample["y"]) for sample in env.trace], view, (255, 120, 96), 2)
            draw_car(dashboard.screen, env.car.state, view)
        label = f"{slot['name']}  {slot['status']}"
        dashboard.screen.blit(dashboard.small.render(label, True, TEXT), (rect.x + 8, rect.bottom - 22))
    dashboard.screen.blit(
        dashboard.small.render("Empty slots stay empty until a checkpoint file exists.", True, MUTED),
        (16, HEIGHT - 28),
    )
    pygame.display.flip()


def _run_compare(dashboard, args, reward_history, loss_history) -> None:
    rows = _comparison_rows()
    if rows:
        save_comparison_bars(rows, Path("data/results/comparison.png"))
    env = RallyEnv(EnvConfig(track_seed=args.track_seed, preset=args.preset, max_steps=10))
    env.reset()
    lines = [f"{row['label']}: completion {row['completion']:.0%}  collisions {row['collisions']:.1f}" for row in rows] or [
        "No eval JSON in data/results yet."
    ]
    dashboard.frame(
        env,
        title="compare",
        algorithm="eval files",
        architecture="—",
        parameters=None,
        episode=0,
        reward_history=reward_history,
        loss_history=loss_history,
        learning_rate=0.0,
        entropy=0.0,
        observation="sensors",
        notes=[],
        extra_lines=lines,
    )


def _comparison_rows() -> list[dict]:
    rows = []
    for path in sorted(Path("data/results").glob("eval_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        summary = payload.get("summary", {})
        progress = summary.get("progress", {}).get("mean")
        collisions = summary.get("collisions", {}).get("mean")
        if progress is None:
            continue
        rows.append(
            {
                "label": payload.get("label", path.stem),
                "completion": float(summary.get("completed_rate", progress)),
                "collisions": float(collisions or 0.0),
                "lap_time": summary.get("lap_time", {}).get("mean"),
                "off_track": summary.get("off_track_fraction", {}).get("mean"),
            }
        )
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autonomous Rally AI lab window.")
    parser.add_argument("--mode", default="classical", choices=("manual", "classical", "policy", "ghosts", "evolution", "compare", "slice"))
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_mode(args)


if __name__ == "__main__":
    main()
