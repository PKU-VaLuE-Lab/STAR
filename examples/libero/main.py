import collections
import dataclasses
import logging
import math
import pathlib
import queue
import threading
import traceback

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class _VideoWriteTask:
    video_path: pathlib.Path
    frames: list[np.ndarray]
    fps: int


class _AsyncVideoWriter:
    """Episode-level async video writer for LIBERO evaluation."""

    def __init__(self, *, enabled: bool, num_workers: int = 1, queue_size: int = 4):
        self._enabled = bool(enabled)
        self._num_workers = max(1, int(num_workers))
        self._queue_size = max(1, int(queue_size))
        self._queue: queue.Queue[_VideoWriteTask | None] | None = None
        self._workers: list[threading.Thread] = []
        self._errors: queue.Queue[tuple[pathlib.Path, Exception, str]] = queue.Queue()
        self._closed = False

        if self._enabled:
            self._queue = queue.Queue(maxsize=self._queue_size)
            for idx in range(self._num_workers):
                worker = threading.Thread(
                    target=self._worker_loop,
                    name=f"libero-video-writer-{idx}",
                    daemon=True,
                )
                worker.start()
                self._workers.append(worker)

    def _worker_loop(self) -> None:
        assert self._queue is not None
        while True:
            task = self._queue.get()
            try:
                if task is None:
                    return
                imageio.mimwrite(task.video_path, [np.asarray(x) for x in task.frames], fps=task.fps)
                logging.info("Saved replay video to: %s", task.video_path.resolve())
            except Exception as exc:
                failed_path = task.video_path if task is not None else pathlib.Path("<shutdown>")
                self._errors.put((failed_path, exc, traceback.format_exc()))
            finally:
                self._queue.task_done()

    def _raise_if_failed(self) -> None:
        if self._errors.empty():
            return
        failed_path, exc, tb = self._errors.get()
        raise RuntimeError(f"Video write failed for {failed_path}: {exc}\n{tb}") from exc

    def submit(self, video_path: pathlib.Path, frames: list[np.ndarray], *, fps: int = 10) -> None:
        self._raise_if_failed()
        if not self._enabled:
            imageio.mimwrite(video_path, [np.asarray(x) for x in frames], fps=fps)
            logging.info("Saved replay video to: %s", video_path.resolve())
            return

        if self._closed:
            raise RuntimeError("Video writer is already closed.")
        assert self._queue is not None
        self._queue.put(_VideoWriteTask(video_path=video_path, frames=list(frames), fps=int(fps)))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if not self._enabled:
            self._raise_if_failed()
            return

        assert self._queue is not None
        # Finish all enqueued tasks before shutting down worker threads.
        self._queue.join()
        for _ in self._workers:
            self._queue.put(None)
        for worker in self._workers:
            worker.join()
        self._raise_if_failed()


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5
    env_control_hz: float = 10.0  # Environment control rate used to derive effective infer/scheduler frequency.

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task
    max_tasks: int | None = None  # Optional cap to run a subset of tasks for quick smoke tests.
    task_ids: str | None = None  # Optional comma-separated subset of task ids to evaluate, e.g. "0,3,7".

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos
    async_video_write: bool = True  # Write episode videos in background threads.
    video_write_workers: int = 1  # Number of background video writer threads.
    video_write_queue_size: int = 4  # Max pending episode videos before backpressure.
    reset_policy_each_episode: bool = True  # Call policy.reset() at every episode start (recommended for STAR).

    seed: int = 7  # Random Seed (for reproducibility)


def eval_libero(args: Args) -> None:
    if args.replan_steps <= 0:
        raise ValueError(f"replan_steps must be > 0, got {args.replan_steps}")
    if args.env_control_hz <= 0:
        raise ValueError(f"env_control_hz must be > 0, got {args.env_control_hz}")
    if args.video_write_workers <= 0:
        raise ValueError(f"video_write_workers must be > 0, got {args.video_write_workers}")
    if args.video_write_queue_size <= 0:
        raise ValueError(f"video_write_queue_size must be > 0, got {args.video_write_queue_size}")

    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    logging.info(f"Task suite: {args.task_suite_name}")
    selected_task_ids = _resolve_task_ids(task_suite.n_tasks, args.task_ids, args.max_tasks)
    logging.info("Evaluating task ids: %s", selected_task_ids)

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    max_steps = _resolve_max_steps(args.task_suite_name)

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    _warn_star_frequency_mismatch(args, client.get_server_metadata())
    video_writer = _AsyncVideoWriter(
        enabled=args.async_video_write,
        num_workers=args.video_write_workers,
        queue_size=args.video_write_queue_size,
    )

    # Start evaluation
    total_episodes, total_successes = 0, 0
    try:
        for task_id in tqdm.tqdm(selected_task_ids):
            # Get task
            task = task_suite.get_task(task_id)

            # Get default LIBERO initial states
            initial_states = task_suite.get_task_init_states(task_id)

            # Initialize LIBERO environment and task description
            env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

            # Start episodes
            task_episodes, task_successes = 0, 0
            for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
                logging.info(f"\nTask: {task_description}")
                if args.reset_policy_each_episode:
                    try:
                        client.reset()
                    except Exception as exc:
                        logging.warning("Policy reset failed; continuing without reset: %s", exc)

                # Reset environment
                env.reset()
                action_plan = collections.deque()

                # Set initial states
                obs = env.set_init_state(initial_states[episode_idx])

                # Setup
                t = 0
                done = False
                replay_images = []
                executed_state_history: list[np.ndarray] = []

                logging.info(f"Starting episode {task_episodes+1}...")
                while t < max_steps + args.num_steps_wait:
                    try:
                        # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                        # and we need to wait for them to fall
                        if t < args.num_steps_wait:
                            obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                            t += 1
                            continue

                        # Get preprocessed image
                        # IMPORTANT: rotate 180 degrees to match train preprocessing
                        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                        wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                        img = image_tools.convert_to_uint8(
                            image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                        )
                        wrist_img = image_tools.convert_to_uint8(
                            image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                        )

                        # Save preprocessed image for replay video
                        replay_images.append(img)

                        if not action_plan:
                            # Finished executing previous action chunk -- compute new chunk
                            # Prepare observations dict
                            proprio_state = _build_proprio_state(obs)
                            element = {
                                "observation/image": img,
                                "observation/wrist_image": wrist_img,
                                "observation/state": proprio_state,
                                "prompt": str(task_description),
                            }
                            if executed_state_history:
                                # Replay executed env-step states on server so STAR stats are
                                # updated at real control frequency even when replanning in chunks.
                                element["observation/state_history"] = np.asarray(executed_state_history)

                            # Query model to get action
                            action_chunk = client.infer(element)["actions"]
                            executed_state_history.clear()
                            assert (
                                len(action_chunk) >= args.replan_steps
                            ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                            action_plan.extend(action_chunk[: args.replan_steps])

                        action = action_plan.popleft()

                        # Execute action in environment
                        obs, reward, done, info = env.step(action.tolist())
                        executed_state_history.append(_build_proprio_state(obs))
                        if done:
                            task_successes += 1
                            total_successes += 1
                            break
                        t += 1

                    except Exception as e:
                        logging.error(f"Caught exception: {e}")
                        break

                task_episodes += 1
                total_episodes += 1

                # Save a replay video of the episode
                suffix = "success" if done else "failure"
                task_segment = task_description.replace(" ", "_")
                video_path = (
                    pathlib.Path(args.video_out_path)
                    / f"task_{task_id:02d}_episode_{episode_idx:03d}_{task_segment}_{suffix}.mp4"
                )
                video_writer.submit(video_path, replay_images, fps=10)
                logging.info(f"Queued replay video to: {video_path.resolve()}")

                # Log current results
                logging.info(f"Success: {done}")
                logging.info(f"# episodes completed so far: {total_episodes}")
                logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

            # Log final results
            logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
            logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

        logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
        logging.info(f"Total episodes: {total_episodes}")
    finally:
        video_writer.close()


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _warn_star_frequency_mismatch(args: Args, server_metadata: dict) -> None:
    effective_scheduler_hz = float(args.env_control_hz) / float(args.replan_steps)
    logging.info(
        "Effective scheduler frequency inferred from evaluation loop: %.3f Hz (= %.3f / %d).",
        effective_scheduler_hz,
        float(args.env_control_hz),
        int(args.replan_steps),
    )

    if str(server_metadata.get("inference_mode", "")).lower() != "star":
        return

    configured_hz = server_metadata.get("star_control_hz")
    if configured_hz is None:
        logging.warning(
            "STAR server metadata does not expose star_control_hz. "
            "Recommended server args: --star-replan-steps=%d --star-env-control-hz=%.3f "
            "(or set --star.control-hz=%.3f manually).",
            int(args.replan_steps),
            float(args.env_control_hz),
            effective_scheduler_hz,
        )
        return

    configured_hz = float(configured_hz)
    tolerance = max(1e-6, 0.05 * max(abs(configured_hz), abs(effective_scheduler_hz), 1.0))
    if abs(configured_hz - effective_scheduler_hz) > tolerance:
        logging.warning(
            "STAR scheduler frequency mismatch: server control_hz=%.3f, expected %.3f from env_control_hz/replan_steps. "
            "Restart server with aligned control_hz to avoid unstable phase statistics.",
            configured_hz,
            effective_scheduler_hz,
        )


def _resolve_task_ids(num_tasks_in_suite: int, task_ids_arg: str | None, max_tasks: int | None) -> list[int]:
    if task_ids_arg is None or not task_ids_arg.strip():
        selected_task_ids = list(range(num_tasks_in_suite))
        if max_tasks is not None:
            selected_task_ids = selected_task_ids[: max_tasks]
            logging.info("Limiting evaluation to first %d tasks", len(selected_task_ids))
        return selected_task_ids

    selected_task_ids = []
    for raw_task_id in task_ids_arg.split(","):
        task_id = int(raw_task_id.strip())
        if task_id < 0 or task_id >= num_tasks_in_suite:
            raise ValueError(
                f"Task id {task_id} is out of range for suite with {num_tasks_in_suite} tasks."
            )
        selected_task_ids.append(task_id)

    # Preserve the user-specified order while removing duplicates.
    selected_task_ids = list(dict.fromkeys(selected_task_ids))
    if max_tasks is not None:
        selected_task_ids = selected_task_ids[: max_tasks]
        logging.info("Limiting explicit task id selection to first %d entries", len(selected_task_ids))
    return selected_task_ids


def _resolve_max_steps(task_suite_name: str) -> int:
    """Resolve max rollout steps for both LIBERO and LIBERO-PRO suite variants."""
    suite_name = task_suite_name.lower()

    # Keep the original canonical suites first.
    if suite_name == "libero_spatial":
        return 220  # longest training demo has 193 steps
    if suite_name == "libero_object":
        return 280  # longest training demo has 254 steps
    if suite_name == "libero_goal":
        return 300  # longest training demo has 270 steps
    if suite_name == "libero_10":
        return 520  # longest training demo has 505 steps
    if suite_name == "libero_90":
        return 400  # longest training demo has 373 steps

    # LIBERO-PRO extends the suite names with suffixes such as *_temp/*_env/*_task/*_swap/*_object/*_lan.
    # Match by base suite prefix to reuse the corresponding rollout horizon.
    if "libero_10" in suite_name:
        return 520
    if "libero_spatial" in suite_name:
        return 220
    if "libero_object" in suite_name:
        return 280
    if "libero_goal" in suite_name:
        return 300
    if "libero_90" in suite_name:
        return 400

    raise ValueError(f"Unknown task suite: {task_suite_name}")


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _build_proprio_state(obs: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate(
        (
            obs["robot0_eef_pos"],
            _quat2axisangle(np.array(obs["robot0_eef_quat"], copy=True)),
            obs["robot0_gripper_qpos"],
        )
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
