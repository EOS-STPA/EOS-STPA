from __future__ import annotations
import argparse
import glob
import json
import math
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import random
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np

def append_carla_path() -> None:
    directories = [os.environ.get('CARLA_PYTHONAPI', ''), '/home/carlauser/CarlaUE4/PythonAPI/carla', '/home/carlauser/CarlaUE4/PythonAPI', '/root/CarlaUE4/PythonAPI/carla', '/root/CarlaUE4/PythonAPI', './CARLA_0.9.14/PythonAPI/carla', './CARLA_0.9.14/PythonAPI']
    for directory in directories:
        if directory and os.path.exists(directory) and (directory not in sys.path):
            sys.path.append(directory)
    patterns = [os.environ.get('CARLA_EGG', ''), '/home/carlauser/CarlaUE4/PythonAPI/carla/dist/carla-*linux-x86_64.egg', '/root/CarlaUE4/PythonAPI/carla/dist/carla-*linux-x86_64.egg', './CARLA_0.9.14/PythonAPI/carla/dist/carla-*linux-x86_64.egg', '/home/carlauser/CarlaUE4/PythonAPI/carla/dist/carla-*.whl', '/root/CarlaUE4/PythonAPI/carla/dist/carla-*.whl', './CARLA_0.9.14/PythonAPI/carla/dist/carla-*.whl']
    for pattern in patterns:
        if not pattern:
            continue
        for match in glob.glob(pattern):
            if match not in sys.path:
                sys.path.append(match)
append_carla_path()
try:
    import carla
except Exception as exc:
    raise RuntimeError('Cannot import the CARLA Python API. Set PYTHONPATH or CARLA_EGG first.\nExample:\nexport PYTHONPATH=$PYTHONPATH:/home/carlauser/CarlaUE4/PythonAPI/carla:/home/carlauser/CarlaUE4/PythonAPI/carla/dist/carla-0.9.14-py3.7-linux-x86_64.egg') from exc
try:
    import stable_baselines3
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.utils import set_random_seed
except Exception as exc:
    raise RuntimeError('stable-baselines3 is not installed. For Python >= 3.8, install:\npip install stable-baselines3==2.3.2 gymnasium==0.29.1 tensorboard\nFor an older Python/CARLA environment, install a compatible SB3/Gym pair.') from exc
try:
    SB3_MAJOR = int(stable_baselines3.__version__.split('.')[0])
except Exception:
    SB3_MAJOR = 2
if SB3_MAJOR >= 2:
    try:
        import gymnasium as gym
        from gymnasium import spaces
    except Exception as exc:
        raise RuntimeError('SB3 2.x requires gymnasium. Install gymnasium==0.29.1.') from exc
    USE_GYMNASIUM_API = True
else:
    try:
        import gym
        from gym import spaces
    except Exception as exc:
        raise RuntimeError('SB3 1.x requires gym. Install a compatible gym release.') from exc
    USE_GYMNASIUM_API = False
try:
    import torch
except Exception as exc:
    raise RuntimeError('PyTorch is required by Stable-Baselines3.') from exc

@dataclass
class CarlaConfig:
    host: str = '127.0.0.1'
    port: int = 2000
    timeout_seconds: float = 60.0
    town: str = 'Town03'
    fixed_delta_seconds: float = 0.05
    no_rendering_mode: bool = True
    route_traffic_light_id: Optional[int] = None

@dataclass
class ScenarioConfig:
    weather_set: List[str] = field(default_factory=lambda: ['clear', 'cloudy', 'foggy', 'rainy'])
    ego_initial_distance_to_stop_m: float = 72.5
    lead_initial_distance_to_stop_m: float = 18.0
    initial_speed_mps: float = 12.0
    lead_braking_duration_s: float = 3.0
    red_duration_min_s: float = 6.0
    red_duration_max_s: float = 10.0
    goal_distance_after_stop_m: float = 25.0
    route_step_m: float = 1.0
    max_episode_seconds: float = 30.0
    speed_limit_fallback_kmh: float = 60.0
    speed_limit_override_kmh: Optional[float] = 60.0
    vehicle_filter: str = 'vehicle.tesla.model3'
    max_target_speed_ratio: float = 1.1
    steering_residual_limit: float = 0.16
    severe_offroute_threshold_m: float = 4.0
    collision_terminates: bool = True
    lateral_initial_offset_m: float = 0.3
    initial_yaw_error_deg: float = 2.0
    adverse_lateral_disturbance: bool = True
    hard_safety_layer_enabled: bool = True
    eos_operational_layer_enabled: bool = True
    hard_min_gap_m: float = 6.0
    hard_ttc_threshold_s: float = 1.35
    hard_deceleration_mps2: float = 7.0
    eos_collision_guard_enabled: bool = True
    eos_collision_guard_min_gap_m: float = 9.0
    eos_collision_guard_time_headway_s: float = 1.35
    eos_collision_guard_ttc_threshold_s: float = 2.6
    eos_collision_guard_deceleration_mps2: float = 8.5
    eos_collision_guard_min_brake: float = 0.18
    eos_collision_guard_max_brake: float = 0.92
    eos_collision_guard_emergency_ttc_s: float = 0.95
    eos_collision_guard_emergency_gap_m: float = 5.5
    stpa_time_headway_s: float = 0.65
    eos_time_headway_s: float = 3.0
    stpa_gap_buffer_m: float = 1.5
    eos_gap_buffer_m: float = 9.0
    stpa_following_preview_m: float = 4.0
    eos_following_preview_m: float = 18.0
    stpa_gap_control_gain: float = 0.8
    eos_gap_control_gain: float = 0.28
    stpa_accel_limit_mps2: float = 3.0
    stpa_decel_limit_mps2: float = 5.5
    eos_accel_limit_mps2: float = 1.4
    eos_decel_limit_mps2: float = 2.4
    stpa_target_jerk_limit_mps3: float = 8.0
    eos_target_jerk_limit_mps3: float = 0.45
    hard_target_jerk_limit_mps3: float = 20.0
    acceleration_filter_alpha: float = 0.12
    jerk_filter_alpha: float = 0.08
    lateral_recovery_threshold_m: float = 2.3
    lateral_recovery_speed_mps: float = 4.0

@dataclass
class RewardConfig:
    omega_time: float = 0.1
    omega_speed: float = 0.05
    collision_penalty: float = -10.0
    comfort_good: float = 1.0
    comfort_bad: float = -1.0
    comfort_jerk_threshold_mps3: float = 0.6
    red_light_penalty: float = -5.0
    dangerous_following_penalty: float = -2.0
    lane_keeping_reward: float = 1.0
    adverse_weather_speed_penalty: float = -1.5
    perception_failure_penalty: float = -1.0
    incremental_time_reward: bool = True

@dataclass
class PPOConfig:
    total_timesteps: int = 100000
    learning_rate: float = 0.0003
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    ent_coef: float = 0.0
    policy_hidden_sizes: List[int] = field(default_factory=lambda: [64, 64])
    checkpoint_frequency: int = 20000

@dataclass
class ExperimentConfig:
    training_seeds: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    output_dir: str = '/root/autodl-tmp/yolo_unzipped/yolo/ESWA'

@dataclass
class Config:
    carla: CarlaConfig = field(default_factory=CarlaConfig)
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))

def wrap_angle_rad(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi

def speed_mps(actor: Any) -> float:
    velocity = actor.get_velocity()
    return math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)

def yaw_to_unit(yaw_deg: float) -> Tuple[float, float]:
    yaw = math.radians(yaw_deg)
    return (math.cos(yaw), math.sin(yaw))

def right_unit(yaw_deg: float) -> Tuple[float, float]:
    yaw = math.radians(yaw_deg)
    return (-math.sin(yaw), math.cos(yaw))

def parse_int_spec(text: str) -> List[int]:
    values: List[int] = []
    for chunk in text.split(','):
        chunk = chunk.strip()
        if not chunk:
            continue
        if '-' in chunk[1:]:
            left, right = chunk.split('-', 1)
            start = int(left)
            end = int(right)
            step = 1 if end >= start else -1
            values.extend(range(start, end + step, step))
        else:
            values.append(int(chunk))
    if not values:
        raise ValueError('No integer values were parsed')
    return values

def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    set_random_seed(seed)
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

@dataclass(frozen=True)
class RouteProjection:
    progress_m: float
    lateral_offset_m: float
    route_yaw_rad: float
    distance_to_route_m: float

class RoutePolyline:

    def __init__(self, xyz: np.ndarray):
        if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) < 2:
            raise ValueError('A route requires at least two 3-D points')
        self.xyz = xyz.astype(np.float64)
        delta = self.xyz[1:, :2] - self.xyz[:-1, :2]
        segment_length = np.linalg.norm(delta, axis=1)
        if np.any(segment_length < 1e-05):
            raise ValueError('The route contains duplicate consecutive points')
        self.segment_vector = delta
        self.segment_length = segment_length
        self.cumulative = np.concatenate(([0.0], np.cumsum(segment_length)))
        self.length_m = float(self.cumulative[-1])

    @classmethod
    def from_waypoints(cls, waypoints: Sequence[Any]) -> 'RoutePolyline':
        xyz = np.array([[wp.transform.location.x, wp.transform.location.y, wp.transform.location.z] for wp in waypoints], dtype=np.float64)
        keep = [0]
        for index in range(1, len(xyz)):
            if np.linalg.norm(xyz[index, :2] - xyz[keep[-1], :2]) > 0.05:
                keep.append(index)
        return cls(xyz[keep])

    def project(self, location: Any) -> RouteProjection:
        point = np.array([location.x, location.y], dtype=np.float64)
        start = self.xyz[:-1, :2]
        segment = self.segment_vector
        relative = point - start
        t = np.sum(relative * segment, axis=1) / np.maximum(self.segment_length ** 2, 1e-09)
        t = np.clip(t, 0.0, 1.0)
        closest = start + t[:, None] * segment
        difference = point - closest
        squared_distance = np.sum(difference ** 2, axis=1)
        index = int(np.argmin(squared_distance))
        tangent = segment[index] / self.segment_length[index]
        cross_z = tangent[0] * difference[index, 1] - tangent[1] * difference[index, 0]
        magnitude = math.sqrt(float(squared_distance[index]))
        signed_offset = math.copysign(magnitude, cross_z if abs(cross_z) > 1e-12 else 1.0)
        progress = float(self.cumulative[index] + t[index] * self.segment_length[index])
        yaw = math.atan2(tangent[1], tangent[0])
        return RouteProjection(progress, signed_offset, yaw, magnitude)

    def sample(self, progress_m: float) -> Tuple[np.ndarray, float]:
        distance = float(np.clip(progress_m, 0.0, self.length_m))
        index = int(np.searchsorted(self.cumulative, distance, side='right') - 1)
        index = min(max(index, 0), len(self.segment_length) - 1)
        local = (distance - self.cumulative[index]) / self.segment_length[index]
        xyz = self.xyz[index] + local * (self.xyz[index + 1] - self.xyz[index])
        yaw = math.atan2(self.segment_vector[index, 1], self.segment_vector[index, 0])
        return (xyz, yaw)

def angle_difference_deg(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)

def select_straight(current: Any, candidates: Iterable[Any]) -> Optional[Any]:
    candidate_list = [candidate for candidate in candidates if candidate is not None]
    if not candidate_list:
        return None
    current_yaw = current.transform.rotation.yaw
    candidate_list.sort(key=lambda wp: (angle_difference_deg(wp.transform.rotation.yaw, current_yaw), 0 if wp.lane_type == current.lane_type else 1, abs(wp.lane_id - current.lane_id)))
    return candidate_list[0]

def walk_waypoint(start_waypoint: Any, distance_m: float, forward: bool, step_m: float) -> Optional[Any]:
    current = start_waypoint
    travelled = 0.0
    while travelled + 1e-06 < distance_m:
        step = min(step_m, distance_m - travelled)
        candidates = current.next(step) if forward else current.previous(step)
        next_waypoint = select_straight(current, candidates)
        if next_waypoint is None:
            return None
        current = next_waypoint
        travelled += step
    return current

def build_forward_waypoints(start_waypoint: Any, length_m: float, step_m: float) -> List[Any]:
    points = [start_waypoint]
    current = start_waypoint
    travelled = 0.0
    while travelled + 1e-06 < length_m:
        step = min(step_m, length_m - travelled)
        next_waypoint = select_straight(current, current.next(step))
        if next_waypoint is None:
            break
        points.append(next_waypoint)
        current = next_waypoint
        travelled += step
    return points

@dataclass
class SignalizedRoute:
    route: RoutePolyline
    traffic_light: Any
    stop_progress_m: float
    ego_start_progress_m: float
    lead_start_progress_m: float
    goal_progress_m: float

def discover_signalized_route(world: Any, scenario: ScenarioConfig, requested_light_id: Optional[int]) -> SignalizedRoute:
    lights = list(world.get_actors().filter('traffic.traffic_light*'))
    lights.sort(key=lambda actor: actor.id)
    if requested_light_id is not None:
        lights = [light for light in lights if int(light.id) == int(requested_light_id)]
        if not lights:
            raise RuntimeError('Traffic light actor id %s was not found' % requested_light_id)
    required_back = scenario.ego_initial_distance_to_stop_m
    required_forward = scenario.goal_distance_after_stop_m + 10.0
    candidates: List[Tuple[Any, ...]] = []
    for light in lights:
        try:
            stop_waypoints = list(light.get_stop_waypoints())
        except Exception:
            continue
        stop_waypoints.sort(key=lambda wp: (wp.road_id, wp.section_id, wp.lane_id, wp.s))
        for stop_waypoint in stop_waypoints:
            start_waypoint = walk_waypoint(stop_waypoint, required_back, forward=False, step_m=scenario.route_step_m)
            if start_waypoint is None:
                continue
            waypoints = build_forward_waypoints(start_waypoint, required_back + required_forward, scenario.route_step_m)
            minimum_points = int(0.75 * (required_back + required_forward) / scenario.route_step_m)
            if len(waypoints) < minimum_points:
                continue
            try:
                route = RoutePolyline.from_waypoints(waypoints)
            except ValueError:
                continue
            stop_projection = route.project(stop_waypoint.transform.location)
            if stop_projection.progress_m < required_back * 0.8:
                continue
            goal_progress = min(route.length_m - 1.0, stop_projection.progress_m + scenario.goal_distance_after_stop_m)
            if goal_progress <= stop_projection.progress_m + 5.0:
                continue
            score = (abs(stop_projection.progress_m - required_back), 0 if not start_waypoint.is_junction else 1, light.id, stop_waypoint.road_id, stop_waypoint.lane_id)
            candidates.append((score, route, light, stop_projection.progress_m))
    if not candidates:
        raise RuntimeError('No suitable signalized route was found in Town03. Use --route-light-id after checking available traffic-light actors, or verify that Town03 is loaded.')
    candidates.sort(key=lambda item: item[0])
    _, route, light, stop_progress = candidates[0]
    ego_progress = max(0.0, stop_progress - scenario.ego_initial_distance_to_stop_m)
    lead_progress = max(ego_progress + 10.0, stop_progress - scenario.lead_initial_distance_to_stop_m)
    goal_progress = min(route.length_m - 1.0, stop_progress + scenario.goal_distance_after_stop_m)
    return SignalizedRoute(route=route, traffic_light=light, stop_progress_m=float(stop_progress), ego_start_progress_m=float(ego_progress), lead_start_progress_m=float(lead_progress), goal_progress_m=float(goal_progress))

@dataclass
class PIDController:
    kp: float
    ki: float
    kd: float
    integral_limit: float
    previous_error: float = 0.0
    integral: float = 0.0

    def reset(self) -> None:
        self.previous_error = 0.0
        self.integral = 0.0

    def step(self, error: float, dt: float) -> float:
        self.integral = clamp(self.integral + error * dt, -self.integral_limit, self.integral_limit)
        derivative = (error - self.previous_error) / max(dt, 1e-06)
        self.previous_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative

class VehicleController:

    def __init__(self, dt: float):
        self.dt = dt
        self.longitudinal = PIDController(0.35, 0.04, 0.02, 10.0)

    def reset(self) -> None:
        self.longitudinal.reset()

    def control(self, vehicle: Any, route: RoutePolyline, projection: RouteProjection, current_speed_mps: float, target_speed_mps: float, steering_residual: float, disturbance: float=0.0) -> Any:
        speed_error = target_speed_mps - current_speed_mps
        longitudinal_command = self.longitudinal.step(speed_error, self.dt)
        if longitudinal_command >= 0.0:
            throttle = clamp(longitudinal_command, 0.0, 0.85)
            brake = 0.0
        else:
            throttle = 0.0
            brake = clamp(-longitudinal_command, 0.0, 1.0)
        lookahead = clamp(5.0 + 0.55 * current_speed_mps, 5.0, 14.0)
        target_xyz, _ = route.sample(projection.progress_m + lookahead)
        transform = vehicle.get_transform()
        dx = float(target_xyz[0] - transform.location.x)
        dy = float(target_xyz[1] - transform.location.y)
        target_heading = math.atan2(dy, dx)
        current_heading = math.radians(transform.rotation.yaw)
        alpha = wrap_angle_rad(target_heading - current_heading)
        wheelbase = 2.8
        steering_angle = math.atan2(2.0 * wheelbase * math.sin(alpha), max(lookahead, 1.0))
        base_steer = steering_angle / math.radians(35.0)
        steer = clamp(base_steer + steering_residual + disturbance, -1.0, 1.0)
        return carla.VehicleControl(throttle=throttle, brake=brake, steer=steer)
LIGHT_RED = 0
LIGHT_YELLOW = 1
LIGHT_GREEN = 2
LIGHT_UNKNOWN = 3

@dataclass(frozen=True)
class PerceptionOutput:
    lead_distance_m: float
    relative_speed_mps: float
    light_state: int
    confidence: float
    failure: bool
WEATHER_DEGRADATION: Dict[str, Dict[str, float]] = {'clear': {'dropout': 0.002, 'wrong': 0.002, 'distance_sigma': 0.15, 'fault_scale': 0.25}, 'cloudy': {'dropout': 0.008, 'wrong': 0.006, 'distance_sigma': 0.35, 'fault_scale': 0.55}, 'foggy': {'dropout': 0.05, 'wrong': 0.025, 'distance_sigma': 1.5, 'fault_scale': 1.0}, 'rainy': {'dropout': 0.035, 'wrong': 0.02, 'distance_sigma': 1.0, 'fault_scale': 0.9}}

class PerceptionDegrader:

    def __init__(self, rng: np.random.Generator, weather: str, red_duration_s: float):
        self.rng = rng
        self.weather = weather
        self.red_duration_s = red_duration_s
        self.parameters = WEATHER_DEGRADATION[weather]
        self.last_distance = 80.0
        self.last_relative_speed = 0.0
        self.signal_fault = self._sample_signal_fault()
        scale = self.parameters['fault_scale']
        self.red_delay_s = float(self.rng.uniform(0.35, 1.2) * scale)
        self.early_green_s = float(self.rng.uniform(0.35, 1.2) * scale)

    def _sample_signal_fault(self) -> str:
        scale = self.parameters['fault_scale']
        p_delayed = 0.1 * scale
        p_missing = 0.08 * scale
        p_early = 0.1 * scale
        draw = float(self.rng.random())
        if draw < p_delayed:
            return 'delayed_red'
        if draw < p_delayed + p_missing:
            return 'missing_red'
        if draw < p_delayed + p_missing + p_early:
            return 'early_green'
        return 'normal'

    def observe(self, simulation_time_s: float, actual_light_state: int, true_lead_distance_m: float, true_relative_speed_mps: float) -> PerceptionOutput:
        failure = False
        confidence = 1.0
        dropout = bool(self.rng.random() < self.parameters['dropout'])
        wrong = bool(self.rng.random() < self.parameters['wrong'])
        if dropout:
            lead_distance = self.last_distance
            relative_speed = self.last_relative_speed
            failure = True
            confidence = 0.0
        else:
            sigma = self.parameters['distance_sigma'] * (3.0 if wrong else 1.0)
            lead_distance = max(0.0, true_lead_distance_m + float(self.rng.normal(0.0, sigma)))
            relative_speed = true_relative_speed_mps + float(self.rng.normal(0.0, 0.25 * sigma))
            self.last_distance = lead_distance
            self.last_relative_speed = relative_speed
            if wrong:
                failure = True
                confidence = 0.25
            else:
                confidence = max(0.45, 1.0 - sigma / 4.0)
        perceived_light = actual_light_state
        if self.signal_fault == 'delayed_red' and actual_light_state == LIGHT_RED and (simulation_time_s < self.red_delay_s):
            perceived_light = LIGHT_GREEN
            failure = True
            confidence = min(confidence, 0.25)
        elif self.signal_fault == 'missing_red' and actual_light_state == LIGHT_RED:
            perceived_light = LIGHT_UNKNOWN
            failure = True
            confidence = min(confidence, 0.1)
        elif self.signal_fault == 'early_green' and actual_light_state == LIGHT_RED and (0.0 < self.red_duration_s - simulation_time_s <= self.early_green_s):
            perceived_light = LIGHT_GREEN
            failure = True
            confidence = min(confidence, 0.25)
        return PerceptionOutput(lead_distance_m=float(lead_distance), relative_speed_mps=float(relative_speed), light_state=int(perceived_light), confidence=float(confidence), failure=bool(failure))

@dataclass(frozen=True)
class RewardInputs:
    dt_s: float
    elapsed_time_s: float
    speed_mps: float
    speed_limit_mps: float
    safe_weather_speed_mps: float
    jerk_mps3: float
    lateral_offset_m: float
    collision_event: bool
    red_light_violation_event: bool
    dangerous_following: bool
    adverse_weather: bool
    perception_failure: bool

def safe_weather_speed_kmh(speed_limit_kmh: float) -> float:
    if speed_limit_kmh <= 50.0:
        return speed_limit_kmh
    if speed_limit_kmh <= 80.0:
        return 0.8 * speed_limit_kmh
    return 65.0

def compute_reward(strategy: str, inputs: RewardInputs, cfg: RewardConfig) -> Tuple[float, Dict[str, float]]:
    normalized = strategy.upper().replace('_', '-')
    if normalized not in {'STPA', 'EOS-STPA'}:
        raise ValueError('Unknown reward strategy: %s' % strategy)
    collision_reward = cfg.collision_penalty if inputs.collision_event else 0.0
    if cfg.incremental_time_reward:
        time_reward = -inputs.dt_s
    else:
        time_reward = -inputs.elapsed_time_s
    speed_reward = -abs(inputs.speed_mps - inputs.speed_limit_mps)
    efficiency_reward = cfg.omega_time * time_reward + cfg.omega_speed * speed_reward
    comfort_reward = cfg.comfort_good if abs(inputs.jerk_mps3) < cfg.comfort_jerk_threshold_mps3 else cfg.comfort_bad
    red_reward = cfg.red_light_penalty if inputs.red_light_violation_event else 0.0
    following_reward = cfg.dangerous_following_penalty if inputs.dangerous_following else 0.0
    components = {'rc': float(collision_reward), 'rtime': float(time_reward), 'rspeed': float(speed_reward), 're': float(efficiency_reward), 'rcom': float(comfort_reward), 'rred': float(red_reward), 'rf': float(following_reward), 'rl': 0.0, 'rw': 0.0, 'rp': 0.0}
    total = collision_reward + efficiency_reward + comfort_reward + red_reward + following_reward
    if normalized == 'EOS-STPA':
        lane_reward = cfg.lane_keeping_reward if abs(inputs.lateral_offset_m) < 1.5 else 0.0
        weather_reward = cfg.adverse_weather_speed_penalty if inputs.adverse_weather and inputs.speed_mps > inputs.safe_weather_speed_mps else 0.0
        perception_reward = cfg.perception_failure_penalty if inputs.perception_failure else 0.0
        components.update({'rl': float(lane_reward), 'rw': float(weather_reward), 'rp': float(perception_reward)})
        total += lane_reward + weather_reward + perception_reward
    components['total'] = float(total)
    return (float(total), components)

class CollisionCounter:

    def __init__(self):
        self.lock = threading.Lock()
        self.count = 0

    def callback(self, _event: Any) -> None:
        with self.lock:
            self.count += 1

    def consume(self) -> int:
        with self.lock:
            value = self.count
            self.count = 0
        return value

class EosStpaCarlaEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, config: Config, reward_strategy: str, seed: Optional[int]=None):
        super().__init__()
        self.config = config
        self.reward_strategy = reward_strategy
        self.client = carla.Client(config.carla.host, config.carla.port)
        self.client.set_timeout(config.carla.timeout_seconds)
        self.world = self._load_world()
        self.original_settings = self.world.get_settings()
        self._configure_world()
        self.route_info = discover_signalized_route(self.world, config.scenario, config.carla.route_traffic_light_id)
        try:
            self.world.freeze_all_traffic_lights(True)
        except Exception:
            pass
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(23,), dtype=np.float32)
        self.ego_controller = VehicleController(config.carla.fixed_delta_seconds)
        self.lead_controller = VehicleController(config.carla.fixed_delta_seconds)
        self.ego = None
        self.lead = None
        self.collision_sensor = None
        self.actor_list: List[Any] = []
        self.collision_counter = CollisionCounter()
        self.perception: Optional[PerceptionDegrader] = None
        self.weather_name = 'clear'
        self.adverse_weather = False
        self.speed_limit_kmh = config.scenario.speed_limit_fallback_kmh
        self.speed_limit_mps = self.speed_limit_kmh / 3.6
        self.safe_weather_speed_mps = safe_weather_speed_kmh(self.speed_limit_kmh) / 3.6
        self.red_duration_s = 8.0
        self.simulation_time_s = 0.0
        self.episode_steps = 0
        self.previous_speed_mps = config.scenario.initial_speed_mps
        self.previous_acceleration_mps2 = 0.0
        self.filtered_acceleration_mps2 = 0.0
        self.filtered_jerk_mps3 = 0.0
        self.previous_commanded_target_speed_mps = config.scenario.initial_speed_mps
        self.previous_commanded_target_acceleration_mps2 = 0.0
        self.previous_progress_m = self.route_info.ego_start_progress_m
        self.previous_target_fraction = 0.5
        self.red_violation_recorded = False
        self.last_safety_layer_info: Dict[str, Any] = {}
        self.current_actual_light_state = LIGHT_RED
        self.crosswind_phase = 0.0
        self.max_episode_steps = int(round(config.scenario.max_episode_seconds / config.carla.fixed_delta_seconds))
        self._legacy_seed: Optional[int] = seed
        if seed is not None and (not USE_GYMNASIUM_API):
            self.seed(seed)

    def seed(self, seed: Optional[int]=None):
        self._legacy_seed = seed
        self.np_random = np.random.default_rng(seed)
        return [seed]

    def route_metadata(self) -> Dict[str, float]:
        stop_xyz, _ = self.route_info.route.sample(self.route_info.stop_progress_m)
        start_xyz, _ = self.route_info.route.sample(self.route_info.ego_start_progress_m)
        goal_xyz, _ = self.route_info.route.sample(self.route_info.goal_progress_m)
        return {'traffic_light_actor_id': int(self.route_info.traffic_light.id), 'route_length_m': float(self.route_info.route.length_m), 'stop_progress_m': float(self.route_info.stop_progress_m), 'ego_start_progress_m': float(self.route_info.ego_start_progress_m), 'lead_start_progress_m': float(self.route_info.lead_start_progress_m), 'goal_progress_m': float(self.route_info.goal_progress_m), 'start_x': float(start_xyz[0]), 'start_y': float(start_xyz[1]), 'stop_x': float(stop_xyz[0]), 'stop_y': float(stop_xyz[1]), 'goal_x': float(goal_xyz[0]), 'goal_y': float(goal_xyz[1])}

    def _load_world(self):
        world = self.client.get_world()
        current_name = world.get_map().name.split('/')[-1]
        if current_name != self.config.carla.town:
            print('[INFO] Loading %s ...' % self.config.carla.town)
            world = self.client.load_world(self.config.carla.town)
            time.sleep(2.0)
        return world

    def _configure_world(self) -> None:
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.config.carla.fixed_delta_seconds
        settings.no_rendering_mode = self.config.carla.no_rendering_mode
        try:
            settings.substepping = True
            settings.max_substep_delta_time = 0.01
            settings.max_substeps = 10
        except Exception:
            pass
        self.world.apply_settings(settings)
        for _ in range(5):
            self.world.tick()

    def _destroy_actors(self) -> None:
        for actor in reversed(self.actor_list):
            try:
                if actor is not None and actor.is_alive:
                    if hasattr(actor, 'stop'):
                        actor.stop()
                    actor.destroy()
            except Exception:
                pass
        self.actor_list.clear()
        self.ego = None
        self.lead = None
        self.collision_sensor = None
        for _ in range(2):
            try:
                self.world.tick()
            except Exception:
                break

    def _weather_parameters(self, name: str):
        if name == 'clear':
            return carla.WeatherParameters.ClearNoon
        if name == 'cloudy':
            return carla.WeatherParameters.CloudyNoon
        if name == 'rainy':
            return carla.WeatherParameters.HardRainNoon
        if name == 'foggy':
            weather = carla.WeatherParameters.CloudyNoon
            weather.fog_density = 75.0
            weather.fog_distance = 20.0
            weather.fog_falloff = 1.0
            return weather
        raise ValueError('Unsupported weather profile: %s' % name)

    def _set_traffic_light_state(self, state: int) -> None:
        light = self.route_info.traffic_light
        desired = carla.TrafficLightState.Red if state == LIGHT_RED else carla.TrafficLightState.Green
        try:
            if state == self.current_actual_light_state and light.get_state() == desired:
                return
        except Exception:
            pass
        try:
            for grouped_light in light.get_group_traffic_lights():
                grouped_light.set_state(carla.TrafficLightState.Red)
            light.set_state(desired)
        except Exception:
            light.set_state(desired)
        self.current_actual_light_state = state

    def _transform_at(self, progress_m: float, lateral_m: float=0.0, yaw_error_deg: float=0.0):
        xyz, yaw = self.route_info.route.sample(progress_m)
        right_x, right_y = right_unit(math.degrees(yaw))
        location = carla.Location(x=float(xyz[0] + lateral_m * right_x), y=float(xyz[1] + lateral_m * right_y), z=float(xyz[2] + 0.45))
        rotation = carla.Rotation(yaw=math.degrees(yaw) + yaw_error_deg)
        return carla.Transform(location, rotation)

    def _spawn_vehicle(self, transform: Any, role_name: str, color: str):
        blueprints = list(self.world.get_blueprint_library().filter(self.config.scenario.vehicle_filter))
        if not blueprints:
            blueprints = list(self.world.get_blueprint_library().filter('vehicle.*'))
        if not blueprints:
            raise RuntimeError('No vehicle blueprint is available')
        blueprint = blueprints[0]
        if blueprint.has_attribute('role_name'):
            blueprint.set_attribute('role_name', role_name)
        if blueprint.has_attribute('color'):
            blueprint.set_attribute('color', color)
        actor = self.world.try_spawn_actor(blueprint, transform)
        if actor is None:
            for dz in (0.8, 1.2, 1.8):
                shifted = carla.Transform(carla.Location(transform.location.x, transform.location.y, transform.location.z + dz), transform.rotation)
                actor = self.world.try_spawn_actor(blueprint, shifted)
                if actor is not None:
                    break
        if actor is None:
            raise RuntimeError('Failed to spawn %s. Verify that the route is clear.' % role_name)
        actor.set_autopilot(False)
        self.actor_list.append(actor)
        return actor

    def _spawn_collision_sensor(self) -> None:
        blueprint = self.world.get_blueprint_library().find('sensor.other.collision')
        sensor = self.world.spawn_actor(blueprint, carla.Transform(), attach_to=self.ego)
        sensor.listen(self.collision_counter.callback)
        self.collision_sensor = sensor
        self.actor_list.append(sensor)

    def _set_initial_velocity(self, vehicle: Any, progress_m: float, speed: float) -> None:
        _, yaw = self.route_info.route.sample(progress_m)
        forward_x, forward_y = yaw_to_unit(math.degrees(yaw))
        vehicle.set_target_velocity(carla.Vector3D(x=forward_x * speed, y=forward_y * speed, z=0.0))

    def _sample_speed_limit(self) -> None:
        override = self.config.scenario.speed_limit_override_kmh
        if override is not None:
            self.speed_limit_kmh = float(override)
        else:
            reported = float(self.ego.get_speed_limit()) if self.ego is not None else 0.0
            self.speed_limit_kmh = reported if reported > 1.0 else self.config.scenario.speed_limit_fallback_kmh
        self.speed_limit_mps = self.speed_limit_kmh / 3.6
        self.safe_weather_speed_mps = safe_weather_speed_kmh(self.speed_limit_kmh) / 3.6

    def _reset_internal(self, seed: Optional[int]=None):
        if USE_GYMNASIUM_API:
            super().reset(seed=seed)
        elif seed is not None:
            self.seed(seed)
        elif not hasattr(self, 'np_random'):
            self.seed(self._legacy_seed)
        self._destroy_actors()
        self.ego_controller.reset()
        self.lead_controller.reset()
        self.weather_name = str(self.np_random.choice(self.config.scenario.weather_set))
        self.adverse_weather = self.weather_name in {'foggy', 'rainy'}
        self.world.set_weather(self._weather_parameters(self.weather_name))
        self.red_duration_s = float(self.np_random.uniform(self.config.scenario.red_duration_min_s, self.config.scenario.red_duration_max_s))
        self.current_actual_light_state = LIGHT_GREEN
        self._set_traffic_light_state(LIGHT_RED)
        lateral_offset = float(self.np_random.uniform(-self.config.scenario.lateral_initial_offset_m, self.config.scenario.lateral_initial_offset_m))
        yaw_error = float(self.np_random.uniform(-self.config.scenario.initial_yaw_error_deg, self.config.scenario.initial_yaw_error_deg))
        ego_spawn_progress = self.route_info.ego_start_progress_m
        lead_spawn_progress = self.route_info.lead_start_progress_m
        ego_initial_speed = self.config.scenario.initial_speed_mps
        lead_initial_speed = self.config.scenario.initial_speed_mps
        self.ego = self._spawn_vehicle(self._transform_at(ego_spawn_progress, lateral_offset, yaw_error), 'hero', '0,0,255')
        self.lead = self._spawn_vehicle(self._transform_at(lead_spawn_progress), 'lead', '255,128,0')
        self._spawn_collision_sensor()
        self._set_initial_velocity(self.ego, ego_spawn_progress, ego_initial_speed)
        self._set_initial_velocity(self.lead, lead_spawn_progress, lead_initial_speed)
        self.world.tick()
        self._sample_speed_limit()
        perception_seed = int(self.np_random.integers(0, 2 ** 31 - 1))
        self.perception = PerceptionDegrader(np.random.default_rng(perception_seed), self.weather_name, self.red_duration_s)
        self.simulation_time_s = 0.0
        self.episode_steps = 0
        self.previous_speed_mps = speed_mps(self.ego)
        self.previous_acceleration_mps2 = 0.0
        self.filtered_acceleration_mps2 = 0.0
        self.filtered_jerk_mps3 = 0.0
        self.previous_commanded_target_speed_mps = float(ego_initial_speed)
        self.previous_commanded_target_acceleration_mps2 = 0.0
        self.previous_progress_m = self.route_info.route.project(self.ego.get_location()).progress_m
        self.previous_target_fraction = 0.5
        self.red_violation_recorded = False
        self.last_safety_layer_info = {}
        self.crosswind_phase = float(self.np_random.uniform(0.0, 2.0 * math.pi))
        self.collision_counter.consume()
        observation, state = self._observation_and_state()
        info = self._base_info(state)
        return (observation, info)

    def reset(self, *, seed: Optional[int]=None, options: Optional[Dict]=None):
        del options
        observation, info = self._reset_internal(seed)
        if USE_GYMNASIUM_API:
            return (observation, info)
        return observation

    def _lead_target_speed(self) -> float:
        elapsed = self.simulation_time_s
        duration = self.config.scenario.lead_braking_duration_s
        initial = self.config.scenario.initial_speed_mps
        if elapsed <= duration:
            amplitude = -initial
            return initial + 0.5 * amplitude * (1.0 - math.cos(math.pi * elapsed / duration))
        if elapsed < self.red_duration_s:
            return 0.0
        return self.speed_limit_mps

    def _crosswind_disturbance(self) -> float:
        if not self.config.scenario.adverse_lateral_disturbance:
            return 0.0
        amplitudes = {'clear': 0.004, 'cloudy': 0.007, 'foggy': 0.018, 'rainy': 0.022}
        amplitude = amplitudes[self.weather_name]
        noise = float(self.np_random.normal(0.0, amplitude * 0.2))
        return amplitude * math.sin(0.55 * self.simulation_time_s + self.crosswind_phase) + noise

    def _is_eos_strategy(self) -> bool:
        return self.reward_strategy.upper().replace('_', '-') == 'EOS-STPA'

    def _apply_longitudinal_safety_layer(self, raw_target_speed_mps: float, ego_speed_mps: float, lead_speed_mps: float, front_distance_m: float) -> Tuple[float, Dict[str, Any]]:
        cfg = self.config.scenario
        dt = self.config.carla.fixed_delta_seconds
        eos = self._is_eos_strategy()
        target = clamp(raw_target_speed_mps, 0.0, self.speed_limit_mps * cfg.max_target_speed_ratio)
        closing_speed = max(0.0, ego_speed_mps - lead_speed_mps)
        ttc = front_distance_m / closing_speed if closing_speed > 0.05 and front_distance_m > 0.0 else float('inf')
        hard_gap = cfg.hard_min_gap_m + 0.2 * ego_speed_mps
        eos_predictive_gap = hard_gap
        if eos and cfg.eos_collision_guard_enabled:
            relative_stopping_distance = max(0.0, (ego_speed_mps ** 2 - lead_speed_mps ** 2) / max(2.0 * cfg.eos_collision_guard_deceleration_mps2, 1e-06))
            eos_predictive_gap = max(hard_gap, cfg.eos_collision_guard_min_gap_m + cfg.eos_collision_guard_time_headway_s * ego_speed_mps + relative_stopping_distance)
        hard_intervention = False
        eos_predictive_intervention = False
        kinematic_limit = float('inf')
        if cfg.hard_safety_layer_enabled:
            available_gap = max(0.0, front_distance_m - hard_gap)
            kinematic_limit = math.sqrt(max(0.0, lead_speed_mps ** 2 + 2.0 * cfg.hard_deceleration_mps2 * available_gap))
            if front_distance_m <= hard_gap:
                target = 0.0
                hard_intervention = True
            elif ttc < cfg.hard_ttc_threshold_s:
                target = min(target, max(0.0, lead_speed_mps - 0.5))
                hard_intervention = True
            elif target > kinematic_limit:
                target = kinematic_limit
                hard_intervention = True
        if eos and cfg.eos_collision_guard_enabled:
            eos_guard_trigger = bool(front_distance_m < eos_predictive_gap or (math.isfinite(ttc) and ttc < cfg.eos_collision_guard_ttc_threshold_s))
            if eos_guard_trigger:
                eos_available_gap = max(0.0, front_distance_m - cfg.eos_collision_guard_min_gap_m)
                eos_kinematic_limit = math.sqrt(max(0.0, lead_speed_mps ** 2 + 2.0 * cfg.eos_collision_guard_deceleration_mps2 * eos_available_gap))
                target = min(target, eos_kinematic_limit, max(0.0, lead_speed_mps + 0.25))
                eos_predictive_intervention = True
        if eos and cfg.eos_operational_layer_enabled:
            time_headway = cfg.eos_time_headway_s
            gap_buffer = cfg.eos_gap_buffer_m
            gap_gain = cfg.eos_gap_control_gain
            desired_gap = max(self.speed_limit_mps + 3.0, gap_buffer + time_headway * ego_speed_mps)
        else:
            time_headway = cfg.stpa_time_headway_s
            gap_buffer = cfg.stpa_gap_buffer_m
            gap_gain = cfg.stpa_gap_control_gain
            desired_gap = max(cfg.hard_min_gap_m + 1.0, gap_buffer + time_headway * ego_speed_mps)
        preview_m = cfg.eos_following_preview_m if eos and cfg.eos_operational_layer_enabled else cfg.stpa_following_preview_m
        soft_intervention = False
        if front_distance_m < desired_gap + preview_m:
            gap_error = front_distance_m - desired_gap
            tracking_limit = max(0.0, lead_speed_mps + gap_gain * gap_error)
            if tracking_limit < target:
                target = tracking_limit
                soft_intervention = True
        if eos and cfg.eos_operational_layer_enabled and self.adverse_weather:
            if target > self.safe_weather_speed_mps:
                target = self.safe_weather_speed_mps
                soft_intervention = True
        if eos and cfg.eos_operational_layer_enabled:
            accel_limit = cfg.eos_accel_limit_mps2
            decel_limit = cfg.eos_decel_limit_mps2
        else:
            accel_limit = cfg.stpa_accel_limit_mps2
            decel_limit = cfg.stpa_decel_limit_mps2
        if hard_intervention:
            decel_limit = max(decel_limit, cfg.hard_deceleration_mps2)
        previous = self.previous_commanded_target_speed_mps
        previous_target_acceleration = self.previous_commanded_target_acceleration_mps2
        emergency_hard_stop = bool(hard_intervention and (front_distance_m <= hard_gap or (math.isfinite(ttc) and ttc < 0.6)))
        desired_target_acceleration = clamp((target - previous) / max(dt, 1e-06), -decel_limit, accel_limit)
        if hard_intervention:
            target_jerk_limit = cfg.hard_target_jerk_limit_mps3
        elif eos_predictive_intervention:
            target_jerk_limit = max(cfg.eos_target_jerk_limit_mps3, 4.0)
        elif eos and cfg.eos_operational_layer_enabled:
            target_jerk_limit = cfg.eos_target_jerk_limit_mps3
        else:
            target_jerk_limit = cfg.stpa_target_jerk_limit_mps3
        acceleration_delta = target_jerk_limit * dt
        commanded_target_acceleration = clamp(desired_target_acceleration, previous_target_acceleration - acceleration_delta, previous_target_acceleration + acceleration_delta)
        if emergency_hard_stop:
            commanded_target_acceleration = -cfg.hard_deceleration_mps2
            target = 0.0
        elif hard_intervention:
            commanded_target_acceleration = desired_target_acceleration
            target = previous + commanded_target_acceleration * dt
        else:
            target = previous + commanded_target_acceleration * dt
        target = clamp(target, 0.0, self.speed_limit_mps * cfg.max_target_speed_ratio)
        info = {'raw_target_speed_mps': float(raw_target_speed_mps), 'filtered_target_speed_mps': float(target), 'desired_target_acceleration_mps2': float(desired_target_acceleration), 'commanded_target_acceleration_mps2': float(commanded_target_acceleration), 'target_jerk_limit_mps3': float(target_jerk_limit), 'front_distance_m': float(front_distance_m), 'closing_speed_mps': float(closing_speed), 'ttc_s': None if not math.isfinite(ttc) else float(ttc), 'desired_gap_m': float(desired_gap), 'following_preview_m': float(preview_m), 'hard_gap_m': float(hard_gap), 'eos_predictive_gap_m': float(eos_predictive_gap), 'kinematic_limit_mps': None if not math.isfinite(kinematic_limit) else float(kinematic_limit), 'hard_intervention': bool(hard_intervention), 'eos_predictive_intervention': bool(eos_predictive_intervention), 'emergency_hard_stop': bool(emergency_hard_stop), 'soft_intervention': bool(soft_intervention), 'eos_operational_layer': bool(eos and cfg.eos_operational_layer_enabled)}
        return (float(target), info)

    def _apply_eos_direct_collision_brake(self, control: Any, ego_speed_mps: float, lead_speed_mps: float, front_distance_m: float, safety_info: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        cfg = self.config.scenario
        eos = self._is_eos_strategy()
        guard_info = {'direct_collision_brake': False, 'direct_brake_command': 0.0, 'required_collision_gap_m': 0.0, 'direct_guard_ttc_s': None}
        if not (eos and cfg.eos_collision_guard_enabled):
            return (control, guard_info)
        closing_speed = max(0.0, ego_speed_mps - lead_speed_mps)
        ttc = front_distance_m / closing_speed if closing_speed > 0.05 and front_distance_m > 0.0 else float('inf')
        relative_stopping_distance = max(0.0, (ego_speed_mps ** 2 - lead_speed_mps ** 2) / max(2.0 * cfg.eos_collision_guard_deceleration_mps2, 1e-06))
        required_gap = cfg.eos_collision_guard_min_gap_m + cfg.eos_collision_guard_time_headway_s * ego_speed_mps + relative_stopping_distance
        triggered = bool(front_distance_m < required_gap or (math.isfinite(ttc) and ttc < cfg.eos_collision_guard_ttc_threshold_s) or bool(safety_info.get('hard_intervention', False)))
        brake_command = 0.0
        if triggered:
            gap_risk = clamp((required_gap - front_distance_m) / max(required_gap, 1.0), 0.0, 1.0)
            ttc_risk = 0.0
            if math.isfinite(ttc):
                ttc_risk = clamp((cfg.eos_collision_guard_ttc_threshold_s - ttc) / max(cfg.eos_collision_guard_ttc_threshold_s, 1e-06), 0.0, 1.0)
            risk = max(gap_risk, ttc_risk)
            brake_command = cfg.eos_collision_guard_min_brake + (cfg.eos_collision_guard_max_brake - cfg.eos_collision_guard_min_brake) * risk
            severe = bool(front_distance_m <= cfg.eos_collision_guard_emergency_gap_m or (math.isfinite(ttc) and ttc <= cfg.eos_collision_guard_emergency_ttc_s))
            if severe:
                brake_command = 1.0
            control.throttle = 0.0
            control.brake = max(float(control.brake), clamp(brake_command, 0.0, 1.0))
        guard_info = {'direct_collision_brake': bool(triggered), 'direct_brake_command': float(brake_command), 'required_collision_gap_m': float(required_gap), 'direct_guard_ttc_s': None if not math.isfinite(ttc) else float(ttc)}
        return (control, guard_info)

    def _apply_lateral_recovery(self, steering_residual: float, lateral_offset_m: float, target_speed_mps: float) -> Tuple[float, float, bool]:
        cfg = self.config.scenario
        if abs(lateral_offset_m) < cfg.lateral_recovery_threshold_m:
            return (steering_residual, target_speed_mps, False)
        return (0.0, min(target_speed_mps, cfg.lateral_recovery_speed_mps), True)

    def _observation_and_state(self) -> Tuple[np.ndarray, Dict[str, float]]:
        ego_transform = self.ego.get_transform()
        lead_transform = self.lead.get_transform()
        ego_projection = self.route_info.route.project(ego_transform.location)
        lead_projection = self.route_info.route.project(lead_transform.location)
        ego_speed = speed_mps(self.ego)
        lead_speed = speed_mps(self.lead)
        dt = self.config.carla.fixed_delta_seconds
        raw_acceleration = (ego_speed - self.previous_speed_mps) / dt
        acceleration_alpha = clamp(self.config.scenario.acceleration_filter_alpha, 0.0, 1.0)
        acceleration = acceleration_alpha * raw_acceleration + (1.0 - acceleration_alpha) * self.filtered_acceleration_mps2
        raw_jerk = (acceleration - self.filtered_acceleration_mps2) / dt
        jerk_alpha = clamp(self.config.scenario.jerk_filter_alpha, 0.0, 1.0)
        jerk = jerk_alpha * raw_jerk + (1.0 - jerk_alpha) * self.filtered_jerk_mps3
        self.filtered_acceleration_mps2 = float(acceleration)
        self.filtered_jerk_mps3 = float(jerk)
        front_distance = max(0.0, lead_projection.progress_m - ego_projection.progress_m - 4.5)
        relative_speed = ego_speed - lead_speed
        yaw_error = wrap_angle_rad(math.radians(ego_transform.rotation.yaw) - ego_projection.route_yaw_rad)
        distance_to_stop = self.route_info.stop_progress_m - ego_projection.progress_m
        assert self.perception is not None
        perceived = self.perception.observe(self.simulation_time_s, self.current_actual_light_state, front_distance, relative_speed)
        light_one_hot = np.zeros(4, dtype=np.float32)
        light_one_hot[int(np.clip(perceived.light_state, 0, 3))] = 1.0
        weather_one_hot = np.zeros(4, dtype=np.float32)
        weather_one_hot[self.config.scenario.weather_set.index(self.weather_name)] = 1.0
        observation = np.array([clamp(ego_speed / max(self.speed_limit_mps, 0.001), 0.0, 2.0) - 1.0, clamp(acceleration / 6.0, -1.0, 1.0), clamp(jerk / 12.0, -1.0, 1.0), clamp(ego_projection.lateral_offset_m / 3.0, -1.0, 1.0), clamp(yaw_error / math.pi, -1.0, 1.0), clamp(perceived.lead_distance_m / 80.0, 0.0, 1.0), clamp(perceived.relative_speed_mps / 20.0, -1.0, 1.0), *light_one_hot.tolist(), clamp(distance_to_stop / 100.0, -1.0, 1.0), clamp(ego_projection.progress_m / max(self.route_info.goal_progress_m, 1.0), 0.0, 1.0), clamp(self.speed_limit_mps / 30.0, 0.0, 1.0), clamp(self.safe_weather_speed_mps / 30.0, 0.0, 1.0), *weather_one_hot.tolist(), 1.0 if self.adverse_weather else 0.0, clamp(perceived.confidence, 0.0, 1.0), clamp(self.previous_target_fraction, 0.0, 1.0), clamp(self.simulation_time_s / max(self.config.scenario.max_episode_seconds, 1e-06), 0.0, 1.0)], dtype=np.float32)
        observation = np.clip(observation, -1.0, 1.0).astype(np.float32)
        state = {'ego_speed_mps': float(ego_speed), 'lead_speed_mps': float(lead_speed), 'accel_mps2': float(acceleration), 'jerk_mps3': float(jerk), 'ego_progress_m': float(ego_projection.progress_m), 'lead_progress_m': float(lead_projection.progress_m), 'lateral_offset_m': float(ego_projection.lateral_offset_m), 'yaw_error_rad': float(yaw_error), 'front_distance_m': float(front_distance), 'distance_to_stop_m': float(distance_to_stop), 'perception_failure': float(perceived.failure), 'perception_confidence': float(perceived.confidence)}
        return (observation, state)

    def _base_info(self, state: Dict[str, float]) -> Dict[str, Any]:
        return {'reward_strategy': self.reward_strategy, 'weather': self.weather_name, 'adverse_weather': self.adverse_weather, 'speed_limit_kmh': self.speed_limit_kmh, 'safe_weather_speed_kmh': self.safe_weather_speed_mps * 3.6, 'safe_following_threshold_m': self.speed_limit_mps, 'state': state}

    def _step_internal(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).reshape(2)
        action = np.clip(action, -1.0, 1.0)
        dt = self.config.carla.fixed_delta_seconds
        self._set_traffic_light_state(LIGHT_RED if self.simulation_time_s < self.red_duration_s else LIGHT_GREEN)
        ego_projection_before = self.route_info.route.project(self.ego.get_location())
        lead_projection_before = self.route_info.route.project(self.lead.get_location())
        ego_speed_before = speed_mps(self.ego)
        lead_speed_before = speed_mps(self.lead)
        lead_control = self.lead_controller.control(self.lead, self.route_info.route, lead_projection_before, lead_speed_before, self._lead_target_speed(), steering_residual=0.0)
        self.lead.apply_control(lead_control)
        target_fraction = 0.5 * (float(action[0]) + 1.0) * self.config.scenario.max_target_speed_ratio
        raw_target_speed = target_fraction * self.speed_limit_mps
        steering_residual = float(action[1]) * self.config.scenario.steering_residual_limit
        front_distance_before = max(0.0, lead_projection_before.progress_m - ego_projection_before.progress_m - 4.5)
        target_speed, safety_info = self._apply_longitudinal_safety_layer(raw_target_speed, ego_speed_before, lead_speed_before, front_distance_before)
        steering_residual, target_speed, lateral_recovery = self._apply_lateral_recovery(steering_residual, ego_projection_before.lateral_offset_m, target_speed)
        safety_info['lateral_recovery'] = bool(lateral_recovery)
        safety_info['final_target_speed_mps'] = float(target_speed)
        self.last_safety_layer_info = safety_info
        ego_control = self.ego_controller.control(self.ego, self.route_info.route, ego_projection_before, ego_speed_before, target_speed, steering_residual=steering_residual, disturbance=self._crosswind_disturbance())
        ego_control, direct_guard_info = self._apply_eos_direct_collision_brake(ego_control, ego_speed_before, lead_speed_before, front_distance_before, safety_info)
        safety_info.update(direct_guard_info)
        self.last_safety_layer_info = safety_info
        self.ego.apply_control(ego_control)
        self.world.tick()
        self.simulation_time_s += dt
        self.episode_steps += 1
        self._set_traffic_light_state(LIGHT_RED if self.simulation_time_s < self.red_duration_s else LIGHT_GREEN)
        observation, state = self._observation_and_state()
        raw_collision_callbacks = self.collision_counter.consume()
        collision_event = raw_collision_callbacks > 0
        crossed_stop_line = self.previous_progress_m < self.route_info.stop_progress_m <= state['ego_progress_m']
        red_violation_event = bool(crossed_stop_line and self.current_actual_light_state == LIGHT_RED and (not self.red_violation_recorded))
        if red_violation_event:
            self.red_violation_recorded = True
        dangerous_following_condition = 0.0 < state['front_distance_m'] < self.speed_limit_mps
        reward_inputs = RewardInputs(dt_s=dt, elapsed_time_s=self.simulation_time_s, speed_mps=state['ego_speed_mps'], speed_limit_mps=self.speed_limit_mps, safe_weather_speed_mps=self.safe_weather_speed_mps, jerk_mps3=state['jerk_mps3'], lateral_offset_m=state['lateral_offset_m'], collision_event=collision_event, red_light_violation_event=red_violation_event, dangerous_following=dangerous_following_condition, adverse_weather=self.adverse_weather, perception_failure=bool(state['perception_failure']))
        reward, reward_components = compute_reward(self.reward_strategy, reward_inputs, self.config.reward)
        reached_goal = state['ego_progress_m'] >= self.route_info.goal_progress_m
        severe_offroute = abs(state['lateral_offset_m']) >= self.config.scenario.severe_offroute_threshold_m
        terminated = bool(reached_goal or (collision_event and self.config.scenario.collision_terminates))
        truncated = bool(self.episode_steps >= self.max_episode_steps or severe_offroute)
        info = self._base_info(state)
        info.update({'reward_components': reward_components, 'safety_layer': dict(self.last_safety_layer_info), 'raw_collision_callbacks': int(raw_collision_callbacks), 'episode_finished': bool(terminated or truncated), 'termination_reason': 'goal' if reached_goal else 'collision' if collision_event and self.config.scenario.collision_terminates else 'offroute' if severe_offroute else 'timeout' if self.episode_steps >= self.max_episode_steps else 'running'})
        self.previous_speed_mps = state['ego_speed_mps']
        self.previous_acceleration_mps2 = state['accel_mps2']
        self.previous_commanded_target_speed_mps = float(target_speed)
        self.previous_commanded_target_acceleration_mps2 = float(safety_info.get('commanded_target_acceleration_mps2', 0.0))
        self.previous_progress_m = state['ego_progress_m']
        self.previous_target_fraction = clamp(target_speed / max(self.speed_limit_mps * self.config.scenario.max_target_speed_ratio, 1e-06), 0.0, 1.0)
        return (observation, float(reward), terminated, truncated, info)

    def step(self, action):
        observation, reward, terminated, truncated, info = self._step_internal(action)
        if USE_GYMNASIUM_API:
            return (observation, reward, terminated, truncated, info)
        done = bool(terminated or truncated)
        if truncated and (not terminated):
            info['TimeLimit.truncated'] = True
        return (observation, reward, done, info)

    def close(self) -> None:
        self._destroy_actors()
        try:
            self.world.freeze_all_traffic_lights(False)
        except Exception:
            pass
        try:
            self.world.apply_settings(self.original_settings)
        except Exception:
            pass

def strategy_folder(strategy: str) -> str:
    return strategy.replace('-', '_')

def train_one(config: Config, strategy: str, seed: int, timesteps: int, device: str) -> Path:
    set_all_seeds(seed)
    output_root = Path(config.experiment.output_dir)
    strategy_dir = output_root / 'models' / strategy_folder(strategy) / ('seed_%d' % seed)
    strategy_dir.mkdir(parents=True, exist_ok=True)
    raw_environment = EosStpaCarlaEnv(config, strategy, seed=seed)
    route_metadata = raw_environment.route_metadata()
    environment = Monitor(raw_environment, filename=str(strategy_dir / 'monitor.csv'))
    checkpoint = CheckpointCallback(save_freq=max(1, config.ppo.checkpoint_frequency), save_path=str(strategy_dir / 'checkpoints'), name_prefix='ppo', save_replay_buffer=False, save_vecnormalize=False)
    if SB3_MAJOR >= 2:
        network_architecture: Any = dict(pi=list(config.ppo.policy_hidden_sizes), vf=list(config.ppo.policy_hidden_sizes))
    else:
        network_architecture = [dict(pi=list(config.ppo.policy_hidden_sizes), vf=list(config.ppo.policy_hidden_sizes))]
    policy_kwargs = {'activation_fn': torch.nn.Tanh, 'net_arch': network_architecture}
    model = PPO('MlpPolicy', environment, learning_rate=config.ppo.learning_rate, n_steps=config.ppo.n_steps, batch_size=config.ppo.batch_size, n_epochs=config.ppo.n_epochs, gamma=config.ppo.gamma, gae_lambda=config.ppo.gae_lambda, clip_range=config.ppo.clip_range, vf_coef=config.ppo.vf_coef, max_grad_norm=config.ppo.max_grad_norm, ent_coef=config.ppo.ent_coef, seed=seed, verbose=1, tensorboard_log=str(output_root / 'tensorboard'), policy_kwargs=policy_kwargs, device=device)
    try:
        learn_kwargs = {'total_timesteps': timesteps, 'callback': checkpoint, 'tb_log_name': '%s_seed_%d' % (strategy_folder(strategy), seed)}
        model.learn(**learn_kwargs)
        save_without_extension = strategy_dir / 'final_model'
        model.save(str(save_without_extension))
        manifest = {'strategy': strategy, 'seed': seed, 'timesteps': timesteps, 'stable_baselines3_version': stable_baselines3.__version__, 'ppo': asdict(config.ppo), 'carla': asdict(config.carla), 'scenario': asdict(config.scenario), 'reward': asdict(config.reward), 'route': route_metadata}
        (strategy_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        return save_without_extension.with_suffix('.zip')
    finally:
        environment.close()

def run_training(config: Config, strategies: Sequence[str], seeds: Sequence[int], timesteps: int, device: str) -> None:
    for strategy in strategies:
        for seed in seeds:
            print('\n[TRAIN] strategy=%s seed=%d timesteps=%d' % (strategy, seed, timesteps))
            path = train_one(config, strategy, int(seed), timesteps, device)
            print('[TRAINED] %s' % path.resolve())

def build_config_from_args(args: argparse.Namespace) -> Config:
    config = Config()
    config.carla.host = args.host
    config.carla.port = args.port
    config.carla.timeout_seconds = args.timeout
    config.carla.town = args.town
    config.carla.no_rendering_mode = not args.render
    config.carla.route_traffic_light_id = args.route_light_id
    config.scenario.speed_limit_override_kmh = args.speed_limit_kmh
    config.scenario.hard_safety_layer_enabled = not args.disable_hard_safety_layer
    config.scenario.eos_operational_layer_enabled = not args.disable_eos_operational_layer
    config.reward.omega_time = args.omega_time
    config.reward.omega_speed = args.omega_speed
    config.reward.incremental_time_reward = not args.literal_elapsed_time_reward
    config.experiment.output_dir = args.output_dir
    config.ppo.checkpoint_frequency = args.checkpoint_frequency
    return config

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='CARLA/PPO training for STPA and EOS-STPA policies')
    parser.add_argument('--strategy', choices=['STPA', 'EOS-STPA', 'all'], default='all')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--timeout', type=float, default=60.0)
    parser.add_argument('--town', default='Town03')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--route-light-id', type=int, default=None)
    parser.add_argument('--speed-limit-kmh', type=float, default=60.0)
    parser.add_argument('--disable-hard-safety-layer', action='store_true')
    parser.add_argument('--disable-eos-operational-layer', action='store_true')
    parser.add_argument('--omega-time', type=float, default=0.1)
    parser.add_argument('--omega-speed', type=float, default=0.05)
    parser.add_argument('--literal-elapsed-time-reward', action='store_true')
    parser.add_argument('--train-seeds', default='0')
    parser.add_argument('--train-steps', type=int, default=100000)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--output-dir', default='/root/autodl-tmp/yolo_unzipped/yolo/ESWA')
    parser.add_argument('--checkpoint-frequency', type=int, default=20000)
    return parser.parse_args()

def main() -> None:
    args = parse_arguments()
    config = build_config_from_args(args)
    training_seeds = parse_int_spec(args.train_seeds)
    strategies = ['STPA', 'EOS-STPA'] if args.strategy == 'all' else [args.strategy]
    print('[INFO] Stable-Baselines3 version:', stable_baselines3.__version__)
    print('[INFO] Gym API:', 'Gymnasium' if USE_GYMNASIUM_API else 'Gym legacy')
    print('[INFO] Connecting to CARLA at %s:%d' % (config.carla.host, config.carla.port))
    run_training(config, strategies, training_seeds, args.train_steps, args.device)
if __name__ == '__main__':
    main()
