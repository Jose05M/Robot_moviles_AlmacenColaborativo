"""
anymal_gait.py
--------------
Trot gait generator for the ANYmal within the mini challenge.

This module implements:
- Kinematic model of an ANYmal leg (FK, IK, Jacobian)
- Simplified model of the full ANYmal with 4 legs
- Cartesian trajectory generation for each foot
- Trot gait: diagonal legs in phase
- Monitoring of the Jacobian determinant on each leg
- Simple singularity avoidance strategy
- Integration with WarehouseSim (sim.py) to move the robot base
- Logging and plotting of Phase 2 performance

"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import math
import numpy as np
import matplotlib.pyplot as plt

from sim import WarehouseSim, wrap_angle, distance, clamp

PLOTS_DIR = Path(__file__).resolve().parent.parent / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# ANYmal leg
# =============================================================================

class ANYmalLeg:
    """
    Kinematic model of an ANYmal leg with 3 DoF:
        q1 = HAA
        q2 = HFE
        q3 = KFE

    Conventions:
        x: forward
        y: lateral
        z: up
        side = +1 left legs, -1 right legs
    """

    def __init__(self, name: str, l0: float = 0.0585, l1: float = 0.35, l2: float = 0.33, side: int = +1):
        self.name = name
        self.l0 = l0
        self.l1 = l1
        self.l2 = l2
        self.side = side

        self.q = np.zeros(3)

        # Soft safety limits
        self.q_min = np.array([-0.72, -1.8, -2.69], dtype=float)
        self.q_max = np.array([+0.49, +1.8, -0.03], dtype=float)

    def forward_kinematics(self, q: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Analytical FK:
            q -> p = [x, y, z]
        """
        if q is not None:
            self.q = np.asarray(q, dtype=float)

        q1, q2, q3 = self.q

        x = self.l1 * np.sin(q2) + self.l2 * np.sin(q2 + q3)
        y = self.side * self.l0 * np.cos(q1)
        z = -self.l1 * np.cos(q2) - self.l2 * np.cos(q2 + q3)

        return np.array([x, y, z], dtype=float)

    def inverse_kinematics(self, p_des: np.ndarray) -> np.ndarray:
        """
        Closed-form geometric IK for the leg.

        Configuration:
            knee bent backward => q3 < 0
        """
        x, y, z = map(float, p_des)

        # Solve q1 via lateral projection
        r_yz_sq = y**2 + z**2 - self.l0**2
        r_yz = math.sqrt(max(r_yz_sq, 1e-9))
        q1 = math.atan2(y, -z) - math.atan2(self.side * self.l0, r_yz)

        # Solve q3 with the law of cosines
        r_sq = x**2 + z**2
        D = (r_sq - self.l1**2 - self.l2**2) / (2.0 * self.l1 * self.l2)
        D = clamp(D, -1.0, 1.0)
        q3 = -math.acos(D)

        # Solve q2
        alpha = math.atan2(x, -z)
        beta = math.atan2(self.l2 * math.sin(-q3),
                          self.l1 + self.l2 * math.cos(q3))
        q2 = alpha - beta

        q = np.array([q1, q2, q3], dtype=float)
        q = np.clip(q, self.q_min, self.q_max)
        return q

    def jacobian(self, q: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Analytical 3x3 Jacobian.
        """
        if q is None:
            q = self.q
        q1, q2, q3 = q

        J = np.zeros((3, 3), dtype=float)

        J[0, 0] = 0.0
        J[0, 1] = self.l1 * math.cos(q2) + self.l2 * math.cos(q2 + q3)
        J[0, 2] = self.l2 * math.cos(q2 + q3)

        J[1, 0] = -self.side * self.l0 * math.sin(q1)
        J[1, 1] = 0.0
        J[1, 2] = 0.0

        J[2, 0] = 0.0
        J[2, 1] = self.l1 * math.sin(q2) + self.l2 * math.sin(q2 + q3)
        J[2, 2] = self.l2 * math.sin(q2 + q3)

        return J

    def det_jacobian(self, q: Optional[np.ndarray] = None) -> float:
        """Jacobian determinant."""
        return float(np.linalg.det(self.jacobian(q)))

    def is_singular(self, q: Optional[np.ndarray] = None, tol: float = 1e-3) -> bool:
        """True if near a singularity."""
        return abs(self.det_jacobian(q)) < tol


# =============================================================================
# Full ANYmal robot
# =============================================================================

class ANYmal:
    """
    Simplified ANYmal model:
    - 4 legs
    - 2D floating base in the warehouse world
    - internal joints for the gait
    """

    LEG_NAMES = ["LF", "RF", "LH", "RH"]

    def __init__(self):
        self.legs: Dict[str, ANYmalLeg] = {
            "LF": ANYmalLeg("LF", side=+1),
            "RF": ANYmalLeg("RF", side=-1),
            "LH": ANYmalLeg("LH", side=+1),
            "RH": ANYmalLeg("RH", side=-1),
        }

        # Simplified base state
        self.base_x = 0.0
        self.base_y = 0.0
        self.base_theta = 0.0

        # Approximate payload requested in the challenge
        self.payload_mass = 6.0

        # Nominal joint posture
        self.q_nominal_leg = np.array([0.0, 0.70, -1.40], dtype=float)

    def set_base_pose(self, x: float, y: float, theta: float) -> None:
        """Updates the base pose."""
        self.base_x = x
        self.base_y = y
        self.base_theta = wrap_angle(theta)

    def get_base_pose(self) -> Tuple[float, float, float]:
        """Returns the base pose."""
        return self.base_x, self.base_y, self.base_theta

    def get_all_joint_angles(self) -> np.ndarray:
        """Concatenates the 12 joint angles."""
        return np.concatenate([self.legs[name].q for name in self.LEG_NAMES])

    def set_all_joint_angles(self, q12: np.ndarray) -> None:
        """Sets the 12 joint angles."""
        q12 = np.asarray(q12, dtype=float)
        assert q12.shape == (12,), f"Expected 12 angles, got {q12.shape}"
        for i, name in enumerate(self.LEG_NAMES):
            self.legs[name].q = q12[3 * i: 3 * (i + 1)].copy()

    def get_all_foot_positions(self) -> Dict[str, np.ndarray]:
        """Foot positions of the 4 legs in their local frames."""
        return {name: self.legs[name].forward_kinematics() for name in self.LEG_NAMES}

    def get_all_detJ(self) -> Dict[str, float]:
        """Jacobian determinant per leg."""
        return {name: self.legs[name].det_jacobian() for name in self.LEG_NAMES}


# =============================================================================
# ANYmal phase log
# =============================================================================

@dataclass
class ANYmalLog:
    """Detailed log of the ANYmal gait."""
    t: List[float] = field(default_factory=list)

    base_x: List[float] = field(default_factory=list)
    base_y: List[float] = field(default_factory=list)
    base_theta: List[float] = field(default_factory=list)

    v_base_cmd: List[float] = field(default_factory=list)
    omega_base_cmd: List[float] = field(default_factory=list)

    q: List[np.ndarray] = field(default_factory=list)

    detJ: Dict[str, List[float]] = field(default_factory=lambda: {
        "LF": [], "RF": [], "LH": [], "RH": []
    })

    foot_x: Dict[str, List[float]] = field(default_factory=lambda: {
        "LF": [], "RF": [], "LH": [], "RH": []
    })
    foot_z: Dict[str, List[float]] = field(default_factory=lambda: {
        "LF": [], "RF": [], "LH": [], "RH": []
    })

    singularity_events: List[int] = field(default_factory=list)
    phase_name: List[str] = field(default_factory=list)

    def append(
        self,
        t: float,
        base_pose: Tuple[float, float, float],
        v_base_cmd: float,
        omega_base_cmd: float,
        q12: np.ndarray,
        feet: Dict[str, np.ndarray],
        detJ: Dict[str, float],
        phase_name: str,
        singularity_violation: bool
    ) -> None:
        """Appends a sample to the log."""
        bx, by, bth = base_pose

        self.t.append(t)
        self.base_x.append(bx)
        self.base_y.append(by)
        self.base_theta.append(bth)
        self.v_base_cmd.append(v_base_cmd)
        self.omega_base_cmd.append(omega_base_cmd)
        self.q.append(q12.copy())
        self.phase_name.append(phase_name)

        for name in ("LF", "RF", "LH", "RH"):
            self.detJ[name].append(detJ[name])
            self.foot_x[name].append(feet[name][0])
            self.foot_z[name].append(feet[name][2])

        if singularity_violation:
            self.singularity_events.append(len(self.t) - 1)


# =============================================================================
# Trot gait generator
# =============================================================================

class ANYmalGaitController:
    """
    Trot gait controller for Phase 2.

    Goals:
    - Move the ANYmal from its initial position to p_destino
    - Generate cartesian foot trajectories
    - Solve IK per leg
    - Monitor det(J)
    - Avoid singularities by reducing swing amplitude if needed
    """

    def __init__(self, sim: WarehouseSim):
        self.sim = sim
        self.anymal = ANYmal()

        # Sync base with the world
        robot = self.sim.robots["anymal"]
        self.anymal.set_base_pose(robot.x, robot.y, robot.theta)

        self.log = ANYmalLog()

        # Gait parameters
        self.period = 0.65               # s
        self.step_height = 0.05          # m
        self.step_length = 0.1         # m
        self.nominal_y = {
            "LF": +0.052,
            "RF": -0.052,
            "LH": +0.052,
            "RH": -0.052,
        }

        # Nominal foot centers in the leg's local frame
        self.foot_centers = {
            "LF": np.array([ 0.02, +0.052, -0.52], dtype=float),
            "RF": np.array([ 0.02, -0.052, -0.52], dtype=float),
            "LH": np.array([ 0.02, +0.052, -0.52], dtype=float),
            "RH": np.array([ 0.02, -0.052, -0.52], dtype=float),
        }

        self.detJ_tol = 1e-3
        self.safe_detJ_target = 2.0e-3

        # Base control
        self.v_base_max = 0.55
        self.omega_base_max = 0.8
        self.k_rho = 0.55
        self.k_alpha = 1.1

        self.max_steps = 5000

        self.reached_goal = False

        # 2) cross through the middle
        # 3) exit
        # 4) turn and reach p_dest_anymal
        cx, cy, cw, ch = self.sim.corridor
        gx, gy = self.sim.anymal_goal

        self.path_waypoints = [
            (cx + 0.15 * cw, cy + 0.50 * ch),   # corridor entrance
            (cx + 0.50 * cw, cy + 0.50 * ch),   # corridor center
            (cx + 0.90 * cw, cy + 0.50 * ch),   # corridor exit
            (gx, gy),                           # final ANYmal destination
        ]
        self.current_waypoint_idx = 0
        self.waypoint_tol = 0.16

        # Mounting offsets of the 3 PuzzleBots on the ANYmal's back
        self.pb_mount_offsets = {
            "pb1": (+0.10, +0.08),
            "pb2": (-0.02,  0.00),
            "pb3": (+0.10, -0.08),
        }

    # -------------------------------------------------------------------------
    # Global base trajectory
    # -------------------------------------------------------------------------

    def _base_control_to_goal(self, goal_x: float, goal_y: float) -> Tuple[float, float]:
        """
        Simple pose->point control for the ANYmal base.
        """
        robot = self.sim.robots["anymal"]
        dx = goal_x - robot.x
        dy = goal_y - robot.y
        rho = math.hypot(dx, dy)

        desired_heading = math.atan2(dy, dx)
        alpha = wrap_angle(desired_heading - robot.theta)

        v_cmd = self.k_rho * rho
        omega_cmd = self.k_alpha * alpha

        if abs(alpha) > math.radians(30.0):
            v_cmd *= 0.35

        v_cmd = clamp(v_cmd, 0.0, self.v_base_max)
        omega_cmd = clamp(omega_cmd, -self.omega_base_max, self.omega_base_max)
        return v_cmd, omega_cmd

    def _get_current_waypoint(self) -> Tuple[float, float]:
        """Returns the current waypoint."""
        idx = min(self.current_waypoint_idx, len(self.path_waypoints) - 1)
        return self.path_waypoints[idx]


    def _update_waypoint_progress(self) -> None:
        """Advances to the next waypoint if the current one was reached."""
        if self.current_waypoint_idx >= len(self.path_waypoints):
            return

        robot = self.sim.robots["anymal"]
        gx, gy = self._get_current_waypoint()
        err = distance((robot.x, robot.y), (gx, gy))

        if err < self.waypoint_tol and self.current_waypoint_idx < len(self.path_waypoints) - 1:
            self.current_waypoint_idx += 1

    def _integrate_base(self, v_cmd: float, omega_cmd: float) -> None:
        """
        Integrates the ANYmal base and syncs it with sim.py.
        """
        dt = self.sim.dt
        robot = self.sim.robots["anymal"]

        theta_mid = robot.theta + 0.5 * omega_cmd * dt
        x_new = robot.x + v_cmd * math.cos(theta_mid) * dt
        y_new = robot.y + v_cmd * math.sin(theta_mid) * dt
        theta_new = wrap_angle(robot.theta + omega_cmd * dt)

        self.sim.set_robot_pose("anymal", x_new, y_new, theta_new)
        self.anymal.set_base_pose(x_new, y_new, theta_new)
        self._sync_puzzlebots_with_anymal()

    def _sync_puzzlebots_with_anymal(self) -> None:
        """
        Keeps the 3 PuzzleBots mounted on the ANYmal during the gait.
        """
        self.sim.sync_puzzlebots_on_anymal(offsets=self.pb_mount_offsets)

    # -------------------------------------------------------------------------
    # Gait: cartesian foot trajectory
    # -------------------------------------------------------------------------

    def _phase_value(self, t: float) -> float:
        """Normalized phase in [0,1)."""
        return (t / self.period) % 1.0

    def _swing_profile(self, phase: float) -> float:
        """
        Smooth swing-lift profile in [0,1].
        Only active during the positive half of the cycle.
        """
        return max(0.0, math.sin(2.0 * math.pi * phase))

    def _foot_target_for_leg(
        self,
        leg_name: str,
        t: float,
        step_length_scale: float = 1.0,
        step_height_scale: float = 1.0
    ) -> np.ndarray:
        """
        Generates the desired cartesian foot position in the leg's local frame.

        Trot:
            LF + RH in phase
            RF + LH in antiphase
        """
        phase = self._phase_value(t)

        if leg_name in ("LF", "RH"):
            lift = self._swing_profile(phase)
        else:
            lift = self._swing_profile((phase + 0.5) % 1.0)

        center = self.foot_centers[leg_name].copy()

        # Local forward/backward foot motion
        x = center[0] + step_length_scale * self.step_length * (lift - 0.5 * max(0.0, 1.0 - lift))
        # Height during swing
        z = center[2] + step_height_scale * self.step_height * lift

        # Keep nominal y
        y = center[1]

        return np.array([x, y, z], dtype=float)

    # -------------------------------------------------------------------------
    # Singularities
    # -------------------------------------------------------------------------

    def _compute_q12_from_cartesian_targets(
        self,
        foot_targets: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, Dict[str, float], Dict[str, np.ndarray]]:
        """
        Solves IK per leg and returns:
            q12, detJ per leg, verified FK positions
        """
        q12 = np.zeros(12, dtype=float)
        detJ = {}
        feet_fk = {}

        for i, name in enumerate(ANYmal.LEG_NAMES):
            leg = self.anymal.legs[name]
            q_leg = leg.inverse_kinematics(foot_targets[name])
            leg.q = q_leg.copy()

            q12[3 * i: 3 * (i + 1)] = q_leg
            detJ[name] = leg.det_jacobian(q_leg)
            feet_fk[name] = leg.forward_kinematics(q_leg)

        return q12, detJ, feet_fk

    def _avoid_singularities(
        self,
        t: float,
        max_iter: int = 5
    ) -> Tuple[np.ndarray, Dict[str, float], Dict[str, np.ndarray], bool]:
        """
        Generates cartesian targets and reduces amplitudes if singularities are found.
        """
        step_length_scale = 1
        step_height_scale = 1
        violated = False

        for _ in range(max_iter):
            foot_targets = {
                name: self._foot_target_for_leg(
                    leg_name=name,
                    t=t,
                    step_length_scale=step_length_scale,
                    step_height_scale=step_height_scale
              )
            for name in ANYmal.LEG_NAMES
            }

            q12, detJ, feet_fk = self._compute_q12_from_cartesian_targets(foot_targets)

            min_det = min(abs(detJ[name]) for name in ANYmal.LEG_NAMES)
            if min_det > self.detJ_tol:
                return q12, detJ, feet_fk, violated

            # Adjust amplitude to move away from the singularity
            step_length_scale *= 0.70
            step_height_scale *= 0.80
            violated = True

        return q12, detJ, feet_fk, violated

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------

    def _log_step(
        self,
        t: float,
        v_base_cmd: float,
        omega_base_cmd: float,
        q12: np.ndarray,
        feet_fk: Dict[str, np.ndarray],
        detJ: Dict[str, float],
        singularity_violation: bool,
        phase_name: str = "anymal_trot"
    ) -> None:
        """Saves an ANYmal sample."""
        self.log.append(
            t=t,
            base_pose=self.anymal.get_base_pose(),
            v_base_cmd=v_base_cmd,
            omega_base_cmd=omega_base_cmd,
            q12=q12,
            feet=feet_fk,
            detJ=detJ,
            phase_name=phase_name,
            singularity_violation=singularity_violation
        )

    # -------------------------------------------------------------------------
    # Main execution
    # -------------------------------------------------------------------------

    def run(self, verbose: bool = True) -> ANYmalLog:
        """
        Runs the full Phase 2:
        - trot
        - advance to p_destino
        - singularity monitoring
        """
        self.reached_goal = False
        gx, gy = self.sim.anymal_goal

        # At the start the PuzzleBots are mounted on the ANYmal
        self._sync_puzzlebots_with_anymal()
        self.sim.record_state(
            phase="anymal_init",
            note="Start of ANYmal phase with 3 PuzzleBots mounted"
        )

        for k in range(self.max_steps):
            t = self.sim.time

            # 1) current waypoint
            wp_x, wp_y = self._get_current_waypoint()

            # 2) base control toward waypoint
            v_base_cmd, omega_base_cmd = self._base_control_to_goal(wp_x, wp_y)

            # 3) cartesian gait + IK + singularity monitoring
            q12, detJ, feet_fk, singularity_violation = self._avoid_singularities(t)

            # 4) apply joints
            self.anymal.set_all_joint_angles(q12)

            # 5) integrate base
            self._integrate_base(v_base_cmd, omega_base_cmd)

            # 6) advance waypoint if reached
            self._update_waypoint_progress()

            # 7) local logging
            self._log_step(
                t=t,
                v_base_cmd=v_base_cmd,
                omega_base_cmd=omega_base_cmd,
                q12=q12,
                feet_fk=feet_fk,
                detJ=detJ,
                singularity_violation=singularity_violation,
                phase_name=f"anymal_trot_wp{self.current_waypoint_idx}"
            )

            # 8) global logging
            min_det = min(abs(detJ[name]) for name in ANYmal.LEG_NAMES)
            self.sim.step(
                phase="anymal",
                note=(
                    f"wp={self.current_waypoint_idx + 1}/{len(self.path_waypoints)} | "
                    f"target=({wp_x:.2f},{wp_y:.2f}) | "
                    f"min|detJ|={min_det:.4e}"
                )
            )

            err = self.sim.anymal_goal_error()

            if verbose and (k % 50 == 0 or err < 0.15):
                print(
                    f"[anymal] step={k:04d} | "
                    f"wp={self.current_waypoint_idx + 1}/{len(self.path_waypoints)} | "
                    f"pos=({self.sim.robots['anymal'].x:.2f},{self.sim.robots['anymal'].y:.2f}) | "
                    f"err_goal={err:.3f} m | "
                    f"min|detJ|={min_det:.4e}"
                )

            # final success
            if err < 0.15 and self.current_waypoint_idx >= len(self.path_waypoints) - 1:
                self.reached_goal = True
                break

        # When the gait finishes, only deploy if the destination was actually reached
        if self.reached_goal:
            self.sim.activate_puzzlebots_at_work_zone()
            self.sim.record_state(
                phase="anymal_done",
                note="ANYmal reached p_dest and deployed the 3 PuzzleBots"
            )
        else:
            self.sim.record_state(
                phase="anymal_failed",
                note="ANYmal did not reach p_dest; PuzzleBots not deployed"
            )

        return self.log


# =============================================================================
# ANYmal phase plots
# =============================================================================

def plot_anymal_phase_results(log, title="ANYmal - Actuators and Foot Trajectory",
                              save_path=None):
    """
    Plot in the style of the course demo/base.
    """
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    gs = fig.add_gridspec(3, 4, hspace=0.45, wspace=0.35)

    leg_colors = {'LF': 'tab:blue', 'RF': 'tab:orange',
                  'LH': 'tab:green', 'RH': 'tab:red'}
    joint_labels = ['HAA (q1)', 'HFE (q2)', 'KFE (q3)']

    q_arr = np.array(log.q)
    t_arr = np.array(log.t)

    # --- Subplots 1-4: joint angles of each leg ---
    for i, name in enumerate(['LF', 'RF', 'LH', 'RH']):
        ax = fig.add_subplot(gs[0, i])
        q_leg = q_arr[:, 3*i:3*(i+1)]
        for j in range(3):
            ax.plot(t_arr, np.degrees(q_leg[:, j]),
                    linewidth=1.8, label=joint_labels[j])
        ax.set_title(f'Leg {name}', color=leg_colors[name], fontweight='bold')
        ax.set_xlabel('t [s]')
        ax.set_ylabel('angle [deg]')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)

    # --- Subplot 5: z height of all feet ---
    ax = fig.add_subplot(gs[1, :2])
    for name in ['LF', 'RF', 'LH', 'RH']:
        ax.plot(t_arr, np.array(log.foot_z[name]),
                color=leg_colors[name], linewidth=2, label=f'Foot {name}')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('foot z [m]')
    ax.set_title('Foot height (stance vs swing)')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Subplot 6: XZ trajectory of the feet ---
    ax = fig.add_subplot(gs[1, 2:])
    for name in ['LF', 'RF', 'LH', 'RH']:
        fx = np.array(log.foot_x[name])
        fz = np.array(log.foot_z[name])
        ax.plot(fx, fz,
                color=leg_colors[name], linewidth=2, label=f'{name}', alpha=0.7)
        ax.plot(fx[0], fz[0], 'o', color=leg_colors[name], markersize=8)
    ax.set_xlabel('foot x [m]')
    ax.set_ylabel('foot z [m]')
    ax.set_title('Lateral (XZ) foot trajectory')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='datalim')

    # --- Subplots 7-10: joint velocities ---
    dt = t_arr[1] - t_arr[0] if len(t_arr) > 1 else 0.04
    dq = np.gradient(q_arr, dt, axis=0)

    for i, name in enumerate(['LF', 'RF', 'LH', 'RH']):
        ax = fig.add_subplot(gs[2, i])
        dq_leg = dq[:, 3*i:3*(i+1)]
        for j in range(3):
            ax.plot(t_arr, dq_leg[:, j],
                    linewidth=1.5, label=joint_labels[j])
        ax.set_title(f'Velocities {name}', fontsize=10)
        ax.set_xlabel('t [s]')
        ax.set_ylabel('dq/dt [rad/s]')
        ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  -> Figure saved at {save_path}")
    return fig


# =============================================================================
# Local demo
# =============================================================================

def demo_anymal_gait():
    """
    Demo of Phase 2 of the challenge.
    """
    sim = WarehouseSim(dt=0.04)

    controller = ANYmalGaitController(sim=sim)
    log = controller.run(verbose=True)

    err = sim.anymal_goal_error()
    print("\nANYmal phase summary:")
    print(f"  Final error to p_destino: {err:.3f} m")
    print(f"  Meets error < 0.15 m: {err < 0.15}")
    print(f"  Singularity adjustment events: {len(log.singularity_events)}")
    print(f"  Total time: {sim.time:.2f} s")

    plot_anymal_phase_results(
        log=log,
        title="ANYmal Phase 2: Trot Gait",
        save_path=str(PLOTS_DIR / "anymal_phase2_gait.png")
    )

    sim.draw_world(
        phase="anymal_done",
        note=f"ANYmal arrived. error={err:.3f} m",
        show_lidar=False
    )
    plt.tight_layout()
    plt.show()

    return sim, controller, log


if __name__ == "__main__":
    demo_anymal_gait()
