"""
husky_pusher.py
---------------
Node/controller for Phase 1 of the mini challenge:
the Husky A200 must locate and push 3 large boxes
out of the corridor using skid-steer + a simple local planner.

This module:
- reuses the Husky's kinematic model
- integrates with WarehouseSim (sim.py)
- uses a simulated 2D LiDAR of the scenario
- implements simple logic for:
    1) selecting the target box
    2) navigating to a pre-push point
    3) aligning with the push direction
    4) pushing the box until it is out of the corridor
- saves logs of:
    * commanded v, omega
    * measured v, omega
    * wheel speeds
    * target box
    * planner state

"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import math
import numpy as np
import matplotlib.pyplot as plt

from sim import WarehouseSim, wrap_angle, distance, clamp

PLOTS_DIR = Path(__file__).resolve().parent.parent / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Husky A200 model
# =============================================================================

class HuskyA200:
    """
    Simplified kinematic model of the Husky A200 (4-wheel skid-steer).

    Model:
        avg_R = (wR1 + wR2)/2
        avg_L = (wL1 + wL2)/2

        v     = r/2 * (avg_R + avg_L) * slip
        omega = r/B * (avg_R - avg_L)

    Note:
    - the slip factor affects v
    - omega follows the base model shared in class
    """

    def __init__(self, r: float = 0.1651, B: float = 0.555):
        self.r = r
        self.B = B

        self.terrain = "asphalt"
        self.slip_factors = {
            "asphalt": 1.00,
            "grass":   0.85,
            "gravel":  0.78,
            "sand":    0.65,
            "mud":     0.50,
        }

        self.v_max = 1.5
        self.omega_max = 2.5
        self.wheel_max = 8.0

    def set_terrain(self, terrain_name: str) -> None:
        """Sets the current terrain."""
        self.terrain = terrain_name

    def get_slip(self) -> float:
        """Returns the slip factor for the current terrain."""
        return self.slip_factors.get(self.terrain, 0.8)

    def forward_kinematics(
        self,
        wR1: float,
        wR2: float,
        wL1: float,
        wL2: float
    ) -> Tuple[float, float]:
        """Forward kinematics from 4 wheels to (v, omega)."""
        avg_R = 0.5 * (wR1 + wR2)
        avg_L = 0.5 * (wL1 + wL2)
        slip = self.get_slip()

        v = self.r * 0.5 * (avg_R + avg_L) * slip
        omega = self.r / self.B * (avg_R - avg_L)
        return v, omega

    def inverse_kinematics(self, v: float, omega: float) -> Tuple[float, float, float, float]:
        """
        Inverse kinematics from (v, omega) to the 4 wheels.
        Assumes the same command on both right wheels and the same on both left wheels.
        """
        v = clamp(v, -self.v_max, self.v_max)
        omega = clamp(omega, -self.omega_max, self.omega_max)

        # Inverse compatible with the shared base model
        wR = (2.0 * v + omega * self.B) / (2.0 * self.r)
        wL = (2.0 * v - omega * self.B) / (2.0 * self.r)

        wR = clamp(wR, -self.wheel_max, self.wheel_max)
        wL = clamp(wL, -self.wheel_max, self.wheel_max)

        return wR, wR, wL, wL


# =============================================================================
# Husky phase log
# =============================================================================

@dataclass
class HuskyLog:
    """Detailed time-series log of the Husky phase."""
    t: List[float] = field(default_factory=list)
    x: List[float] = field(default_factory=list)
    y: List[float] = field(default_factory=list)
    theta: List[float] = field(default_factory=list)

    v_cmd: List[float] = field(default_factory=list)
    omega_cmd: List[float] = field(default_factory=list)

    v_meas: List[float] = field(default_factory=list)
    omega_meas: List[float] = field(default_factory=list)

    wR1: List[float] = field(default_factory=list)
    wR2: List[float] = field(default_factory=list)
    wL1: List[float] = field(default_factory=list)
    wL2: List[float] = field(default_factory=list)

    target_box: List[str] = field(default_factory=list)
    state: List[str] = field(default_factory=list)
    lidar_front_min: List[float] = field(default_factory=list)

    box_positions: Dict[str, List[Tuple[float, float]]] = field(default_factory=lambda: {
        "B1": [],
        "B2": [],
        "B3": []
    })

    def append(
        self,
        t: float,
        x: float,
        y: float,
        theta: float,
        v_cmd: float,
        omega_cmd: float,
        v_meas: float,
        omega_meas: float,
        wR1: float,
        wR2: float,
        wL1: float,
        wL2: float,
        target_box: str,
        state: str,
        lidar_front_min: float,
        boxes_snapshot: Dict[str, Tuple[float, float]]
    ) -> None:
        """Adds a time sample to the log."""
        self.t.append(t)
        self.x.append(x)
        self.y.append(y)
        self.theta.append(theta)

        self.v_cmd.append(v_cmd)
        self.omega_cmd.append(omega_cmd)

        self.v_meas.append(v_meas)
        self.omega_meas.append(omega_meas)

        self.wR1.append(wR1)
        self.wR2.append(wR2)
        self.wL1.append(wL1)
        self.wL2.append(wL2)

        self.target_box.append(target_box)
        self.state.append(state)
        self.lidar_front_min.append(lidar_front_min)

        for name in ("B1", "B2", "B3"):
            self.box_positions[name].append(boxes_snapshot[name])


# =============================================================================
# Local planner / Husky controller
# =============================================================================

class HuskyPusher:
    """
    High-level controller for clearing the corridor.

    States:
        - SELECT_BOX
        - GO_PREPUSH
        - ALIGN_PUSH
        - PUSH
        - PARK
        - DONE

    Strategy:
        1) pick the next box inside the corridor
        2) define a pre-push point behind the box
        3) reach that point with a simple proportional controller
        4) align orientation with the push direction
        5) push until the box's center leaves the corridor
    """

    def __init__(
        self,
        sim: WarehouseSim,
        terrain: str = "grass"
    ):
        self.sim = sim
        self.husky = HuskyA200()
        self.husky.set_terrain(terrain)

        self.state = "SELECT_BOX"
        self.current_box: Optional[str] = None

        self.log = HuskyLog()

        # Gains and thresholds
        self.k_rho = 0.9
        self.k_alpha = 1.8
        self.k_push_heading = 2.2

        self.pos_tol = 0.14
        self.ang_tol = math.radians(7.0)

        self.push_speed_cmd = 0.55
        self.max_steps = 4000
        # Final parking pose for the Husky to clear the corridor
        sx, sy, sw, sh = self.sim.start_zone
        self.park_pose = (
            sx + 0.28 * sw,   # x
            sy + 0.28 * sh,   # y
            math.radians(180.0)  # orientation facing left
        )

        # Preferred ejection direction per box
        # B1 and B3 downward, B2 upward
        self.push_dirs = {
            "B1": (0.0, -1.0),
            "B2": (0.0, +1.0),
            "B3": (0.0, -1.0),
        }

    # -------------------------------------------------------------------------
    # Planner geometry utilities
    # -------------------------------------------------------------------------

    def _boxes_still_blocking(self) -> List[str]:
        """Returns the large boxes that are still inside the corridor."""
        return [name for name in ("B1", "B2", "B3") if not self.sim.is_box_out_of_corridor(name)]

    def _select_next_box(self) -> Optional[str]:
        """
        Selects the next target box.
        Simple policy: the closest one to the Husky among those still blocking.
        """
        candidates = self._boxes_still_blocking()
        if not candidates:
            return None

        rx = self.sim.robots["husky"].x
        ry = self.sim.robots["husky"].y

        best_name = None
        best_d = float("inf")
        for name in candidates:
            cx, cy = self.sim.boxes[name].center()
            d = distance((rx, ry), (cx, cy))
            if d < best_d:
                best_d = d
                best_name = name

        return best_name

    def _get_push_direction(self, box_name: str) -> Tuple[float, float]:
        """Desired unit direction to push the box out."""
        dx, dy = self.push_dirs[box_name]
        norm = math.hypot(dx, dy)
        return dx / norm, dy / norm

    def _get_prepush_point(self, box_name: str, margin: float = 0.70) -> Tuple[float, float]:
        """
        Point behind the box relative to the push direction.
        If the box moves upward, the Husky is placed below it; vice versa.
        """
        box = self.sim.boxes[box_name]
        cx, cy = box.center()
        dir_x, dir_y = self._get_push_direction(box_name)

        # Behind the box = center - dir * margin
        px = cx - dir_x * margin
        py = cy - dir_y * margin
        return px, py

    def _get_recontact_point(self, box_name: str, margin: float = 0.42) -> Tuple[float, float]:
        """
        Point close behind the box to recover contact if the Husky
        separates from it during PUSH.
        """
        box = self.sim.boxes[box_name]
        cx, cy = box.center()
        dir_x, dir_y = self._get_push_direction(box_name)

        rx = cx - dir_x * margin
        ry = cy - dir_y * margin
        return rx, ry

    def _get_push_heading(self, box_name: str) -> float:
        """Desired Husky orientation during the push."""
        dir_x, dir_y = self._get_push_direction(box_name)
        return math.atan2(dir_y, dir_x)

    def _front_lidar_min(self) -> float:
        """
        Approximate minimum distance in front of the Husky using a small angular window.
        """
        angles, ranges = self.sim.simulate_lidar_2d(
            robot_name="husky",
            n_beams=181,
            max_range=8.0,
            fov_deg=180.0
        )
        front_mask = np.abs(angles) < math.radians(10.0)
        if np.any(front_mask):
            return float(np.min(ranges[front_mask]))
        return 8.0
    
    def _is_in_contact_with_box(self, box_name: str, extra_margin: float = 0.06) -> bool:
        """
        Checks whether the Husky is close enough to a box to be
        considered in pushing contact.
        """
        robot = self.sim.robots["husky"]
        box = self.sim.boxes[box_name]

        cx, cy = box.center()
        d = distance((robot.x, robot.y), (cx, cy))

        contact_threshold = robot.radius + 0.5 * math.hypot(box.w, box.h) + extra_margin
        return d <= contact_threshold

    # -------------------------------------------------------------------------
    # Simple continuous control
    # -------------------------------------------------------------------------

    def _go_to_point(self, goal_x: float, goal_y: float) -> Tuple[float, float]:
        """
        Simple proportional control for pose->point navigation.
        Returns (v_cmd, omega_cmd).
        """
        husky = self.sim.robots["husky"]
        dx = goal_x - husky.x
        dy = goal_y - husky.y
        rho = math.hypot(dx, dy)

        desired_heading = math.atan2(dy, dx)
        alpha = wrap_angle(desired_heading - husky.theta)

        v_cmd = self.k_rho * rho
        omega_cmd = self.k_alpha * alpha

        # Reduce forward speed if still very poorly oriented
        if abs(alpha) > math.radians(35.0):
            v_cmd *= 0.25

        v_cmd = clamp(v_cmd, -0.8, 0.8)
        omega_cmd = clamp(omega_cmd, -1.6, 1.6)

        return v_cmd, omega_cmd

    def _align_to_heading(self, theta_des: float) -> Tuple[float, float]:
        """
        Pure orientation alignment before pushing.
        """
        husky = self.sim.robots["husky"]
        e = wrap_angle(theta_des - husky.theta)

        v_cmd = 0.0
        omega_cmd = clamp(self.k_push_heading * e, -1.2, 1.2)
        return v_cmd, omega_cmd
    
    def _go_to_park_pose(self) -> Tuple[float, float]:
        """
        Control to move to the final parking pose.
        """
        gx, gy, _ = self.park_pose
        return self._go_to_point(gx, gy)

    def _park_pose_reached(self) -> bool:
        """
        Checks whether the Husky has already reached the parking zone.
        """
        husky = self.sim.robots["husky"]
        gx, gy, gtheta = self.park_pose

        pos_ok = distance((husky.x, husky.y), (gx, gy)) < self.pos_tol
        ang_ok = abs(wrap_angle(gtheta - husky.theta)) < self.ang_tol

        return pos_ok and ang_ok

    def _push_command(self, box_name: str) -> Tuple[float, float]:
        """
        Command during the pushing phase.
        Keeps the push orientation and also slightly corrects toward
        the box's current center to avoid losing contact.
        """
        husky = self.sim.robots["husky"]
        box = self.sim.boxes[box_name]
        cx, cy = box.center()

        # ideal push orientation
        theta_push = self._get_push_heading(box_name)

        # orientation toward the box's current center
        theta_box = math.atan2(cy - husky.y, cx - husky.x)

        # blend of both references
        theta_des = wrap_angle(0.75 * theta_push + 0.25 * theta_box)

        e = wrap_angle(theta_des - husky.theta)

        v_cmd = self.push_speed_cmd
        omega_cmd = clamp(1.6 * e, -0.9, 0.9)

        if abs(e) > math.radians(20.0):
            v_cmd *= 0.40

        return v_cmd, omega_cmd
    
    def _recover_contact_command(self, box_name: str) -> Tuple[float, float]:
        """
        Command to move back closer to the box when contact was lost
        during the pushing phase.
        """
        gx, gy = self._get_recontact_point(box_name)
        v_cmd, omega_cmd = self._go_to_point(gx, gy)

        # more conservative than normal navigation
        v_cmd = clamp(v_cmd, 0.0, 0.35)
        omega_cmd = clamp(omega_cmd, -1.0, 1.0)
        return v_cmd, omega_cmd

    # -------------------------------------------------------------------------
    # Integrating the Husky into the scenario
    # -------------------------------------------------------------------------

    def _apply_motion(self, v_cmd: float, omega_cmd: float) -> Tuple[float, float, float, float, float, float]:
        """
        Converts (v_cmd, omega_cmd) to wheels, obtains (v_meas, omega_meas)
        and updates the Husky's pose in the simulator.
        """
        wR1, wR2, wL1, wL2 = self.husky.inverse_kinematics(v_cmd, omega_cmd)
        v_meas, omega_meas = self.husky.forward_kinematics(wR1, wR2, wL1, wL2)

        dt = self.sim.dt
        robot = self.sim.robots["husky"]

        theta_mid = robot.theta + 0.5 * omega_meas * dt
        new_x = robot.x + v_meas * math.cos(theta_mid) * dt
        new_y = robot.y + v_meas * math.sin(theta_mid) * dt
        new_theta = wrap_angle(robot.theta + omega_meas * dt)

        self.sim.set_robot_pose("husky", new_x, new_y, new_theta)

        return wR1, wR2, wL1, wL2, v_meas, omega_meas

    def _log_step(
        self,
        v_cmd: float,
        omega_cmd: float,
        v_meas: float,
        omega_meas: float,
        wR1: float,
        wR2: float,
        wL1: float,
        wL2: float
    ) -> None:
        """Saves a sample in the Husky's local log."""
        husky = self.sim.robots["husky"]
        lidar_front = self._front_lidar_min()

        snapshot = {
            "B1": self.sim.boxes["B1"].center(),
            "B2": self.sim.boxes["B2"].center(),
            "B3": self.sim.boxes["B3"].center(),
        }

        self.log.append(
            t=self.sim.time,
            x=husky.x,
            y=husky.y,
            theta=husky.theta,
            v_cmd=v_cmd,
            omega_cmd=omega_cmd,
            v_meas=v_meas,
            omega_meas=omega_meas,
            wR1=wR1,
            wR2=wR2,
            wL1=wL1,
            wL2=wL2,
            target_box=self.current_box if self.current_box else "NONE",
            state=self.state,
            lidar_front_min=lidar_front,
            boxes_snapshot=snapshot
        )

    # -------------------------------------------------------------------------
    # State machine
    # -------------------------------------------------------------------------

    def update(self) -> None:
        """
        Executes one step of the Husky's planner/control.
        """
        v_cmd = 0.0
        omega_cmd = 0.0
        note = ""

        if self.state == "SELECT_BOX":
            self.current_box = self._select_next_box()
            if self.current_box is None:
                self.state = "PARK"
                note = "Corridor cleared; Husky is heading to park"
            else:
                self.state = "GO_PREPUSH"
                note = f"New target box: {self.current_box}"

        elif self.state == "GO_PREPUSH":
            assert self.current_box is not None
            gx, gy = self._get_prepush_point(self.current_box)
            v_cmd, omega_cmd = self._go_to_point(gx, gy)

            husky = self.sim.robots["husky"]
            if distance((husky.x, husky.y), (gx, gy)) < self.pos_tol:
                self.state = "ALIGN_PUSH"
                note = f"Reached pre-push point for {self.current_box}"
            else:
                note = f"Heading to pre-push point {self.current_box}"

        elif self.state == "ALIGN_PUSH":
            assert self.current_box is not None
            theta_des = self._get_push_heading(self.current_box)
            v_cmd, omega_cmd = self._align_to_heading(theta_des)

            husky = self.sim.robots["husky"]
            e = wrap_angle(theta_des - husky.theta)
            if abs(e) < self.ang_tol:
                self.state = "PUSH"
                note = f"Aligned to push {self.current_box}"
            else:
                note = f"Aligning with {self.current_box}"

        elif self.state == "PUSH":
            assert self.current_box is not None

            # If it has already fully exited, move to the next box
            if self.sim.is_box_out_of_corridor(self.current_box):
                note = f"{self.current_box} completely out of the corridor"
                self.current_box = None
                self.state = "SELECT_BOX"

            else:
                # Check whether there is still real contact with the box
                in_contact = self._is_in_contact_with_box(self.current_box)

                if in_contact:
                    # Keep pushing continuously
                    v_cmd, omega_cmd = self._push_command(self.current_box)

                    pushed = self.sim.push_box_if_contact(
                        robot_name="husky",
                        box_name=self.current_box,
                        push_distance=max(0.0, v_cmd * self.sim.dt * 1.10)
                    )

                    note = f"Pushing {self.current_box} with contact"
                    if not pushed:
                        note += " (marginal contact)"

                else:
                    # Recover contact before continuing to push
                    v_cmd, omega_cmd = self._recover_contact_command(self.current_box)
                    note = f"Recovering contact with {self.current_box}"

        elif self.state == "PARK":
            gx, gy, gtheta = self.park_pose
            husky = self.sim.robots["husky"]

            # First approach the point
            if distance((husky.x, husky.y), (gx, gy)) > self.pos_tol:
                v_cmd, omega_cmd = self._go_to_park_pose()
                note = "Husky heading to the parking corner"

            else:
                # Then align the final orientation
                v_cmd, omega_cmd = self._align_to_heading(gtheta)
                note = "Husky aligning while parking"

                if self._park_pose_reached():
                    self.state = "DONE"
                    v_cmd = 0.0
                    omega_cmd = 0.0
                    note = "Phase 1 complete: corridor clear and Husky parked"

        elif self.state == "DONE":
            v_cmd = 0.0
            omega_cmd = 0.0
            note = "Husky phase complete"

        else:
            raise RuntimeError(f"Unknown state: {self.state}")

        # Integrate the Husky's real motion
        wR1, wR2, wL1, wL2, v_meas, omega_meas = self._apply_motion(v_cmd, omega_cmd)

        # Record in the local log
        self._log_step(
            v_cmd=v_cmd,
            omega_cmd=omega_cmd,
            v_meas=v_meas,
            omega_meas=omega_meas,
            wR1=wR1,
            wR2=wR2,
            wL1=wL1,
            wL2=wL2
        )

        # Also record in the global world log
        self.sim.step(
            phase="husky",
            note=note
        )

    def run(self, max_steps: Optional[int] = None, verbose: bool = True) -> HuskyLog:
        """
        Runs the full Husky phase until the 3 boxes are cleared
        or the step limit is reached.
        """
        if max_steps is None:
            max_steps = self.max_steps

        # Save initial snapshot
        self.sim.record_state(phase="husky_init", note="Start of Husky phase")

        for k in range(max_steps):
            self.update()

            if verbose and (k % 50 == 0 or self.state == "DONE"):
                print(
                    f"[husky] step={k:04d} | state={self.state:>10s} | "
                    f"target={self.current_box} | "
                    f"cleared={self.sim.all_large_boxes_cleared()}"
                )

            if self.state == "DONE" and self.sim.all_large_boxes_cleared():
                break

        return self.log


# =============================================================================
# Husky plots
# =============================================================================

def plot_husky_phase_results(
    sim: WarehouseSim,
    log: HuskyLog,
    title: str = "Husky - Phase 1: corridor clearing",
    save_path: Optional[str] = None
):
    """
    Generates a summary figure for the Husky's phase 1.

    Subplots:
        1) XY trajectory + boxes
        2) wheel speeds
        3) commanded vs measured v, omega
        4) minimum front LiDAR distance
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(title, fontsize=15, fontweight="bold")

    # -----------------------------------------------------------------
    # 1) XY trajectory
    # -----------------------------------------------------------------
    ax = axes[0, 0]
    ax.set_title("Husky trajectory and boxes")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlim(sim.world_xmin, sim.world_xmax)
    ax.set_ylim(sim.world_ymin, sim.world_ymax)

    # Draw zones
    sx, sy, sw, sh = sim.start_zone
    ax.add_patch(plt.Rectangle((sx, sy), sw, sh, fill=False, linestyle="--", edgecolor="tab:green"))
    cx, cy, cw, ch = sim.corridor
    ax.add_patch(plt.Rectangle((cx, cy), cw, ch, fill=False, linestyle="-", linewidth=2, edgecolor="dimgray"))
    wx, wy, ww, wh = sim.work_zone
    ax.add_patch(plt.Rectangle((wx, wy), ww, wh, fill=False, linestyle="--", edgecolor="tab:blue"))

    # Trajectory
    ax.plot(log.x, log.y, color="goldenrod", linewidth=2.5, label="Husky")
    ax.plot(log.x[0], log.y[0], "go", markersize=9, label="Start")
    ax.plot(log.x[-1], log.y[-1], "rs", markersize=9, label="End")

    # Orientation arrows
    step = max(1, len(log.t) // 20)
    for i in range(0, len(log.t), step):
        dx = 0.25 * math.cos(log.theta[i])
        dy = 0.25 * math.sin(log.theta[i])
        ax.arrow(log.x[i], log.y[i], dx, dy,
                 head_width=0.08, head_length=0.08,
                 fc="orange", ec="orange", alpha=0.75)

    # Large box trajectories
    colors = {"B1": "#8B5A2B", "B2": "#A0522D", "B3": "#CD853F"}
    for name in ("B1", "B2", "B3"):
        pts = np.array(log.box_positions[name])
        ax.plot(pts[:, 0], pts[:, 1], linestyle="--", linewidth=2,
                color=colors[name], label=f"{name}")
        ax.plot(pts[0, 0], pts[0, 1], "o", color=colors[name], markersize=6)
        ax.plot(pts[-1, 0], pts[-1, 1], "s", color=colors[name], markersize=6)

    ax.legend(loc="best", fontsize=8)

    # -----------------------------------------------------------------
    # 2) Wheels
    # -----------------------------------------------------------------
    ax = axes[0, 1]
    ax.set_title("Actuators: Husky's 4 wheels")
    ax.plot(log.t, log.wR1, "b-", linewidth=2, label=r"$\omega_{R1}$ (FR)")
    ax.plot(log.t, log.wR2, "b--", linewidth=2, label=r"$\omega_{R2}$ (RR)")
    ax.plot(log.t, log.wL1, "r-", linewidth=2, label=r"$\omega_{L1}$ (FL)")
    ax.plot(log.t, log.wL2, "r--", linewidth=2, label=r"$\omega_{L2}$ (RL)")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Angular velocity [rad/s]")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="best")

    # -----------------------------------------------------------------
    # 3) v and omega: commanded vs measured
    # -----------------------------------------------------------------
    ax = axes[1, 0]
    ax2 = ax.twinx()

    l1 = ax.plot(log.t, log.v_cmd, "g--", linewidth=2, label="v_cmd [m/s]")
    l2 = ax.plot(log.t, log.v_meas, "g-", linewidth=2, label="v_meas [m/s]")

    l3 = ax2.plot(log.t, log.omega_cmd, "m--", linewidth=2, label=r"$\omega$_cmd [rad/s]")
    l4 = ax2.plot(log.t, log.omega_meas, "m-", linewidth=2, label=r"$\omega$_meas [rad/s]")

    ax.set_title("Body velocities: commanded vs measured")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Linear velocity v [m/s]", color="g")
    ax2.set_ylabel(r"Angular velocity $\omega$ [rad/s]", color="m")
    ax.tick_params(axis="y", labelcolor="g")
    ax2.tick_params(axis="y", labelcolor="m")
    ax.grid(True, alpha=0.3)

    lines = l1 + l2 + l3 + l4
    labels = [ln.get_label() for ln in lines]
    ax.legend(lines, labels, loc="best", fontsize=9)

    # -----------------------------------------------------------------
    # 4) Front LiDAR
    # -----------------------------------------------------------------
    ax = axes[1, 1]
    ax.set_title("Simulated 2D LiDAR: minimum front distance")
    ax.plot(log.t, log.lidar_front_min, color="darkorange", linewidth=2)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Distance [m]")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  -> Figure saved to {save_path}")

    return fig

def husky_log_to_demo_dict(log):
    """
    Converts the challenge's HuskyLog into a demo/base-style dictionary.
    If it is already a dict, it is left unchanged.
    """
    if isinstance(log, dict):
        return log

    return {
        't': np.array(log.t),
        'x': np.array(log.x),
        'y': np.array(log.y),
        'theta': np.array(log.theta),
        'wR1': np.array(log.wR1),
        'wR2': np.array(log.wR2),
        'wL1': np.array(log.wL1),
        'wL2': np.array(log.wL2),
        'v': np.array(log.v_meas),
        'omega': np.array(log.omega_meas),
    }

def plot_husky_demo_style(log, title="Husky A200 - Trajectory and Actuators",
                          save_path=None):
    """
    Plot using the same visual style as the course's base code.
    """
    d = husky_log_to_demo_dict(log)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # --- Subplot 1: XY trajectory ---
    ax = axes[0, 0]
    ax.plot(d['x'], d['y'], 'b-', linewidth=2, label='Trajectory')
    ax.plot(d['x'][0], d['y'][0], 'go', markersize=10, label='Start')
    ax.plot(d['x'][-1], d['y'][-1], 'rs', markersize=10, label='End')
    step = max(1, len(d['t']) // 20)
    for i in range(0, len(d['t']), step):
        dx = 0.3 * np.cos(d['theta'][i])
        dy = 0.3 * np.sin(d['theta'][i])
        ax.arrow(d['x'][i], d['y'][i], dx, dy,
                 head_width=0.08, head_length=0.08, fc='orange', ec='orange')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('Trajectory in the XY plane')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='datalim')

    # --- Subplot 2: actuators ---
    ax = axes[0, 1]
    ax.plot(d['t'], d['wR1'], 'b-', linewidth=2, label=r'$\omega_{R1}$ (FR)')
    ax.plot(d['t'], d['wR2'], 'b--', linewidth=2, label=r'$\omega_{R2}$ (RR)')
    ax.plot(d['t'], d['wL1'], 'r-', linewidth=2, label=r'$\omega_{L1}$ (FL)')
    ax.plot(d['t'], d['wL2'], 'r--', linewidth=2, label=r'$\omega_{L2}$ (RL)')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Angular velocity [rad/s]')
    ax.set_title('Actuators: Husky\'s 4 wheels')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Subplot 3: body velocities ---
    ax = axes[1, 0]
    ax2 = ax.twinx()
    l1 = ax.plot(d['t'], d['v'], 'g-', linewidth=2, label='v [m/s]')
    l2 = ax2.plot(d['t'], d['omega'], 'm-', linewidth=2, label=r'$\omega$ [rad/s]')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Linear velocity v [m/s]', color='g')
    ax2.set_ylabel(r'Angular velocity $\omega$ [rad/s]', color='m')
    ax.tick_params(axis='y', labelcolor='g')
    ax2.tick_params(axis='y', labelcolor='m')
    ax.set_title('Body velocities')
    lines = l1 + l2
    ax.legend(lines, [l.get_label() for l in lines], loc='best')
    ax.grid(True, alpha=0.3)

    # --- Subplot 4: orientation ---
    ax = axes[1, 1]
    ax.plot(d['t'], np.degrees(d['theta']), 'k-', linewidth=2)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel(r'$\theta$ [deg]')
    ax.set_title('Robot orientation')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  -> Figure saved to {save_path}")
    return fig


# =============================================================================
# Local demo
# =============================================================================

def demo_husky_pusher():
    """
    Demo of the Husky phase within the mini challenge scenario.
    """
    sim = WarehouseSim(dt=0.05)
    controller = HuskyPusher(sim=sim, terrain="grass")

    log = controller.run(verbose=True)

    print("\nHusky phase summary:")
    print(f"  Boxes out of the corridor: {sim.all_large_boxes_cleared()}")
    print(f"  Final planner state: {controller.state}")
    print(f"  Total time: {sim.time:.2f} s")

    for name in ("B1", "B2", "B3"):
        print(f"  {name} out of the corridor: {sim.is_box_out_of_corridor(name)}")

    plot_husky_phase_results(
        sim=sim,
        log=log,
        title="Husky - Phase 1: corridor clearing",
        save_path=str(PLOTS_DIR / "husky_phase1_pusher.png")
    )

    # Final snapshot
    sim.draw_world(
        phase="husky_done",
        note="Phase 1 complete",
        show_lidar=True,
        lidar_robot_name="husky"
    )
    plt.tight_layout()
    plt.show()

    return sim, controller, log


if __name__ == "__main__":
    demo_husky_pusher()