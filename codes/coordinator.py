"""
coordinator.py
--------------
State machine for the complete mini challenge:

Phase 1:
    Husky clears the corridor by pushing 3 large boxes

Phase 2:
    ANYmal crosses the corridor and reaches the work zone

Phase 3:
    3 PuzzleBots coordinate to stack small boxes
    in mandatory order: C at the bottom, B in the middle, A on top

Expected required files:
    - sim.py
    - husky_pusher.py
    - anymal_gait.py
    - puzzlebot_arm.py

This coordinator:
    - reuses the same global scenario (WarehouseSim)
    - runs the phases in real sequence
    - uses time-slotting to avoid collisions between PuzzleBots
    - records the metrics requested for the challenge
    - can generate a final animation of the scenario

"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

from sim import WarehouseSim, distance, wrap_angle, clamp
from husky_pusher import HuskyPusher, plot_husky_phase_results, plot_husky_demo_style
from anymal_gait import ANYmalGaitController, plot_anymal_phase_results
from puzzlebot_arm import PuzzleBotArm, GraspResult

PLOTS_DIR = Path(__file__).resolve().parent.parent / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# Global coordinator logging

@dataclass
class CoordinatorMetrics:
    """Global metrics for the challenge."""
    husky_time: float = 0.0
    anymal_time: float = 0.0
    puzzlebot_time: float = 0.0

    anymal_final_error: float = 0.0
    stack_final_error: float = 0.0

    detJ_violations_anymal: int = 0
    puzzlebot_collisions: int = 0

    stack_success: bool = False
    total_time: float = 0.0


@dataclass
class CoordinatorLog:
    """Summary log of coordinator states."""
    t: List[float] = field(default_factory=list)
    state: List[str] = field(default_factory=list)
    note: List[str] = field(default_factory=list)

    def append(self, t: float, state: str, note: str = "") -> None:
        self.t.append(t)
        self.state.append(state)
        self.note.append(note)


class PuzzleBotMobile:
    """
    Very simple 2D navigation model for the PuzzleBot within the coordinator.
    It does not replace the course's PuzzleBot file; it only serves to
    orchestrate phase 3 on the global scenario.
    """

    def __init__(self, sim: WarehouseSim, robot_name: str):
        self.sim = sim
        self.robot_name = robot_name

        self.v_max = 0.45
        self.omega_max = 2.0
        self.k_rho = 1.0
        self.k_alpha = 2.0

        self.pos_tol = 0.08
        self.ang_tol = math.radians(8.0)

        # Demo/base-style PuzzleBot parameters
        self.r = 0.05
        self.L = 0.19

        # Demo-style log
        self.motion_log = {
            't': [],
            'x': [],
            'y': [],
            'theta': [],
            'wR': [],
            'wL': [],
            'v': [],
            'omega': [],
        }

    def get_pose(self) -> Tuple[float, float, float]:
        r = self.sim.robots[self.robot_name]
        return r.x, r.y, r.theta

    def set_pose(self, x: float, y: float, theta: float) -> None:
        self.sim.set_robot_pose(self.robot_name, x, y, theta)

    def step_to_pose(self, gx: float, gy: float, gtheta: Optional[float] = None) -> bool:
        """
        Takes a navigation step toward a target pose.
        Returns True if the pose has already been reached.
        Also stores a PuzzleBot demo-style log.
        """
        dt = self.sim.dt
        x, y, theta = self.get_pose()

        dx = gx - x
        dy = gy - y
        rho = math.hypot(dx, dy)

        desired_heading = math.atan2(dy, dx)
        alpha = wrap_angle(desired_heading - theta)

        if rho < self.pos_tol:
            if gtheta is None:
                v = 0.0
                omega = 0.0
                done = True
            else:
                e_theta = wrap_angle(gtheta - theta)
                if abs(e_theta) < self.ang_tol:
                    v = 0.0
                    omega = 0.0
                    done = True
                else:
                    v = 0.0
                    omega = clamp(1.6 * e_theta, -self.omega_max, self.omega_max)
                    done = False
        else:
            v = clamp(self.k_rho * rho, 0.0, self.v_max)
            omega = clamp(self.k_alpha * alpha, -self.omega_max, self.omega_max)

            if abs(alpha) > math.radians(35.0):
                v *= 0.25

            done = False

        theta_mid = theta + 0.5 * omega * dt
        x_new = x + v * math.cos(theta_mid) * dt
        y_new = y + v * math.sin(theta_mid) * dt
        theta_new = wrap_angle(theta + omega * dt)

        self.set_pose(x_new, y_new, theta_new)

        # Inverse kinematics, base PuzzleBot style
        wR = (2.0 * v + omega * self.L) / (2.0 * self.r)
        wL = (2.0 * v - omega * self.L) / (2.0 * self.r)

        self.motion_log['t'].append(self.sim.time)
        self.motion_log['x'].append(x_new)
        self.motion_log['y'].append(y_new)
        self.motion_log['theta'].append(theta_new)
        self.motion_log['wR'].append(wR)
        self.motion_log['wL'].append(wL)
        self.motion_log['v'].append(v)
        self.motion_log['omega'].append(omega)

        return done

def plot_puzzlebot_demo_style(log, title="PuzzleBot - Trajectory and Actuators",
                              save_path=None):
    """
    Demo/base-style plot for the PuzzleBot.
    """
    t = np.array(log['t'])
    x = np.array(log['x'])
    y = np.array(log['y'])
    theta = np.array(log['theta'])
    wR = np.array(log['wR'])
    wL = np.array(log['wL'])
    v = np.array(log['v'])
    omega = np.array(log['omega'])

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # --- XY trajectory ---
    ax = axes[0, 0]
    ax.plot(x, y, 'b-', linewidth=2, label='Trajectory')
    if len(x) > 0:
        ax.plot(x[0], y[0], 'go', markersize=10, label='Start')
        ax.plot(x[-1], y[-1], 'rs', markersize=10, label='End')
        step = max(1, len(t) // 20)
        for i in range(0, len(t), step):
            dx = 0.05 * np.cos(theta[i])
            dy = 0.05 * np.sin(theta[i])
            ax.arrow(x[i], y[i], dx, dy,
                     head_width=0.02, head_length=0.02,
                     fc='orange', ec='orange')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('Trajectory in the XY plane')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='datalim')

    # --- Actuators ---
    ax = axes[0, 1]
    ax.plot(t, wR, 'b-', linewidth=2, label=r'$\omega_R$ (right wheel)')
    ax.plot(t, wL, 'r-', linewidth=2, label=r'$\omega_L$ (left wheel)')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Angular velocity [rad/s]')
    ax.set_title('Actuators: wheel velocities')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Body velocities ---
    ax = axes[1, 0]
    ax2 = ax.twinx()
    l1 = ax.plot(t, v, 'g-', linewidth=2, label='v [m/s]')
    l2 = ax2.plot(t, omega, 'm-', linewidth=2, label=r'$\omega$ [rad/s]')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Linear velocity v [m/s]', color='g')
    ax2.set_ylabel(r'Angular velocity $\omega$ [rad/s]', color='m')
    ax.tick_params(axis='y', labelcolor='g')
    ax2.tick_params(axis='y', labelcolor='m')
    ax.set_title('Body velocities')
    lines = l1 + l2
    ax.legend(lines, [l.get_label() for l in lines], loc='best')
    ax.grid(True, alpha=0.3)

    # --- Orientation ---
    ax = axes[1, 1]
    ax.plot(t, np.degrees(theta), 'k-', linewidth=2)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel(r'$\theta$ [deg]')
    ax.set_title('Robot orientation')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  -> Figure saved to {save_path}")
    return fig


class MissionCoordinator:
    """
    Global state machine for the challenge.

    States:
        - HUSKY_PHASE
        - ANYMAL_PHASE
        - PUZZLEBOT_PHASE
        - DONE
    """

    def __init__(self, dt: float = 0.04):
        self.sim = WarehouseSim(dt=dt)

        self.state = "HUSKY_PHASE"
        self.log = CoordinatorLog()
        self.metrics = CoordinatorMetrics()
        self.husky_log = None
        self.anymal_log = None

        self._set_box_world_center("A", 10.5, 1.3)
        self._set_box_world_center("B", 11.5, 1.1)
        self._set_box_world_center("C", 12.5, 1.3)
        

        # Controllers for the first two phases
        self.husky_controller = HuskyPusher(sim=self.sim, terrain="grass")
        self.anymal_controller = ANYmalGaitController(sim=self.sim)

        self.pb_names = ["pb1", "pb2", "pb3"]

        self.pb_mobile: Dict[str, PuzzleBotMobile] = {
            name: PuzzleBotMobile(sim=self.sim, robot_name=name)
            for name in self.pb_names
        }

        self.pb_arms: Dict[str, PuzzleBotArm] = {
            name: PuzzleBotArm(l1=0.10, l2=0.08, l3=0.06)
            for name in self.pb_names
        }

        # pb3 -> C (bottom), pb2 -> B (middle), pb1 -> A (top)
        self.stack_plan = [
            ("pb3", "C", 0),
            ("pb2", "B", 1),
            ("pb1", "A", 2),
        ]

        # Offset so each PuzzleBot positions itself beside the box before grasping
        self.pick_standoff_offset = (-0.22, 0.00)

        # Placement poses close to the target stack
        self.place_standoff_positions = {
            "pb1": (11.20, 2.35, math.radians(-90.0)),   # approaches from above
            "pb2": (10.75, 1.60, 0.0),                   # approaches from the left
            "pb3": (11.20, 0.95, math.radians(90.0)),    # approaches from below
        }

        self.pb_wait_positions = {
            "pb1": (10.0, 4.0, 0.0),
            "pb2": (10.0, 3.25, 0.0),
            "pb3": (10.0, 2.5, 0.0),
        }

        # Logs per grasp
        self.grasp_logs: Dict[str, List[GraspResult]] = {
            "pb1": [],
            "pb2": [],
            "pb3": [],
        }

        # Virtual stacking height per level
        self.stack_heights = {
            0: 0.135,   # C bottom
            1: 0.165,   # B middle
            2: 0.195,   # A top
        }
        self.train_collision_model()
        self.avoid_dir = {name: 1 for name in self.pb_names}
        self.pb_status = {name: "IDLE" for name in self.pb_names}

    def train_collision_model(self):
        # Features: [distance, vel_difference, angle_difference]
        X = np.array([
            [0.1, 0.5, 0.2],
            [0.2, 0.4, 0.3],
            [0.5, 0.2, 0.1],
            [1.0, 0.1, 0.05],
            [0.15, 0.6, 0.4],
            [0.8, 0.1, 0.1]
        ])

        # 1 = high risk, 0 = low
        y = np.array([1, 1, 0, 0, 1, 0])

        model = LogisticRegression()
        model.fit(X, y)

        self.collision_model = model

    def collision_risk(self, robot1, robot2):

        x1, y1, th1 = self.pb_mobile[robot1].get_pose()
        x2, y2, th2 = self.pb_mobile[robot2].get_pose()

        # distance
        dist = math.hypot(x1 - x2, y1 - y2)

        # velocities
        log1 = self.pb_mobile[robot1].motion_log
        log2 = self.pb_mobile[robot2].motion_log

        v1 = log1['v'][-1] if log1['v'] else 0.0
        v2 = log2['v'][-1] if log2['v'] else 0.0

        dv = abs(v1 - v2)

        # orientation
        dtheta = abs(wrap_angle(th1 - th2))

        X = np.array([[dist, dv, dtheta]])
        risk = self.collision_model.predict(X)[0]

        return risk  # 0 or 1

    def _send_other_puzzlebots_to_wait(self, active_robot: str) -> None:
        """
        Moves the PuzzleBots that are not working to their waiting zones,
        so they don't invade the active robot's trajectory.
        """
        for robot_name in self.pb_names:

            if robot_name == active_robot:
                continue

            if self.pb_status.get(robot_name) == "DONE":
                continue

            gx, gy, gtheta = self.pb_wait_positions[robot_name]
            self._navigate_robot_to(
                robot_name=robot_name,
                gx=gx,
                gy=gy,
                gtheta=gtheta,
                phase_note=f"wait while {active_robot} works",
                max_steps=250
            )

    def _navigate_robot_via_waypoints(
        self,
        robot_name: str,
        waypoints: List[Tuple[float, float, Optional[float]]],
        phase_note: str = "",
        max_steps_per_wp: int = 250
    ) -> None:
        """
        Navigates a PuzzleBot through a sequence of waypoints.
        Each waypoint is (x, y, theta) and theta can be None.
        """
        for i, (gx, gy, gtheta) in enumerate(waypoints):
            self._navigate_robot_to(
                robot_name=robot_name,
                gx=gx,
                gy=gy,
                gtheta=gtheta,
                phase_note=f"{phase_note} | wp {i+1}/{len(waypoints)}",
                max_steps=max_steps_per_wp
            )

    # Coordinator logging

    def _record(self, state: str, note: str = "") -> None:
        self.log.append(self.sim.time, state, note)
        self.sim.record_state(phase=state.lower(), note=note)

    # Phase 1: Husky

    def run_husky_phase(self, verbose: bool = True) -> None:
        """Runs the complete phase 1."""
        t0 = self.sim.time
        self._record("HUSKY_PHASE", "Start of Husky phase")

        self.husky_log = self.husky_controller.run(verbose=verbose)
        husky_log = self.husky_log

        self.metrics.husky_time = self.sim.time - t0
        self._record(
            "HUSKY_PHASE",
            f"End of Husky phase | boxes_cleared={self.sim.all_large_boxes_cleared()}"
        )

        plot_husky_phase_results(
            sim=self.sim,
            log=husky_log,
            title="Phase 1 - Husky: clearing the corridor",
            save_path=str(PLOTS_DIR / "coordinator_husky_phase.png")
        )

    # Phase 2: ANYmal

    def run_anymal_phase(self, verbose: bool = True) -> None:
        """Runs the complete phase 2."""
        t0 = self.sim.time
        self._record("ANYMAL_PHASE", "Start of ANYmal phase")

        self.anymal_log = self.anymal_controller.run(verbose=verbose)
        anymal_log = self.anymal_log

        self.metrics.anymal_time = self.sim.time - t0
        self.metrics.anymal_final_error = self.sim.anymal_goal_error()
        self.metrics.detJ_violations_anymal = len(anymal_log.singularity_events)

        reached = getattr(self.anymal_controller, "reached_goal", False)

        self._record(
            "ANYMAL_PHASE",
            (
                f"End of ANYmal phase | reached={reached} | "
                f"err={self.metrics.anymal_final_error:.3f} m | "
                f"detJ_viol={self.metrics.detJ_violations_anymal}"
            )
        )

        plot_anymal_phase_results(
            log=anymal_log,
            title="ANYmal Phase 2: Trot Gait",
            save_path=str(PLOTS_DIR / "coordinator_anymal_phase.png")
        )

        if not reached:
            raise RuntimeError(
                f"ANYmal phase failed: did not reach p_dest_anymal. "
                f"Final error = {self.metrics.anymal_final_error:.3f} m"
            )

    # Phase 3: PuzzleBots

    def _get_box_world_center(self, box_name: str) -> Tuple[float, float]:
        """XY center of a small box in the world."""
        return self.sim.boxes[box_name].center()

    def _set_box_world_center(self, box_name: str, cx: float, cy: float) -> None:
        """Repositions a small box using its center."""
        self.sim.boxes[box_name].set_center(cx, cy)

    def _stack_point_for_level(self, level: int) -> Tuple[float, float]:
        """
        XY stacking point.
        In this 2D simulation the XY is the same; the level is handled virtually.
        """
        return self.sim.stack_point
    
    def _navigate_robot_to(self, robot_name: str, gx: float, gy: float, gtheta: Optional[float] = None,
                            phase_note: str = "", max_steps: int = 500) -> None:

        controller = self.pb_mobile[robot_name]
        if self.pb_status.get(robot_name) == "DONE":
            return

        for _ in range(max_steps):

            x, y, theta = controller.get_pose()

            # Direction to goal
            dx = gx - x
            dy = gy - y

            dist_goal = math.hypot(dx, dy)

            dir_goal_x, dir_goal_y = 0.0, 0.0
            if dist_goal > 1e-6:
                dir_goal_x = dx / dist_goal
                dir_goal_y = dy / dist_goal

            # Avoidance force (robots + boxes)
            avoid_x = 0.0
            avoid_y = 0.0

            for other in self.pb_names:
                if other == robot_name:
                    continue

                x2, y2, th2 = self.pb_mobile[other].get_pose()

                # features for ML
                dist = math.hypot(x - x2, y - y2)

                log1 = self.pb_mobile[robot_name].motion_log
                log2 = self.pb_mobile[other].motion_log

                v1 = log1['v'][-1] if log1['v'] else 0.0
                v2 = log2['v'][-1] if log2['v'] else 0.0

                dv = abs(v1 - v2)

                dtheta = abs(wrap_angle(theta - th2))

                X_ml = np.array([[dist, dv, dtheta]])

                # logistic regression
                risk_prob = self.collision_model.predict_proba(X_ml)[0][1]

                # decision using ML
                if risk_prob > 0.5:

                    rx = x - x2
                    ry = y - y2

                    norm = math.hypot(rx, ry)

                    if norm > 1e-6:
                        rx /= norm
                        ry /= norm

                        avoid_x += risk_prob * rx
                        avoid_y += risk_prob * ry

                    print(f"[ML] {robot_name} vs {other} -> risk={risk_prob:.2f}")

            for box_name, box in self.sim.boxes.items():

                if getattr(box, "stacked", False):
                    continue

                bx, by = box.center()
                dist = math.hypot(x - bx, y - by)

                d_safe = 0.5
                d_crit = 0.2

                if dist < d_safe:
                    risk = (d_safe - dist) / (d_safe - d_crit)
                    risk = np.clip(risk, 0.0, 1.0)

                    rx = x - bx
                    ry = y - by

                    norm = math.hypot(rx, ry)
                    if norm > 1e-6:
                        rx /= norm
                        ry /= norm

                        avoid_x += risk * rx
                        avoid_y += risk * ry

                    print(f"[ML] {robot_name} avoids box {box_name} | risk={risk:.2f}")

            # normalize avoidance
            norm_avoid = math.hypot(avoid_x, avoid_y)
            if norm_avoid > 1e-6:
                avoid_x /= norm_avoid
                avoid_y /= norm_avoid

            if norm_avoid < 1e-6:
                mix_x = dir_goal_x
                mix_y = dir_goal_y
            else:
                # dynamic weight
                alpha = 0.7   # toward goal

                if dist_goal < 0.5:
                    alpha = 1.0
                    avoid_x = 0.0
                    avoid_y = 0.0

                mix_x = alpha * dir_goal_x + (1 - alpha) * avoid_x
                mix_y = alpha * dir_goal_y + (1 - alpha) * avoid_y

                norm_mix = math.hypot(mix_x, mix_y)
                if norm_mix > 1e-6:
                    mix_x /= norm_mix
                    mix_y /= norm_mix

            # avoids large curves
            step_size = 0.25

            gx_new = x + mix_x * step_size
            gy_new = y + mix_y * step_size

            # Movement
            done = controller.step_to_pose(gx_new, gy_new, None)

            self.sim.step(
                phase="puzzlebot",
                note=f"{robot_name} navigating smoothly | {phase_note}"
            )

            if done:
                break
            
    def _move_box_with_robot(self, robot_name: str, box_name: str,
                            target_x: float, target_y: float,
                            level: int) -> None:
        controller = self.pb_mobile[robot_name]

        # Final approach point for this robot to the stack
        gx, gy, gth = self.place_standoff_positions[robot_name]

        # current robot position
        x, y, _ = controller.get_pose()

        # final stack point
        gx, gy, gth = self.place_standoff_positions[robot_name]

        # vector toward the stack
        dx = gx - x
        dy = gy - y
        dist = math.hypot(dx, dy)

        if dist > 1e-6:
            dx /= dist
            dy /= dist

        # lateral offset (entry side)
        side_offset = 0.35

        # perpendicular
        px = -dy
        py = dx

        # choose side based on robot
        if robot_name == "pb1":
            side = 1   # above
        elif robot_name == "pb2":
            side = -1  # left
        else:
            side = 1   # below

        # dynamic waypoint
        wx = gx + side * px * side_offset
        wy = gy + side * py * side_offset

        waypoints = [
            (wx, wy, None),
            (gx, gy, gth),
        ]

        for wp_x, wp_y, wp_th in waypoints:
            for _ in range(250):
                done = controller.step_to_pose(wp_x, wp_y, wp_th)
                x, y, theta = controller.get_pose()

                # The box follows the robot during transport
                attach_x = x + 0.12 * math.cos(theta)
                attach_y = y + 0.12 * math.sin(theta)
                self._set_box_world_center(box_name, attach_x, attach_y)

                self.sim.step(
                    phase="puzzlebot",
                    note=f"{robot_name} transporting {box_name}"
                )
                if done:
                    break

        # First leave the box near the stack, from its approach side
        x, y, theta = controller.get_pose()
        preplace_x = target_x - 0.05 * math.cos(theta)
        preplace_y = target_y - 0.05 * math.sin(theta)
        self._set_box_world_center(box_name, preplace_x, preplace_y)

        self.sim.step(
            phase="puzzlebot",
            note=f"{robot_name} brought {box_name} close to the stack from its side"
        )

        # Exact final placement
        self._set_box_world_center(box_name, target_x, target_y)
        self.sim.boxes[box_name].stacked = True
        self.sim.step(
            phase="puzzlebot",
            note=f"{robot_name} placed {box_name} at level {level}"
        )


    def _execute_single_stack_task(self, robot_name: str, box_name: str, level: int) -> None:
        """
        Executes an individual stacking task for a PuzzleBot already deployed
        in the work zone.
        """
        self.pb_status[robot_name] = "ACTIVE"
        # Send the other robots to wait before this one starts
        self._send_other_puzzlebots_to_wait(active_robot=robot_name)

        box = self.sim.boxes[box_name]
        bx, by = box.center()


        # 1) Navigate to pick-up position
        pick_x = bx + self.pick_standoff_offset[0]
        pick_y = by + self.pick_standoff_offset[1]
        pick_theta = 0.0

        self._navigate_robot_to(
            robot_name=robot_name,
            gx=pick_x,
            gy=pick_y,
            gtheta=pick_theta,
            phase_note=f"approach {box_name}"
        )

        # 2) Grasp with the arm
        arm = self.pb_arms[robot_name]
        arm.q = np.array([0.0, 0.15, -0.60], dtype=float)

        local_pick = np.array([0.09, 0.00, 0.135], dtype=float)
        grasp_result_pick = arm.grasp_box(
            box_pos=local_pick,
            grip_force=5.0,
            n_points=35
        )
        self.grasp_logs[robot_name].append(grasp_result_pick)

        self.sim.step(
            phase="puzzlebot",
            note=f"{robot_name} grasped {box_name}"
        )

        # 3) Transport the box to the stack
        sx, sy = self._stack_point_for_level(level)
        self._move_box_with_robot(
            robot_name=robot_name,
            box_name=box_name,
            target_x=sx,
            target_y=sy,
            level=level
        )

        # 4) Place on the stack with force control
        local_place = np.array([0.09, 0.00, self.stack_heights[level]], dtype=float)
        grasp_result_place = arm.grasp_box(
            box_pos=local_place,
            grip_force=4.0,
            n_points=30
        )
        self.grasp_logs[robot_name].append(grasp_result_place)

        self.sim.step(
            phase="puzzlebot",
            note=f"{robot_name} applied tau=J^T f to place {box_name}"
        )
        self.pb_status[robot_name] = "DONE"

    def run_puzzlebot_phase(self, verbose: bool = True) -> None:
        """
        Runs the complete phase 3 in turns.
        This follows the PDF's recommendation to use time-slotting
        as the simplest form of coordination.
        """
        t0 = self.sim.time
        self._record("PUZZLEBOT_PHASE", "Start of PuzzleBots phase")

        # Time-slotting: one robot at a time
        for robot_name, box_name, level in self.stack_plan:
            if verbose:
                print(f"[puzzlebot] {robot_name} -> {box_name} -> level {level}")

            self._execute_single_stack_task(robot_name, box_name, level)

        self.metrics.puzzlebot_time = self.sim.time - t0

        # Final stacking error in XY
        sx, sy = self.sim.stack_point
        errs = []
        for name in ("A", "B", "C"):
            cx, cy = self.sim.boxes[name].center()
            errs.append(math.hypot(cx - sx, cy - sy))
        self.metrics.stack_final_error = max(errs) if errs else 0.0

        self.metrics.stack_success = self.sim.stack_order_is_correct()

        self._record(
            "PUZZLEBOT_PHASE",
            (
                f"End of PuzzleBots phase | stack_success={self.metrics.stack_success} | "
                f"stack_err={self.metrics.stack_final_error:.3f} m"
            )
        )

    # Complete mission

    def run_mission(self, verbose: bool = True) -> CoordinatorMetrics:
        """
        Runs the entire mission sequentially.
        """
        self._record("START", "Start of complete mission")

        self.run_husky_phase(verbose=verbose)
        self.run_anymal_phase(verbose=verbose)
        self.run_puzzlebot_phase(verbose=verbose)

        self.state = "DONE"
        self.metrics.total_time = self.sim.time

        self._record("DONE", "Complete mission finished")
        return self.metrics

    # Reports and figures

    def plot_coordinator_summary(self, save_path: Optional[str] = None):
        """
        Summary figure of the coordinator and global metrics.
        """
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        fig.suptitle("Global summary of the mini challenge", fontsize=15, fontweight="bold")

        # 1) state timeline
        ax = axes[0, 0]
        state_to_num = {
            "START": 0,
            "HUSKY_PHASE": 1,
            "ANYMAL_PHASE": 2,
            "PUZZLEBOT_PHASE": 3,
            "DONE": 4,
        }
        y = [state_to_num.get(s, -1) for s in self.log.state]
        ax.step(self.log.t, y, where="post", linewidth=2)
        ax.set_title("State timeline")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("State")
        ax.set_yticks(list(state_to_num.values()))
        ax.set_yticklabels(list(state_to_num.keys()))
        ax.grid(True, alpha=0.3)

        # 2) time per phase
        ax = axes[0, 1]
        labels = ["Husky", "ANYmal", "PuzzleBots"]
        vals = [
            self.metrics.husky_time,
            self.metrics.anymal_time,
            self.metrics.puzzlebot_time,
        ]
        ax.bar(labels, vals)
        ax.set_title("Time per phase")
        ax.set_ylabel("Time [s]")
        ax.grid(True, axis="y", alpha=0.3)

        # 3) numeric metrics
        ax = axes[1, 0]
        ax.axis("off")
        text = (
            f"Total time: {self.metrics.total_time:.2f} s\n"
            f"ANYmal final error: {self.metrics.anymal_final_error:.3f} m\n"
            f"ANYmal det(J) violations: {self.metrics.detJ_violations_anymal}\n"
            f"Final stacking error: {self.metrics.stack_final_error:.3f} m\n"
            f"PuzzleBot collisions: {self.metrics.puzzlebot_collisions}\n"
            f"Correct C-B-A stacking: {self.metrics.stack_success}"
        )
        ax.text(
            0.05, 0.95, text,
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=11,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
        )
        ax.set_title("Global metrics")

        # 4) final grasp/place torques per robot
        ax = axes[1, 1]
        robot_names = []
        tau_norms = []
        for robot_name in self.pb_names:
            for res in self.grasp_logs[robot_name]:
                robot_names.append(robot_name)
                tau_norms.append(float(np.linalg.norm(res.torques[-1])))

        if tau_norms:
            ax.bar(range(len(tau_norms)), tau_norms)
            ax.set_xticks(range(len(tau_norms)))
            ax.set_xticklabels(robot_names, rotation=45)
        ax.set_title(r"Final torque norm ($\tau = J^T f$)")
        ax.set_ylabel("||tau|| [N·m]")
        ax.grid(True, axis="y", alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  -> Figure saved to {save_path}")
        return fig

    def show_final_world(self, save_path: Optional[str] = None):
        """Shows the final snapshot of the world."""
        fig, ax = self.sim.draw_world(
            phase="done",
            note="Final state of the mission",
            show_lidar=False
        )
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  -> Figure saved to {save_path}")
        return fig

    def save_animation(self, save_path: Optional[str] = None):
        """
        Generates the complete animation from the global world log.
        """
        return self.sim.animate_log(interval_ms=50, save_path=save_path)


# Main demo

def demo_coordinator():
    """
    Demo of the complete mission.
    """
    coordinator = MissionCoordinator(dt=0.04)
    metrics = coordinator.run_mission(verbose=True)

    print("\n" + "=" * 70)
    print("FINAL CHALLENGE SUMMARY")
    print("=" * 70)
    print(f"Husky time:         {metrics.husky_time:.2f} s")
    print(f"ANYmal time:        {metrics.anymal_time:.2f} s")
    print(f"PuzzleBots time:    {metrics.puzzlebot_time:.2f} s")
    print(f"Total time:         {metrics.total_time:.2f} s")
    print(f"ANYmal final error: {metrics.anymal_final_error:.3f} m")
    print(f"det(J) violations:  {metrics.detJ_violations_anymal}")
    print(f"Stacking error:     {metrics.stack_final_error:.3f} m")
    print(f"Correct stacking:   {metrics.stack_success}")

    coordinator.plot_coordinator_summary(save_path=str(PLOTS_DIR / "coordinator_summary.png"))
    coordinator.show_final_world(save_path=str(PLOTS_DIR / "coordinator_final_world.png"))

    # --- Base/demo style plots ---
    if coordinator.husky_log is not None:
        plot_husky_demo_style(
            coordinator.husky_log,
            title="Husky Phase 1: Clearing the Corridor",
            save_path=str(PLOTS_DIR / "coordinator_husky_demo_style.png")
        )

    if coordinator.anymal_log is not None:
        plot_anymal_phase_results(
            log=coordinator.anymal_log,
            title="ANYmal Phase 2: Trot Gait",
            save_path=str(PLOTS_DIR / "coordinator_anymal_demo_style.png")
        )

    for pb_name in coordinator.pb_names:
        pb_log = coordinator.pb_mobile[pb_name].motion_log
        if len(pb_log['t']) > 0:
            plot_puzzlebot_demo_style(
                pb_log,
                title=f"PuzzleBot {pb_name.upper()} Phase 3",
                save_path=str(PLOTS_DIR / f"coordinator_{pb_name}_demo_style.png")
            )

    # Optional animation
    anim = coordinator.save_animation(save_path=None)

    plt.show()
    return coordinator, metrics, anim


if __name__ == "__main__":
    demo_coordinator()