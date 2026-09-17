"""
sim.py
------
2D simulator of the full mini-challenge scenario for Mobile Robots.

This module defines:
- The warehouse map
- The blocked corridor
- The large boxes pushed by the Husky
- The small boxes A, B, C to be stacked
- The 2D states of the robots
- Geometric utilities, basic collisions and logging
- Static visualization and matplotlib animation

Designed to integrate with:
    1) husky_pusher.py
    2) anymal_gait.py
    3) puzzlebot_arm.py
    4) coordinator.py

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch
from matplotlib.animation import FuncAnimation


# =============================================================================
# Geometric utilities
# =============================================================================

def wrap_angle(theta: float) -> float:
    """Normalizes an angle to the interval [-pi, pi]."""
    return math.atan2(math.sin(theta), math.cos(theta))


def distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """2D Euclidean distance."""
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def clamp(value: float, low: float, high: float) -> float:
    """Clamps a value to [low, high]."""
    return max(low, min(high, value))


def point_in_rect(
    px: float,
    py: float,
    rx: float,
    ry: float,
    rw: float,
    rh: float
) -> bool:
    """Indicates whether a point is inside an axis-aligned rectangle."""
    return (rx <= px <= rx + rw) and (ry <= py <= ry + rh)


def rect_center(x: float, y: float, w: float, h: float) -> Tuple[float, float]:
    """Geometric center of a rectangle."""
    return (x + w / 2.0, y + h / 2.0)


# =============================================================================
# World entities
# =============================================================================

@dataclass
class Box2D:
    """
    Rectangular box in 2D.

    Attributes:
        name: identifier
        x, y: bottom-left corner [m]
        w, h: dimensions [m]
        kind: "large" or "small"
        color: color for drawing
        movable: whether it can be moved
        stacked: whether it is already part of the final stack
    """
    name: str
    x: float
    y: float
    w: float
    h: float
    kind: str = "large"
    color: str = "saddlebrown"
    movable: bool = True
    stacked: bool = False

    def center(self) -> Tuple[float, float]:
        """Returns the center of the box."""
        return rect_center(self.x, self.y, self.w, self.h)

    def as_patch(self, alpha: float = 0.85) -> Rectangle:
        """Generates the matplotlib patch to draw the box."""
        return Rectangle(
            (self.x, self.y),
            self.w,
            self.h,
            facecolor=self.color,
            edgecolor="black",
            linewidth=1.2,
            alpha=alpha
        )

    def move_by(self, dx: float, dy: float) -> None:
        """Moves the box."""
        if self.movable:
            self.x += dx
            self.y += dy

    def set_center(self, cx: float, cy: float) -> None:
        """Repositions the box using its center."""
        self.x = cx - self.w / 2.0
        self.y = cy - self.h / 2.0


@dataclass
class RobotState:
    """
    Simplified state of a mobile robot in 2D.

    Attributes:
        name: robot name
        x, y, theta: pose in the plane
        radius: approximate collision radius
        color: color for drawing it
        active: whether it should be shown
        payload: optional descriptive text
    """
    name: str
    x: float
    y: float
    theta: float = 0.0
    radius: float = 0.25
    color: str = "tab:blue"
    active: bool = True
    payload: Optional[str] = None

    def pose(self) -> Tuple[float, float, float]:
        """Returns the current pose."""
        return (self.x, self.y, self.theta)

    def set_pose(self, x: float, y: float, theta: float) -> None:
        """Updates the robot's pose."""
        self.x = x
        self.y = y
        self.theta = wrap_angle(theta)


@dataclass
class WorldLog:
    """
    Logging structure for the scenario.

    Stores temporal snapshots of the world for plotting and animation.
    """
    t: List[float] = field(default_factory=list)
    robot_states: List[Dict[str, Tuple[float, float, float]]] = field(default_factory=list)
    box_states: List[Dict[str, Tuple[float, float]]] = field(default_factory=list)
    phase: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def append(
        self,
        t: float,
        robots: Dict[str, RobotState],
        boxes: Dict[str, Box2D],
        phase: str,
        note: str = ""
    ) -> None:
        """Adds a snapshot of the current state."""
        self.t.append(t)
        self.robot_states.append(
            {name: (r.x, r.y, r.theta) for name, r in robots.items()}
        )
        self.box_states.append(
            {name: (b.x, b.y) for name, b in boxes.items()}
        )
        self.phase.append(phase)
        self.notes.append(note)


# =============================================================================
# Main challenge scenario
# =============================================================================

class WarehouseSim:
    """
    2D simulator of the collaborative warehouse.

    The world includes:
    - start zone
    - corridor (6x2 m rectangle)
    - work zone
    - three large boxes blocking the corridor
    - three small boxes A, B, C
    - stacking destination point

    World coordinates:
        X pointing right
        Y pointing up
    """

    def __init__(self, dt: float = 0.05):
        self.dt = dt
        self.time = 0.0

        # General world dimensions
        self.world_xmin = -1.0
        self.world_xmax = 13.5
        self.world_ymin = -1.5
        self.world_ymax = 5.5

        # Reference rectangles
        self.start_zone = (0.0, 1.2, 1.8, 2.0)         # x, y, w, h
        self.corridor = (2.0, 1.2, 6.0, 2.0)           # 6x2 m rectangle
        self.work_zone = (9.0, 0.6, 3.5, 3.6)

        # ANYmal destination point
        self.anymal_goal = (11.0, 3.6)

        # C-B-A stacking destination point
        self.stack_point = (11.4, 1.6)

        # Scenario robots
        self.robots: Dict[str, RobotState] = {}
        self._init_robots()

        # Large and small boxes
        self.boxes: Dict[str, Box2D] = {}
        self._init_boxes()

        # World's temporal history
        self.log = WorldLog()

        # Visual configuration
        self.title = "Mini Challenge - Collaborative Warehouse"

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------

    def _init_robots(self) -> None:
        """Initializes the scenario robots."""
        self.robots["husky"] = RobotState(
            name="husky",
            x=0.8,
            y=2.2,
            theta=0.0,
            radius=0.45,
            color="goldenrod",
            payload=None
        )

        self.robots["anymal"] = RobotState(
            name="anymal",
            x=0.9,
            y=3.0,
            theta=0.0,
            radius=0.35,
            color="tab:red",
            payload="3 PB"
        )

        self.robots["pb1"] = RobotState(
            name="pb1",
            x=1.1,
            y=3.05,
            theta=0.0,
            radius=0.16,
            color="tab:blue",
            active=False
        )
        self.robots["pb2"] = RobotState(
            name="pb2",
            x=0.9,
            y=2.85,
            theta=0.0,
            radius=0.16,
            color="tab:green",
            active=False
        )
        self.robots["pb3"] = RobotState(
            name="pb3",
            x=0.7,
            y=3.05,
            theta=0.0,
            radius=0.16,
            color="tab:purple",
            active=False
        )

    def _init_boxes(self) -> None:
        """Initializes the large corridor boxes and the small boxes in the work area."""
        # Large boxes blocking the corridor
        self.boxes["B1"] = Box2D(
            name="B1",
            x=4.0,
            y=1.55,
            w=0.65,
            h=0.65,
            kind="large",
            color="#8B5A2B"
        )
        self.boxes["B2"] = Box2D(
            name="B2",
            x=5.2,
            y=2.15,
            w=0.65,
            h=0.65,
            kind="large",
            color="#A0522D"
        )
        self.boxes["B3"] = Box2D(
            name="B3",
            x=6.35,
            y=1.75,
            w=0.65,
            h=0.65,
            kind="large",
            color="#CD853F"
        )

        # Small stacking boxes
        self.boxes["A"] = Box2D(
            name="A",
            x=10.0,
            y=1.0,
            w=0.22,
            h=0.22,
            kind="small",
            color="tomato"
        )
        self.boxes["B"] = Box2D(
            name="B",
            x=10.6,
            y=1.0,
            w=0.22,
            h=0.22,
            kind="small",
            color="cornflowerblue"
        )
        self.boxes["C"] = Box2D(
            name="C",
            x=11.2,
            y=1.0,
            w=0.22,
            h=0.22,
            kind="small",
            color="mediumseagreen"
        )

    # -------------------------------------------------------------------------
    # Scenario utilities
    # -------------------------------------------------------------------------

    def reset(self) -> None:
        """Fully resets the scenario to its initial state."""
        self.time = 0.0
        self.log = WorldLog()
        self._init_robots()
        self._init_boxes()

    def step(self, n: int = 1, phase: str = "idle", note: str = "") -> None:
        """
        Advances the world time and stores snapshots.

        Does not integrate dynamics by itself; used as the central clock.
        """
        for _ in range(n):
            self.time += self.dt
            self.record_state(phase=phase, note=note)

    def record_state(self, phase: str = "idle", note: str = "") -> None:
        """Stores the current world state in the log."""
        self.log.append(
            t=self.time,
            robots=self.robots,
            boxes=self.boxes,
            phase=phase,
            note=note
        )

    def set_robot_pose(self, name: str, x: float, y: float, theta: float) -> None:
        """Updates a robot's pose by name."""
        if name not in self.robots:
            raise KeyError(f"Robot '{name}' does not exist.")
        self.robots[name].set_pose(x, y, theta)

    def move_robot_by(self, name: str, dx: float, dy: float, dtheta: float = 0.0) -> None:
        """Moves a robot incrementally."""
        if name not in self.robots:
            raise KeyError(f"Robot '{name}' does not exist.")
        robot = self.robots[name]
        robot.set_pose(robot.x + dx, robot.y + dy, robot.theta + dtheta)

    def robot_to_box_distance(self, robot_name: str, box_name: str) -> float:
        """Distance between the robot's center and a box's center."""
        robot = self.robots[robot_name]
        box = self.boxes[box_name]
        return distance((robot.x, robot.y), box.center())

    def is_box_out_of_corridor(self, box_name: str) -> bool:
        """
        Checks whether a box has ended up completely outside the corridor rectangle.

        The box is only considered outside if its rectangle no longer intersects
        the corridor rectangle at all.
        """
        if box_name not in self.boxes:
            raise KeyError(f"Box '{box_name}' does not exist.")

        box = self.boxes[box_name]
        cx, cy, cw, ch = self.corridor

        box_left = box.x
        box_right = box.x + box.w
        box_bottom = box.y
        box_top = box.y + box.h

        corridor_left = cx
        corridor_right = cx + cw
        corridor_bottom = cy
        corridor_top = cy + ch

        intersects = not (
            box_right < corridor_left or
            box_left > corridor_right or
            box_top < corridor_bottom or
            box_bottom > corridor_top
        )

        return not intersects

    def all_large_boxes_cleared(self) -> bool:
        """Indicates whether B1, B2 and B3 are already out of the corridor."""
        targets = ["B1", "B2", "B3"]
        return all(self.is_box_out_of_corridor(name) for name in targets)

    def anymal_goal_error(self) -> float:
        """Euclidean error of the ANYmal with respect to the final goal."""
        anymal = self.robots["anymal"]
        return distance((anymal.x, anymal.y), self.anymal_goal)

    def stack_order_is_correct(self, tol_xy: float = 0.05, tol_z_virtual: float = 1e-9) -> bool:
        """
        Simplified verification of the C-B-A stacking.

        In this 2D simulation we do not model real physical height in the world.
        Instead, we consider that:
        - the boxes must end up centered on stack_point
        - and each one must be marked as 'stacked'
        - coordinator.py will keep track of the level count

        This function validates the XY projection.
        """
        sx, sy = self.stack_point
        for name in ("A", "B", "C"):
            box = self.boxes[name]
            cx, cy = box.center()
            if distance((cx, cy), (sx, sy)) > tol_xy:
                return False
            if not box.stacked:
                return False
        return True

    def activate_puzzlebots_at_work_zone(self) -> None:
        """
        Deploys the 3 PuzzleBots near the ANYmal when it reaches the work zone.
        The idea is that they look like they are getting off the robot, not that
        they appear magically at distant positions.
        """
        anymal = self.robots["anymal"]

        # Local offsets around the ANYmal to "get off"
        # pb1: front-right
        # pb2: front-left
        # pb3: back
        local_offsets = {
            "pb1": (+0.28, +0.18),
            "pb2": (+0.28, -0.18),
            "pb3": (-0.20,  0.00),
        }

        c = math.cos(anymal.theta)
        s = math.sin(anymal.theta)

        for name, (dx_local, dy_local) in local_offsets.items():
            dx_world = c * dx_local - s * dy_local
            dy_world = s * dx_local + c * dy_local

            self.robots[name].active = True
            self.robots[name].set_pose(
                anymal.x + dx_world,
                anymal.y + dy_world,
                anymal.theta
            )

    def sync_puzzlebots_on_anymal(self, offsets: Optional[Dict[str, Tuple[float, float]]] = None) -> None:
        """
        Places the 3 PuzzleBots mounted on the ANYmal using 2D offsets
        relative to the robot's center.

        offsets:
            dictionary with local displacements (dx, dy) in the ANYmal's frame.
        """
        if offsets is None:
            offsets = {
                "pb1": (+0.10, +0.08),
                "pb2": (-0.02,  0.00),
                "pb3": (+0.10, -0.08),
            }

        anymal = self.robots["anymal"]
        c = math.cos(anymal.theta)
        s = math.sin(anymal.theta)

        for name, (dx_local, dy_local) in offsets.items():
            if name not in self.robots:
                continue

            dx_world = c * dx_local - s * dy_local
            dy_world = s * dx_local + c * dy_local

            self.robots[name].active = True
            self.robots[name].set_pose(
                anymal.x + dx_world,
                anymal.y + dy_world,
                anymal.theta
            )

    # -------------------------------------------------------------------------
    # Simulated 2D LiDAR for the Husky
    # -------------------------------------------------------------------------

    def simulate_lidar_2d(
        self,
        robot_name: str = "husky",
        n_beams: int = 181,
        max_range: float = 8.0,
        fov_deg: float = 180.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulates a very simplified 2D LiDAR.

        Returns:
            angles: angles relative to the robot [rad]
            ranges: estimated distance [m]

        Simplified model:
        - only detects approximate intersection with large boxes
        - uses the center of each box and an equivalent radius
        - sufficient for a simple local planner for the challenge
        """
        if robot_name not in self.robots:
            raise KeyError(f"Robot '{robot_name}' does not exist.")

        robot = self.robots[robot_name]
        half_fov = math.radians(fov_deg / 2.0)
        angles = np.linspace(-half_fov, half_fov, n_beams)
        ranges = np.full(n_beams, max_range, dtype=float)

        large_boxes = [b for b in self.boxes.values() if b.kind == "large"]

        for i, a_rel in enumerate(angles):
            beam_theta = wrap_angle(robot.theta + a_rel)
            dx_beam = math.cos(beam_theta)
            dy_beam = math.sin(beam_theta)

            for box in large_boxes:
                cx, cy = box.center()
                vx = cx - robot.x
                vy = cy - robot.y

                proj = vx * dx_beam + vy * dy_beam
                if proj <= 0:
                    continue

                # Perpendicular distance from the box center to the ray
                perp = abs(vx * dy_beam - vy * dx_beam)

                # Equivalent box radius as a collision circle
                eq_radius = 0.5 * math.hypot(box.w, box.h)

                if perp <= eq_radius:
                    hit_range = max(0.0, proj - eq_radius)
                    if hit_range < ranges[i]:
                        ranges[i] = hit_range

        return angles, ranges

    # -------------------------------------------------------------------------
    # Synthetic RGB camera
    # -------------------------------------------------------------------------

    def render_camera(
        self,
        robot_name: str = "husky",
        img_size=(1280, 720)
    ):

        W, H = img_size

        # ============================================================
        # background image
        # ============================================================

        frame = np.ones((H, W, 3), dtype=np.uint8) * 240

        # ============================================================
        # REAL world dimensions
        # ============================================================

        world_w = self.world_xmax - self.world_xmin
        world_h = self.world_ymax - self.world_ymin

        # ============================================================
        # SINGLE SCALE
        # (SAME for x and y)
        # ============================================================

        scale = min(W / world_w, H / world_h)

        # center world in image
        x_offset = (W - scale * world_w) / 2
        y_offset = (H - scale * world_h) / 2

        # ============================================================
        # world -> image
        # ============================================================

        def world_to_camera(px, py):

            u = int(
                x_offset +
                (px - self.world_xmin) * scale
            )

            v = int(
                H - (
                    y_offset +
                    (py - self.world_ymin) * scale
                )
            )

            return u, v

        # ============================================================
        # GRID
        # ============================================================

        for gx in np.arange(self.world_xmin, self.world_xmax, 0.5):

            p1 = world_to_camera(gx, self.world_ymin)
            p2 = world_to_camera(gx, self.world_ymax)

            cv2.line(frame, p1, p2, (220,220,220), 1)

        for gy in np.arange(self.world_ymin, self.world_ymax, 0.5):

            p1 = world_to_camera(self.world_xmin, gy)
            p2 = world_to_camera(self.world_xmax, gy)

            cv2.line(frame, p1, p2, (220,220,220), 1)

        # ============================================================
        # main zones
        # ============================================================

        for rect, color in [

            (self.start_zone, (0,180,0)),
            (self.corridor, (120,120,120)),
            (self.work_zone, (180,80,0)),

        ]:

            x, y, w, h = rect

            p1 = world_to_camera(x, y)
            p2 = world_to_camera(x+w, y+h)

            cv2.rectangle(
                frame,
                p1,
                p2,
                color,
                3
            )

        # ============================================================
        # boxes
        # ============================================================

        for box in self.boxes.values():

            p1 = world_to_camera(box.x, box.y)
            p2 = world_to_camera(box.x + box.w, box.y + box.h)

            if box.kind == "large":
                color = (19,69,139)
            else:
                color = (0,140,255)

            cv2.rectangle(
                frame,
                p1,
                p2,
                color,
                -1
            )

            cx, cy = box.center()

            tx, ty = world_to_camera(cx, cy)

            cv2.putText(
                frame,
                box.name,
                (tx-10, ty+5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255,255,255),
                2
            )

        # ============================================================
        # robots
        # ============================================================

        for name, r in self.robots.items():

            if not r.active:
                continue

            u, v = world_to_camera(r.x, r.y)

            radius_px = int(r.radius * scale)

            if name == robot_name:
                color = (0,0,255)
            else:
                color = (0,255,0)

            cv2.circle(
                frame,
                (u,v),
                radius_px,
                color,
                -1
            )

            # orientation
            ux = int(
                u + 1.5 * radius_px * math.cos(r.theta)
            )

            vy = int(
                v - 1.5 * radius_px * math.sin(r.theta)
            )

            cv2.arrowedLine(
                frame,
                (u,v),
                (ux,vy),
                (0,0,0),
                3
            )

        return frame

    # -------------------------------------------------------------------------
    # Simple robot-box contact
    # -------------------------------------------------------------------------

    def push_box_if_contact(
        self,
        robot_name: str,
        box_name: str,
        push_distance: float
    ) -> bool:
        """
        Simplified push model.

        If the robot is close enough to a box, it pushes it in the direction
        of its current orientation.

        Returns:
            True if there was contact and box movement
            False otherwise
        """
        robot = self.robots[robot_name]
        box = self.boxes[box_name]

        cx, cy = box.center()
        d = distance((robot.x, robot.y), (cx, cy))

        contact_threshold = robot.radius + 0.5 * math.hypot(box.w, box.h)
        if d <= contact_threshold + 0.12:
            dx = push_distance * math.cos(robot.theta)
            dy = push_distance * math.sin(robot.theta)
            box.move_by(dx, dy)
            return True

        return False

    # -------------------------------------------------------------------------
    # Visualization
    # -------------------------------------------------------------------------

    def _draw_robot(self, ax, robot: RobotState) -> None:
        """Draws a robot as a circle with an orientation arrow."""
        if not robot.active:
            return

        body = Circle(
            (robot.x, robot.y),
            radius=robot.radius,
            facecolor=robot.color,
            edgecolor="black",
            alpha=0.85
        )
        ax.add_patch(body)

        tip_x = robot.x + 1.25 * robot.radius * math.cos(robot.theta)
        tip_y = robot.y + 1.25 * robot.radius * math.sin(robot.theta)
        arrow = FancyArrowPatch(
            (robot.x, robot.y),
            (tip_x, tip_y),
            arrowstyle="->",
            mutation_scale=13,
            linewidth=2.0,
            color="black"
        )
        ax.add_patch(arrow)

        label = robot.name.upper()
        if robot.payload:
            label += f"\n[{robot.payload}]"

        ax.text(
            robot.x,
            robot.y,
            label,
            ha="center",
            va="center",
            fontsize=8,
            color="white",
            fontweight="bold"
        )

    def _draw_static_map(self, ax) -> None:
        """Draws the fixed zones of the scenario."""
        # Start zone
        x, y, w, h = self.start_zone
        ax.add_patch(Rectangle((x, y), w, h, fill=False, linestyle="--",
                               linewidth=2.0, edgecolor="tab:green"))
        ax.text(x + 0.05, y + h + 0.08, "START ZONE", color="tab:green",
                fontsize=10, fontweight="bold")

        # Corridor
        x, y, w, h = self.corridor
        ax.add_patch(Rectangle((x, y), w, h, fill=False, linestyle="-",
                               linewidth=2.5, edgecolor="dimgray"))
        ax.text(x + 0.05, y + h + 0.08, "CORRIDOR (6x2 m)", color="dimgray",
                fontsize=10, fontweight="bold")

        # Work zone
        x, y, w, h = self.work_zone
        ax.add_patch(Rectangle((x, y), w, h, fill=False, linestyle="--",
                               linewidth=2.0, edgecolor="tab:blue"))
        ax.text(x + 0.05, y + h + 0.08, "WORK ZONE", color="tab:blue",
                fontsize=10, fontweight="bold")

        # ANYmal goal
        gx, gy = self.anymal_goal
        ax.plot(gx, gy, marker="*", markersize=14, color="tab:red")
        ax.text(gx + 0.1, gy + 0.08, "p_dest ANYmal", color="tab:red",
                fontsize=9, fontweight="bold")

        # Stacking point
        sx, sy = self.stack_point
        ax.plot(sx, sy, marker="s", markersize=10, color="black")
        ax.text(sx + 0.1, sy + 0.08, "destination stack", color="black",
                fontsize=9, fontweight="bold")

    def draw_world(
        self,
        ax=None,
        phase: str = "idle",
        note: str = "",
        show_lidar: bool = False,
        lidar_robot_name: str = "husky"
    ):
        """
        Draws the current state of the world.

        Parameters:
            ax: existing axis or None
            phase: current phase text
            note: descriptive note
            show_lidar: draws LiDAR beams
        """
        created_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
            created_fig = True
        else:
            fig = ax.figure

        ax.clear()
        ax.set_xlim(self.world_xmin, self.world_xmax)
        ax.set_ylim(self.world_ymin, self.world_ymax)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(f"{self.title} | Phase: {phase}")

        self._draw_static_map(ax)

        # Draw boxes
        for box in self.boxes.values():
            patch = box.as_patch()
            ax.add_patch(patch)
            cx, cy = box.center()
            ax.text(cx, cy, box.name, ha="center", va="center",
                    fontsize=9, fontweight="bold", color="white")

        # Draw robots
        for robot in self.robots.values():
            self._draw_robot(ax, robot)

        # Optional LiDAR
        if show_lidar and lidar_robot_name in self.robots:
            robot = self.robots[lidar_robot_name]
            angles, ranges = self.simulate_lidar_2d(robot_name=lidar_robot_name)
            step = max(1, len(angles) // 45)
            for a_rel, r in zip(angles[::step], ranges[::step]):
                beam_theta = robot.theta + a_rel
                bx = robot.x + r * math.cos(beam_theta)
                by = robot.y + r * math.sin(beam_theta)
                ax.plot([robot.x, bx], [robot.y, by], color="orange",
                        alpha=0.15, linewidth=1.0)

        # Informative text
        info_lines = [
            f"t = {self.time:.2f} s",
            f"phase = {phase}",
        ]
        if note:
            info_lines.append(note)

        info_lines.append(
            f"boxes_cleared = {self.all_large_boxes_cleared()}"
        )
        info_lines.append(
            f"anymal_error = {self.anymal_goal_error():.3f} m"
        )

        ax.text(
            0.01,
            0.99,
            "\n".join(info_lines),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.75)
        )

        if created_fig:
            plt.tight_layout()
            return fig, ax

        return ax

    # -------------------------------------------------------------------------
    # Animation from the log
    # -------------------------------------------------------------------------

    def animate_log(
        self,
        interval_ms: int = 60,
        save_path: Optional[str] = None
    ) -> FuncAnimation:
        """
        Generates a matplotlib animation from the world log.

        Requirements:
            self.log must already be filled with snapshots.
        """
        if len(self.log.t) == 0:
            raise RuntimeError("No data in the log. Call record_state() during the simulation.")

        fig, ax = plt.subplots(figsize=(12, 6))

        def update(frame_idx: int):
            ax.clear()

            # Restore snapshot
            robot_snapshot = self.log.robot_states[frame_idx]
            box_snapshot = self.log.box_states[frame_idx]
            phase = self.log.phase[frame_idx]
            note = self.log.notes[frame_idx]

            for name, (x, y, theta) in robot_snapshot.items():
                self.robots[name].set_pose(x, y, theta)

            for name, (x, y) in box_snapshot.items():
                self.boxes[name].x = x
                self.boxes[name].y = y

            self.time = self.log.t[frame_idx]
            self.draw_world(ax=ax, phase=phase, note=note, show_lidar=False)

        anim = FuncAnimation(
            fig,
            update,
            frames=len(self.log.t),
            interval=interval_ms,
            repeat=False
        )

        if save_path:
            # Requires ffmpeg or pillow depending on the format
            anim.save(save_path, dpi=120)

        return anim


# =============================================================================
# Local test demo of the simulator
# =============================================================================

def demo_sim() -> WarehouseSim:
    """
    Minimal demo to verify that the map and the log work.
    Does not solve the challenge; it only tests the scenario.
    """
    sim = WarehouseSim(dt=0.1)

    # Save initial state
    sim.record_state(phase="init", note="Initial scenario state")

    # Move the Husky slightly
    for _ in range(10):
        sim.move_robot_by("husky", dx=0.08, dy=0.0, dtheta=0.01)
        sim.step(phase="demo_husky", note="Basic movement test")

    # Activate PuzzleBots in the work zone
    sim.activate_puzzlebots_at_work_zone()
    sim.record_state(phase="demo_deploy", note="PuzzleBot deployment")

    # ============================================================
    # simultaneous visualization
    # ============================================================

    plt.ion()

    fig, ax = plt.subplots(figsize=(12,6))

    for k in range(400):

        # ========================================================
        # move robot DEMO ONLY
        # ========================================================

        sim.move_robot_by(
            "husky",
            dx=0.01,
            dy=0.0,
            dtheta=0.01
        )

        sim.step(
            phase="camera_demo",
            note="Real-time camera demo"
        )

        # ========================================================
        # update matplotlib simulator
        # ========================================================

        sim.draw_world(
            ax=ax,
            phase="camera_demo",
            note="Global view",
            show_lidar=False
        )

        fig.canvas.draw()
        fig.canvas.flush_events()

        # ========================================================
        # update OpenCV camera
        # ========================================================

        frame = sim.render_camera("husky")

        cv2.imshow(
            "Husky Camera",
            frame
        )

        key = cv2.waitKey(30)

        if key == 27:
            break

    plt.ioff()

    cv2.destroyAllWindows()

    return sim


if __name__ == "__main__":
    demo_sim()
