"""
puzzlebot_arm.py
----------------
Mini 3-DoF arm mounted on a PuzzleBot.

Kinematic configuration adopted:
    - q1: base rotation (yaw) around z
    - q2: shoulder joint in a vertical plane
    - q3: elbow joint in the same vertical plane

Geometry:
    - l1: fixed height of the support/base relative to the arm origin
    - l2: length of the first link
    - l3: length of the second link

Arm base frame:
    - origin at the arm base on the PuzzleBot
    - z axis pointing up
    - x, y in the horizontal plane

This module includes:
    - FK
    - closed-form geometric IK
    - analytic 3x3 Jacobian
    - force->torque mapping: tau = J^T f
    - simple Cartesian trajectory for grasp_box
    - demo and plotting utilities

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import math
import numpy as np
import matplotlib.pyplot as plt

PLOTS_DIR = Path(__file__).resolve().parent.parent / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Utilities
# =============================================================================

def clamp(value: float, low: float, high: float) -> float:
    """Saturates a value to the interval [low, high]."""
    return max(low, min(high, value))


def wrap_angle(theta: float) -> float:
    """Normalizes an angle to [-pi, pi]."""
    return math.atan2(math.sin(theta), math.cos(theta))


def linspace_points(p0: np.ndarray, p1: np.ndarray, n: int) -> np.ndarray:
    """Linearly interpolates n points between p0 and p1."""
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    return np.linspace(p0, p1, n)


# =============================================================================
# Result of a grasp
# =============================================================================

@dataclass
class GraspResult:
    """
    Result of the grasp maneuver.

    Fields:
        cartesian_path: end-effector trajectory in Cartesian space
        joint_path: corresponding joint trajectory
        torques: joint torques estimated by tau = J^T f
        final_q: final joint configuration
        reached: whether the box was reached without significant IK error
    """
    cartesian_path: np.ndarray
    joint_path: np.ndarray
    torques: np.ndarray
    final_q: np.ndarray
    reached: bool


# =============================================================================
# PuzzleBot mini arm
# =============================================================================

class PuzzleBotArm:
    """
    Mini planar 3-DoF arm mounted on a PuzzleBot.

    Interpretation used:
        q1 = base yaw
        q2 = shoulder pitch
        q3 = elbow pitch

    With l1 as the fixed base height:
        rho = l2*cos(q2) + l3*cos(q2+q3)
        x   = rho*cos(q1)
        y   = rho*sin(q1)
        z   = l1 + l2*sin(q2) + l3*sin(q2+q3)
    """

    def __init__(self, l1: float = 0.10, l2: float = 0.08, l3: float = 0.06):
        self.l1 = float(l1)
        self.l2 = float(l2)
        self.l3 = float(l3)

        self.q = np.zeros(3, dtype=float)

        # Reasonable safety limits
        self.q_min = np.array([-math.pi, -1.4, -2.5], dtype=float)
        self.q_max = np.array([+math.pi, +1.4, +2.5], dtype=float)

    # -------------------------------------------------------------------------
    # Forward kinematics
    # -------------------------------------------------------------------------

    def forward_kinematics(self, q: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Computes the (x, y, z) position of the end effector.

        If q is None, uses self.q.
        """
        if q is not None:
            self.q = np.asarray(q, dtype=float)

        q1, q2, q3 = self.q

        rho = self.l2 * math.cos(q2) + self.l3 * math.cos(q2 + q3)

        x = rho * math.cos(q1)
        y = rho * math.sin(q1)
        z = self.l1 + self.l2 * math.sin(q2) + self.l3 * math.sin(q2 + q3)

        return np.array([x, y, z], dtype=float)

    # -------------------------------------------------------------------------
    # Inverse kinematics
    # -------------------------------------------------------------------------

    def inverse_kinematics(self, p_des: np.ndarray) -> np.ndarray:
        """
        Closed-form geometric IK -> (q1, q2, q3).

        Strategy:
            1) q1 is obtained from the XY plane
            2) a 2R manipulator is solved in the (rho, z-l1) plane

        Convention:
            the "elbow down" branch is chosen by default (negative q3)
            if it is not exactly reachable, it is projected onto the workspace.
        """
        x, y, z = map(float, p_des)

        # q1 from horizontal projection
        q1 = math.atan2(y, x)

        # 2R problem in (rho, z_hat)
        rho = math.hypot(x, y)
        z_hat = z - self.l1

        # Maximum and minimum reach
        r2 = rho**2 + z_hat**2
        cos_q3 = (r2 - self.l2**2 - self.l3**2) / (2.0 * self.l2 * self.l3)
        cos_q3 = clamp(cos_q3, -1.0, 1.0)

        # Preferred branch: elbow down
        sin_q3 = -math.sqrt(max(0.0, 1.0 - cos_q3**2))
        q3 = math.atan2(sin_q3, cos_q3)

        k1 = self.l2 + self.l3 * math.cos(q3)
        k2 = self.l3 * math.sin(q3)
        q2 = math.atan2(z_hat, rho) - math.atan2(k2, k1)

        q = np.array([q1, q2, q3], dtype=float)
        q = np.clip(q, self.q_min, self.q_max)
        return q

    # -------------------------------------------------------------------------
    # Jacobian
    # -------------------------------------------------------------------------

    def jacobian(self, q: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Analytic 3x3 Jacobian of the end effector.

        p = [x, y, z]
        q = [q1, q2, q3]
        """
        if q is None:
            q = self.q
        q1, q2, q3 = map(float, q)

        rho = self.l2 * math.cos(q2) + self.l3 * math.cos(q2 + q3)
        drho_dq2 = -self.l2 * math.sin(q2) - self.l3 * math.sin(q2 + q3)
        drho_dq3 = -self.l3 * math.sin(q2 + q3)

        dz_dq2 = self.l2 * math.cos(q2) + self.l3 * math.cos(q2 + q3)
        dz_dq3 = self.l3 * math.cos(q2 + q3)

        J = np.zeros((3, 3), dtype=float)

        # dx/dq
        J[0, 0] = -rho * math.sin(q1)
        J[0, 1] = math.cos(q1) * drho_dq2
        J[0, 2] = math.cos(q1) * drho_dq3

        # dy/dq
        J[1, 0] = +rho * math.cos(q1)
        J[1, 1] = math.sin(q1) * drho_dq2
        J[1, 2] = math.sin(q1) * drho_dq3

        # dz/dq
        J[2, 0] = 0.0
        J[2, 1] = dz_dq2
        J[2, 2] = dz_dq3

        return J

    def det_jacobian(self, q: Optional[np.ndarray] = None) -> float:
        """Determinant of the Jacobian."""
        return float(np.linalg.det(self.jacobian(q)))

    def is_singular(self, q: Optional[np.ndarray] = None, tol: float = 1e-5) -> bool:
        """Indicates whether the configuration is close to a singularity."""
        return abs(self.det_jacobian(q)) < tol

    # -------------------------------------------------------------------------
    # Force to torque
    # -------------------------------------------------------------------------

    def force_to_torque(self, f_tip: np.ndarray, q: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Maps a force at the end effector to joint torques:
            tau = J^T * f
        """
        f_tip = np.asarray(f_tip, dtype=float).reshape(3)
        J = self.jacobian(q)
        return J.T @ f_tip

    # -------------------------------------------------------------------------
    # Trajectories
    # -------------------------------------------------------------------------

    def current_pose(self) -> np.ndarray:
        """Returns the current Cartesian pose of the end effector."""
        return self.forward_kinematics(self.q)

    def cartesian_trajectory(self, p_start: np.ndarray, p_goal: np.ndarray, n_points: int = 30) -> np.ndarray:
        """
        Generates a linear Cartesian trajectory from p_start to p_goal.
        """
        return linspace_points(p_start, p_goal, n_points)

    # -------------------------------------------------------------------------
    # Grasp
    # -------------------------------------------------------------------------

    def grasp_box(
        self,
        box_pos: np.ndarray,
        grip_force: float = 5.0,
        n_points: int = 30
    ) -> GraspResult:
        """
        Moves the end effector to box_pos and applies a vertical grasping force.

        Steps:
            1) Generate a Cartesian trajectory from the current pose to box_pos.
            2) For each point, solve IK.
            3) At the final point, apply a vertical downward force:
                   f = [0, 0, -grip_force]
               and convert it to torques with tau = J^T f.

        Note:
            Here 'grip' is a kinematic/static abstraction to satisfy
            the challenge criteria and be able to log torques.
        """
        box_pos = np.asarray(box_pos, dtype=float).reshape(3)

        p_start = self.current_pose()
        cart_path = self.cartesian_trajectory(p_start, box_pos, n_points=n_points)

        joint_path = np.zeros((n_points, 3), dtype=float)
        torques = np.zeros((n_points, 3), dtype=float)

        reached = True

        for i, p in enumerate(cart_path):
            q_i = self.inverse_kinematics(p)
            self.q = q_i.copy()
            joint_path[i, :] = q_i

            # No contact along the path; force torque only at the end
            if i < n_points - 1:
                torques[i, :] = np.zeros(3)
            else:
                f_contact = np.array([0.0, 0.0, -abs(grip_force)], dtype=float)
                torques[i, :] = self.force_to_torque(f_contact, q=q_i)

            # FK error check
            p_check = self.forward_kinematics(q_i)
            if np.linalg.norm(p_check - p) > 3e-2:
                reached = False

        self.q = joint_path[-1].copy()

        return GraspResult(
            cartesian_path=cart_path,
            joint_path=joint_path,
            torques=torques,
            final_q=joint_path[-1].copy(),
            reached=reached
        )


# =============================================================================
# Tests and demos
# =============================================================================

def unit_test_fk_ik(arm: PuzzleBotArm, test_points: List[np.ndarray]) -> None:
    """
    Simple FK/IK consistency test.
    """
    print("=" * 70)
    print("FK / IK TEST - PuzzleBotArm")
    print("=" * 70)

    for i, p in enumerate(test_points, start=1):
        q = arm.inverse_kinematics(p)
        p_rec = arm.forward_kinematics(q)
        err = np.linalg.norm(p_rec - p)

        print(f"[{i}] p_des = {np.round(p, 4)}")
        print(f"    q     = {np.round(q, 4)}")
        print(f"    p_rec = {np.round(p_rec, 4)}")
        print(f"    err   = {err:.6f} m")
        print(f"    sing? = {arm.is_singular(q)}")
        print("-" * 70)


def plot_grasp_result(result: GraspResult, title: str = "PuzzleBotArm - grasp_box", save_path: Optional[str] = None):
    """
    Plots the Cartesian trajectory, joint angles, and torques.
    """
    t = np.arange(len(result.cartesian_path))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # 3D trajectory projected onto XZ
    ax = axes[0, 0]
    ax.plot(result.cartesian_path[:, 0], result.cartesian_path[:, 2], "b-", linewidth=2)
    ax.plot(result.cartesian_path[0, 0], result.cartesian_path[0, 2], "go", markersize=8, label="Start")
    ax.plot(result.cartesian_path[-1, 0], result.cartesian_path[-1, 2], "rs", markersize=8, label="End")
    ax.set_title("Cartesian trajectory (XZ plane)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # XY trajectory
    ax = axes[0, 1]
    ax.plot(result.cartesian_path[:, 0], result.cartesian_path[:, 1], "m-", linewidth=2)
    ax.plot(result.cartesian_path[0, 0], result.cartesian_path[0, 1], "go", markersize=8)
    ax.plot(result.cartesian_path[-1, 0], result.cartesian_path[-1, 1], "rs", markersize=8)
    ax.set_title("Cartesian trajectory (XY plane)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")

    # Joint angles
    ax = axes[1, 0]
    ax.plot(t, result.joint_path[:, 0], label="q1")
    ax.plot(t, result.joint_path[:, 1], label="q2")
    ax.plot(t, result.joint_path[:, 2], label="q3")
    ax.set_title("Joint trajectory")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Angle [rad]")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Torques
    ax = axes[1, 1]
    ax.plot(t, result.torques[:, 0], label=r"$\tau_1$")
    ax.plot(t, result.torques[:, 1], label=r"$\tau_2$")
    ax.plot(t, result.torques[:, 2], label=r"$\tau_3$")
    ax.set_title(r"Torques from $\tau = J^T f$")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Torque [N·m]")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  -> Figure saved to {save_path}")

    return fig


def demo_puzzlebot_arm():
    """
    Main demo of the mini arm.
    """
    arm = PuzzleBotArm(l1=0.10, l2=0.08, l3=0.06)

    # Comfortable initial configuration
    arm.q = np.array([0.0, 0.15, -0.60], dtype=float)

    # A few reachable test points
    test_points = [
        np.array([0.10, 0.00, 0.14]),
        np.array([0.08, 0.04, 0.13]),
        np.array([0.06, -0.05, 0.12]),
    ]

    unit_test_fk_ik(arm, test_points)

    # Box grasp simulation
    box_pos = np.array([0.09, 0.03, 0.135], dtype=float)
    result = arm.grasp_box(box_pos=box_pos, grip_force=5.0, n_points=35)

    print("\ngrasp_box summary:")
    print(f"  reached = {result.reached}")
    print(f"  final_q = {np.round(result.final_q, 4)}")
    print(f"  final torque = {np.round(result.torques[-1], 4)}")

    plot_grasp_result(
        result,
        title="PuzzleBotArm - Trajectory and Torques",
        save_path=str(PLOTS_DIR / "puzzlebot_arm_grasp.png")
    )

    plt.show()

    return arm, result


if __name__ == "__main__":
    demo_puzzlebot_arm()
