"""
puzzlebot_arm.py
----------------
Mini brazo 3 DoF montado sobre un PuzzleBot.

Configuración cinemática adoptada:
    - q1: rotación de base (yaw) alrededor de z
    - q2: articulación del hombro en un plano vertical
    - q3: articulación del codo en el mismo plano vertical

Geometría:
    - l1: altura fija del soporte/base respecto al origen del brazo
    - l2: longitud del primer eslabón
    - l3: longitud del segundo eslabón

Marco base del brazo:
    - origen en la base del brazo sobre el PuzzleBot
    - eje z hacia arriba
    - x,y en el plano horizontal

Este módulo incluye:
    - FK
    - IK geométrica cerrada
    - Jacobiano analítico 3x3
    - mapeo fuerza->torques: tau = J^T f
    - trayectoria cartesiana simple para grasp_box
    - utilidades de demo y plots

Autores: 
Josue Ureña Valencia				IRS | A01738940
César Arellano Arellano			    IRS | A00839373
Jose Eduardo Sanchez Martinez		IRS | A01738476
Rafael André Gamiz Salazar			IRS | A00838280
Curso: TE3002B - Robots Móviles Terrestres
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math
import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# Utilidades
# =============================================================================

def clamp(value: float, low: float, high: float) -> float:
    """Satura un valor al intervalo [low, high]."""
    return max(low, min(high, value))


def wrap_angle(theta: float) -> float:
    """Normaliza un ángulo a [-pi, pi]."""
    return math.atan2(math.sin(theta), math.cos(theta))


def linspace_points(p0: np.ndarray, p1: np.ndarray, n: int) -> np.ndarray:
    """Interpola linealmente n puntos entre p0 y p1."""
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    return np.linspace(p0, p1, n)


# =============================================================================
# Resultado de un grasp
# =============================================================================

@dataclass
class GraspResult:
    """
    Resultado de la maniobra de agarre.

    Campos:
        cartesian_path: trayectoria del efector final en espacio cartesiano
        joint_path: trayectoria articular correspondiente
        torques: torques articulares estimados por tau = J^T f
        final_q: configuración articular final
        reached: si se alcanzó la caja sin error de IK significativo
    """
    cartesian_path: np.ndarray
    joint_path: np.ndarray
    torques: np.ndarray
    final_q: np.ndarray
    reached: bool


# =============================================================================
# Mini brazo del PuzzleBot
# =============================================================================

class PuzzleBotArm:
    """
    Mini brazo planar de 3 DoF montado sobre un PuzzleBot.

    Interpretación usada:
        q1 = yaw de base
        q2 = pitch del hombro
        q3 = pitch del codo

    Con l1 como altura fija de la base:
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

        # Límites razonables de seguridad
        self.q_min = np.array([-math.pi, -1.4, -2.5], dtype=float)
        self.q_max = np.array([+math.pi, +1.4, +2.5], dtype=float)

    # -------------------------------------------------------------------------
    # Cinemática directa
    # -------------------------------------------------------------------------

    def forward_kinematics(self, q: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calcula la posición (x, y, z) del efector final.

        Si q es None, usa self.q.
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
    # Cinemática inversa
    # -------------------------------------------------------------------------

    def inverse_kinematics(self, p_des: np.ndarray) -> np.ndarray:
        """
        IK geométrica cerrada -> (q1, q2, q3).

        Estrategia:
            1) q1 se obtiene del plano XY
            2) se resuelve un manipulador 2R en el plano (rho, z-l1)

        Convención:
            se elige la rama "codo abajo" por defecto (q3 negativo)
            si no es alcanzable exactamente, se proyecta al workspace.
        """
        x, y, z = map(float, p_des)

        # q1 por proyección horizontal
        q1 = math.atan2(y, x)

        # problema 2R en (rho, z_hat)
        rho = math.hypot(x, y)
        z_hat = z - self.l1

        # Alcance máximo y mínimo
        r2 = rho**2 + z_hat**2
        cos_q3 = (r2 - self.l2**2 - self.l3**2) / (2.0 * self.l2 * self.l3)
        cos_q3 = clamp(cos_q3, -1.0, 1.0)

        # Rama preferida: codo abajo
        sin_q3 = -math.sqrt(max(0.0, 1.0 - cos_q3**2))
        q3 = math.atan2(sin_q3, cos_q3)

        k1 = self.l2 + self.l3 * math.cos(q3)
        k2 = self.l3 * math.sin(q3)
        q2 = math.atan2(z_hat, rho) - math.atan2(k2, k1)

        q = np.array([q1, q2, q3], dtype=float)
        q = np.clip(q, self.q_min, self.q_max)
        return q

    # -------------------------------------------------------------------------
    # Jacobiano
    # -------------------------------------------------------------------------

    def jacobian(self, q: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Jacobiano analítico 3x3 del efector final.

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
        """Determinante del Jacobiano."""
        return float(np.linalg.det(self.jacobian(q)))

    def is_singular(self, q: Optional[np.ndarray] = None, tol: float = 1e-5) -> bool:
        """Indica si la configuración está cerca de singularidad."""
        return abs(self.det_jacobian(q)) < tol

    # -------------------------------------------------------------------------
    # Fuerza a torque
    # -------------------------------------------------------------------------

    def force_to_torque(self, f_tip: np.ndarray, q: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Mapea una fuerza en el efector a torques articulares:
            tau = J^T * f
        """
        f_tip = np.asarray(f_tip, dtype=float).reshape(3)
        J = self.jacobian(q)
        return J.T @ f_tip

    # -------------------------------------------------------------------------
    # Trayectorias
    # -------------------------------------------------------------------------

    def current_pose(self) -> np.ndarray:
        """Retorna la pose cartesiana actual del efector final."""
        return self.forward_kinematics(self.q)

    def cartesian_trajectory(self, p_start: np.ndarray, p_goal: np.ndarray, n_points: int = 30) -> np.ndarray:
        """
        Genera una trayectoria cartesiana lineal desde p_start hasta p_goal.
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
        Mueve el efector a box_pos y aplica una fuerza vertical de agarre.

        Pasos:
            1) Generar trayectoria cartesiana desde pose actual a box_pos.
            2) Para cada punto, resolver IK.
            3) En el punto final, aplicar una fuerza vertical hacia abajo:
                   f = [0, 0, -grip_force]
               y convertirla a torques con tau = J^T f.

        Nota:
            Aquí el 'grip' es una abstracción cinemática/estática para cumplir
            con el criterio del reto y poder loguear torques.
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

            # Sin contacto en el trayecto; torque de fuerza solo al final
            if i < n_points - 1:
                torques[i, :] = np.zeros(3)
            else:
                f_contact = np.array([0.0, 0.0, -abs(grip_force)], dtype=float)
                torques[i, :] = self.force_to_torque(f_contact, q=q_i)

            # Verificación de error FK
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
# Tests y demos
# =============================================================================

def unit_test_fk_ik(arm: PuzzleBotArm, test_points: List[np.ndarray]) -> None:
    """
    Test simple de consistencia FK/IK.
    """
    print("=" * 70)
    print("TEST FK / IK - PuzzleBotArm")
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
    Grafica la trayectoria cartesiana, ángulos articulares y torques.
    """
    t = np.arange(len(result.cartesian_path))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Trayectoria 3D proyectada en XZ
    ax = axes[0, 0]
    ax.plot(result.cartesian_path[:, 0], result.cartesian_path[:, 2], "b-", linewidth=2)
    ax.plot(result.cartesian_path[0, 0], result.cartesian_path[0, 2], "go", markersize=8, label="Inicio")
    ax.plot(result.cartesian_path[-1, 0], result.cartesian_path[-1, 2], "rs", markersize=8, label="Fin")
    ax.set_title("Trayectoria cartesiana (plano XZ)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Trayectoria XY
    ax = axes[0, 1]
    ax.plot(result.cartesian_path[:, 0], result.cartesian_path[:, 1], "m-", linewidth=2)
    ax.plot(result.cartesian_path[0, 0], result.cartesian_path[0, 1], "go", markersize=8)
    ax.plot(result.cartesian_path[-1, 0], result.cartesian_path[-1, 1], "rs", markersize=8)
    ax.set_title("Trayectoria cartesiana (plano XY)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")

    # Ángulos articulares
    ax = axes[1, 0]
    ax.plot(t, result.joint_path[:, 0], label="q1")
    ax.plot(t, result.joint_path[:, 1], label="q2")
    ax.plot(t, result.joint_path[:, 2], label="q3")
    ax.set_title("Trayectoria articular")
    ax.set_xlabel("Muestra")
    ax.set_ylabel("Ángulo [rad]")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Torques
    ax = axes[1, 1]
    ax.plot(t, result.torques[:, 0], label=r"$\tau_1$")
    ax.plot(t, result.torques[:, 1], label=r"$\tau_2$")
    ax.plot(t, result.torques[:, 2], label=r"$\tau_3$")
    ax.set_title(r"Torques por $\tau = J^T f$")
    ax.set_xlabel("Muestra")
    ax.set_ylabel("Torque [N·m]")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  -> Figura guardada en {save_path}")

    return fig


def demo_puzzlebot_arm():
    """
    Demo principal del mini brazo.
    """
    arm = PuzzleBotArm(l1=0.10, l2=0.08, l3=0.06)

    # Configuración inicial cómoda
    arm.q = np.array([0.0, 0.15, -0.60], dtype=float)

    # Algunos puntos de prueba alcanzables
    test_points = [
        np.array([0.10, 0.00, 0.14]),
        np.array([0.08, 0.04, 0.13]),
        np.array([0.06, -0.05, 0.12]),
    ]

    unit_test_fk_ik(arm, test_points)

    # Simulación de agarre de caja
    box_pos = np.array([0.09, 0.03, 0.135], dtype=float)
    result = arm.grasp_box(box_pos=box_pos, grip_force=5.0, n_points=35)

    print("\nResumen grasp_box:")
    print(f"  reached = {result.reached}")
    print(f"  final_q = {np.round(result.final_q, 4)}")
    print(f"  torque final = {np.round(result.torques[-1], 4)}")

    plot_grasp_result(
        result,
        title="PuzzleBotArm - Trayectoria y Torques",
        save_path="puzzlebot_arm_grasp.png"
    )

    plt.show()

    return arm, result


if __name__ == "__main__":
    demo_puzzlebot_arm()