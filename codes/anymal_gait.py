"""
anymal_gait.py
--------------
Generador de marcha trote para el ANYmal dentro del mini reto.

Este módulo implementa:
- Modelo cinemático de una pata ANYmal (FK, IK, Jacobiano)
- Modelo simplificado del ANYmal completo con 4 patas
- Generación de trayectoria cartesiana de cada pie
- Marcha trote: patas diagonales en fase
- Monitoreo del determinante del Jacobiano en cada pata
- Estrategia simple de evitación de singularidades
- Integración con WarehouseSim (sim.py) para desplazar la base del robot
- Logging y graficación de desempeño de la Fase 2

Autor: Tu equipo
Curso: TE3002B - Robots Móviles Terrestres
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import math
import numpy as np
import matplotlib.pyplot as plt

from sim import WarehouseSim, wrap_angle, distance, clamp


# =============================================================================
# Pata del ANYmal
# =============================================================================

class ANYmalLeg:
    """
    Modelo cinemático de una pata del ANYmal con 3 DoF:
        q1 = HAA
        q2 = HFE
        q3 = KFE

    Convenciones:
        x: adelante
        y: lateral
        z: arriba
        side = +1 patas izquierdas, -1 patas derechas
    """

    def __init__(self, name: str, l0: float = 0.0585, l1: float = 0.35, l2: float = 0.33, side: int = +1):
        self.name = name
        self.l0 = l0
        self.l1 = l1
        self.l2 = l2
        self.side = side

        self.q = np.zeros(3)

        # Límites suaves de seguridad
        self.q_min = np.array([-0.72, -1.8, -2.69], dtype=float)
        self.q_max = np.array([+0.49, +1.8, -0.03], dtype=float)

    def forward_kinematics(self, q: Optional[np.ndarray] = None) -> np.ndarray:
        """
        FK analítica:
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
        IK geométrica cerrada para la pata.

        Configuración:
            rodilla hacia atrás => q3 < 0
        """
        x, y, z = map(float, p_des)

        # Resolver q1 por proyección lateral
        r_yz_sq = y**2 + z**2 - self.l0**2
        r_yz = math.sqrt(max(r_yz_sq, 1e-9))
        q1 = math.atan2(y, -z) - math.atan2(self.side * self.l0, r_yz)

        # Resolver q3 con ley de cosenos
        r_sq = x**2 + z**2
        D = (r_sq - self.l1**2 - self.l2**2) / (2.0 * self.l1 * self.l2)
        D = clamp(D, -1.0, 1.0)
        q3 = -math.acos(D)

        # Resolver q2
        alpha = math.atan2(x, -z)
        beta = math.atan2(self.l2 * math.sin(-q3),
                          self.l1 + self.l2 * math.cos(q3))
        q2 = alpha - beta

        q = np.array([q1, q2, q3], dtype=float)
        q = np.clip(q, self.q_min, self.q_max)
        return q

    def jacobian(self, q: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Jacobiano analítico 3x3.
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
        """Determinante del Jacobiano."""
        return float(np.linalg.det(self.jacobian(q)))

    def is_singular(self, q: Optional[np.ndarray] = None, tol: float = 1e-3) -> bool:
        """True si está cerca de singularidad."""
        return abs(self.det_jacobian(q)) < tol


# =============================================================================
# Robot ANYmal completo
# =============================================================================

class ANYmal:
    """
    Modelo simplificado del ANYmal:
    - 4 patas
    - base flotante 2D en el mundo del almacén
    - articulaciones internas para la marcha
    """

    LEG_NAMES = ["LF", "RF", "LH", "RH"]

    def __init__(self):
        self.legs: Dict[str, ANYmalLeg] = {
            "LF": ANYmalLeg("LF", side=+1),
            "RF": ANYmalLeg("RF", side=-1),
            "LH": ANYmalLeg("LH", side=+1),
            "RH": ANYmalLeg("RH", side=-1),
        }

        # Estado de base simplificada
        self.base_x = 0.0
        self.base_y = 0.0
        self.base_theta = 0.0

        # Payload aproximado pedido en el reto
        self.payload_mass = 6.0

        # Postura nominal articular
        self.q_nominal_leg = np.array([0.0, 0.70, -1.40], dtype=float)

    def set_base_pose(self, x: float, y: float, theta: float) -> None:
        """Actualiza la pose de la base."""
        self.base_x = x
        self.base_y = y
        self.base_theta = wrap_angle(theta)

    def get_base_pose(self) -> Tuple[float, float, float]:
        """Retorna pose de la base."""
        return self.base_x, self.base_y, self.base_theta

    def get_all_joint_angles(self) -> np.ndarray:
        """Concatena los 12 ángulos articulares."""
        return np.concatenate([self.legs[name].q for name in self.LEG_NAMES])

    def set_all_joint_angles(self, q12: np.ndarray) -> None:
        """Asigna los 12 ángulos articulares."""
        q12 = np.asarray(q12, dtype=float)
        assert q12.shape == (12,), f"Se esperaban 12 ángulos, llegó {q12.shape}"
        for i, name in enumerate(self.LEG_NAMES):
            self.legs[name].q = q12[3 * i: 3 * (i + 1)].copy()

    def get_all_foot_positions(self) -> Dict[str, np.ndarray]:
        """Posiciones de pie de las 4 patas en sus marcos locales."""
        return {name: self.legs[name].forward_kinematics() for name in self.LEG_NAMES}

    def get_all_detJ(self) -> Dict[str, float]:
        """Determinante del Jacobiano por pata."""
        return {name: self.legs[name].det_jacobian() for name in self.LEG_NAMES}


# =============================================================================
# Log de la fase ANYmal
# =============================================================================

@dataclass
class ANYmalLog:
    """Log detallado de la marcha del ANYmal."""
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
        """Agrega una muestra al log."""
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
# Generador de marcha trote
# =============================================================================

class ANYmalGaitController:
    """
    Controlador de marcha trote para la Fase 2.

    Objetivos:
    - Desplazar al ANYmal desde su posición inicial a p_destino
    - Generar trayectorias cartesianas de pie
    - Resolver IK por pata
    - Monitorear det(J)
    - Evitar singularidades reduciendo la amplitud de swing si es necesario
    """

    def __init__(self, sim: WarehouseSim):
        self.sim = sim
        self.anymal = ANYmal()

        # Sincronizar base con el mundo
        robot = self.sim.robots["anymal"]
        self.anymal.set_base_pose(robot.x, robot.y, robot.theta)

        self.log = ANYmalLog()

        # Parámetros de gait
        self.period = 0.65               # s
        self.step_height = 0.05          # m
        self.step_length = 0.1         # m
        self.nominal_y = {
            "LF": +0.052,
            "RF": -0.052,
            "LH": +0.052,
            "RH": -0.052,
        }

        # Centros nominales de pie en marco local de pata
        self.foot_centers = {
            "LF": np.array([ 0.02, +0.052, -0.52], dtype=float),
            "RF": np.array([ 0.02, -0.052, -0.52], dtype=float),
            "LH": np.array([ 0.02, +0.052, -0.52], dtype=float),
            "RH": np.array([ 0.02, -0.052, -0.52], dtype=float),
        }

        self.detJ_tol = 1e-3
        self.safe_detJ_target = 2.0e-3

        # Control base
        self.v_base_max = 0.55
        self.omega_base_max = 0.8
        self.k_rho = 0.55
        self.k_alpha = 1.1

        self.max_steps = 5000

        self.reached_goal = False

        # 2) cruzarlo por en medio
        # 3) salir
        # 4) girar y llegar a p_dest_anymal
        cx, cy, cw, ch = self.sim.corridor
        gx, gy = self.sim.anymal_goal

        self.path_waypoints = [
            (cx + 0.15 * cw, cy + 0.50 * ch),   # entrada al corredor
            (cx + 0.50 * cw, cy + 0.50 * ch),   # centro del corredor
            (cx + 0.90 * cw, cy + 0.50 * ch),   # salida del corredor
            (gx, gy),                           # destino final ANYmal
        ]
        self.current_waypoint_idx = 0
        self.waypoint_tol = 0.16

        # Offsets de montaje de los 3 PuzzleBots sobre el lomo del ANYmal
        self.pb_mount_offsets = {
            "pb1": (+0.10, +0.08),
            "pb2": (-0.02,  0.00),
            "pb3": (+0.10, -0.08),
        }

    # -------------------------------------------------------------------------
    # Trayectoria global de la base
    # -------------------------------------------------------------------------

    def _base_control_to_goal(self, goal_x: float, goal_y: float) -> Tuple[float, float]:
        """
        Control simple pose->punto para la base del ANYmal.
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
        """Retorna el waypoint actual."""
        idx = min(self.current_waypoint_idx, len(self.path_waypoints) - 1)
        return self.path_waypoints[idx]


    def _update_waypoint_progress(self) -> None:
        """Avanza al siguiente waypoint si ya se alcanzó el actual."""
        if self.current_waypoint_idx >= len(self.path_waypoints):
            return

        robot = self.sim.robots["anymal"]
        gx, gy = self._get_current_waypoint()
        err = distance((robot.x, robot.y), (gx, gy))

        if err < self.waypoint_tol and self.current_waypoint_idx < len(self.path_waypoints) - 1:
            self.current_waypoint_idx += 1

    def _integrate_base(self, v_cmd: float, omega_cmd: float) -> None:
        """
        Integra la base del ANYmal y sincroniza con sim.py.
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
        Mantiene a los 3 PuzzleBots montados sobre el ANYmal durante la marcha.
        """
        self.sim.sync_puzzlebots_on_anymal(offsets=self.pb_mount_offsets)

    # -------------------------------------------------------------------------
    # Gait: trayectoria cartesiana del pie
    # -------------------------------------------------------------------------

    def _phase_value(self, t: float) -> float:
        """Fase normalizada en [0,1)."""
        return (t / self.period) % 1.0

    def _swing_profile(self, phase: float) -> float:
        """
        Perfil suave de elevación de swing en [0,1].
        Solo activa durante la mitad positiva del ciclo.
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
        Genera posición cartesiana deseada del pie en marco local de la pata.

        Trote:
            LF + RH en fase
            RF + LH en antifase
        """
        phase = self._phase_value(t)

        if leg_name in ("LF", "RH"):
            lift = self._swing_profile(phase)
        else:
            lift = self._swing_profile((phase + 0.5) % 1.0)

        center = self.foot_centers[leg_name].copy()

        # Avance/retroceso local del pie
        x = center[0] + step_length_scale * self.step_length * (lift - 0.5 * max(0.0, 1.0 - lift))
        # Altura en swing
        z = center[2] + step_height_scale * self.step_height * lift

        # Mantener y nominal
        y = center[1]

        return np.array([x, y, z], dtype=float)

    # -------------------------------------------------------------------------
    # Singularidades
    # -------------------------------------------------------------------------

    def _compute_q12_from_cartesian_targets(
        self,
        foot_targets: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, Dict[str, float], Dict[str, np.ndarray]]:
        """
        Resuelve IK por pata y retorna:
            q12, detJ por pata, posiciones FK verificadas
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
        Genera targets cartesianos y reduce amplitudes si encuentra singularidades.
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

            # Ajustar amplitud para alejarse de singularidad
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
        """Guarda una muestra del ANYmal."""
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
    # Ejecución principal
    # -------------------------------------------------------------------------

    def run(self, verbose: bool = True) -> ANYmalLog:
        """
        Corre la Fase 2 completa:
        - trote
        - avance a p_destino
        - monitoreo de singularidades
        """
        self.reached_goal = False
        gx, gy = self.sim.anymal_goal

        # Al inicio los PuzzleBots van montados sobre el ANYmal
        self._sync_puzzlebots_with_anymal()
        self.sim.record_state(
            phase="anymal_init",
            note="Inicio fase ANYmal con 3 PuzzleBots montados"
        )

        for k in range(self.max_steps):
            t = self.sim.time

            # 1) waypoint actual
            wp_x, wp_y = self._get_current_waypoint()

            # 2) control de base hacia waypoint
            v_base_cmd, omega_base_cmd = self._base_control_to_goal(wp_x, wp_y)

            # 3) gait cartesiano + IK + monitoreo de singularidades
            q12, detJ, feet_fk, singularity_violation = self._avoid_singularities(t)

            # 4) aplicar articulaciones
            self.anymal.set_all_joint_angles(q12)

            # 5) integrar base
            self._integrate_base(v_base_cmd, omega_base_cmd)

            # 6) avanzar waypoint si ya llegó
            self._update_waypoint_progress()

            # 7) logging local
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

            # 8) logging global
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

            # éxito final
            if err < 0.15 and self.current_waypoint_idx >= len(self.path_waypoints) - 1:
                self.reached_goal = True
                break

        # Al terminar la marcha, solo desplegar si realmente llegó al destino
        if self.reached_goal:
            self.sim.activate_puzzlebots_at_work_zone()
            self.sim.record_state(
                phase="anymal_done",
                note="ANYmal llegó a p_dest y desplegó los 3 PuzzleBots"
            )
        else:
            self.sim.record_state(
                phase="anymal_failed",
                note="ANYmal no llegó a p_dest; no se despliegan PuzzleBots"
            )

        return self.log


# =============================================================================
# Gráficas de la fase ANYmal
# =============================================================================

def plot_anymal_phase_results(log, title="ANYmal - Actuadores y Trayectoria de Pies",
                              save_path=None):
    """
    Gráfica estilo demo/base del curso.
    """
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    gs = fig.add_gridspec(3, 4, hspace=0.45, wspace=0.35)

    leg_colors = {'LF': 'tab:blue', 'RF': 'tab:orange',
                  'LH': 'tab:green', 'RH': 'tab:red'}
    joint_labels = ['HAA (q1)', 'HFE (q2)', 'KFE (q3)']

    q_arr = np.array(log.q)
    t_arr = np.array(log.t)

    # --- Subplots 1-4: ángulos articulares de cada pata ---
    for i, name in enumerate(['LF', 'RF', 'LH', 'RH']):
        ax = fig.add_subplot(gs[0, i])
        q_leg = q_arr[:, 3*i:3*(i+1)]
        for j in range(3):
            ax.plot(t_arr, np.degrees(q_leg[:, j]),
                    linewidth=1.8, label=joint_labels[j])
        ax.set_title(f'Pata {name}', color=leg_colors[name], fontweight='bold')
        ax.set_xlabel('t [s]')
        ax.set_ylabel('angulo [deg]')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)

    # --- Subplot 5: altura z de todos los pies ---
    ax = fig.add_subplot(gs[1, :2])
    for name in ['LF', 'RF', 'LH', 'RH']:
        ax.plot(t_arr, np.array(log.foot_z[name]),
                color=leg_colors[name], linewidth=2, label=f'Pie {name}')
    ax.set_xlabel('Tiempo [s]')
    ax.set_ylabel('z del pie [m]')
    ax.set_title('Altura de los pies (stance vs swing)')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Subplot 6: trayectoria XZ de los pies ---
    ax = fig.add_subplot(gs[1, 2:])
    for name in ['LF', 'RF', 'LH', 'RH']:
        fx = np.array(log.foot_x[name])
        fz = np.array(log.foot_z[name])
        ax.plot(fx, fz,
                color=leg_colors[name], linewidth=2, label=f'{name}', alpha=0.7)
        ax.plot(fx[0], fz[0], 'o', color=leg_colors[name], markersize=8)
    ax.set_xlabel('x del pie [m]')
    ax.set_ylabel('z del pie [m]')
    ax.set_title('Trayectoria lateral (XZ) de los pies')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='datalim')

    # --- Subplots 7-10: velocidades articulares ---
    dt = t_arr[1] - t_arr[0] if len(t_arr) > 1 else 0.04
    dq = np.gradient(q_arr, dt, axis=0)

    for i, name in enumerate(['LF', 'RF', 'LH', 'RH']):
        ax = fig.add_subplot(gs[2, i])
        dq_leg = dq[:, 3*i:3*(i+1)]
        for j in range(3):
            ax.plot(t_arr, dq_leg[:, j],
                    linewidth=1.5, label=joint_labels[j])
        ax.set_title(f'Velocidades {name}', fontsize=10)
        ax.set_xlabel('t [s]')
        ax.set_ylabel('dq/dt [rad/s]')
        ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  -> Figura guardada en {save_path}")
    return fig


# =============================================================================
# Demo local
# =============================================================================

def demo_anymal_gait():
    """
    Demo de la Fase 2 del reto.
    """
    sim = WarehouseSim(dt=0.04)

    controller = ANYmalGaitController(sim=sim)
    log = controller.run(verbose=True)

    err = sim.anymal_goal_error()
    print("\nResumen fase ANYmal:")
    print(f"  Error final a p_destino: {err:.3f} m")
    print(f"  Cumple error < 0.15 m: {err < 0.15}")
    print(f"  Eventos de ajuste por singularidad: {len(log.singularity_events)}")
    print(f"  Tiempo total: {sim.time:.2f} s")

    plot_anymal_phase_results(
        log=log,
        title="ANYmal Fase 2: Marcha Trote",
        save_path="anymal_phase2_gait.png"
    )

    sim.draw_world(
        phase="anymal_done",
        note=f"ANYmal llegó. error={err:.3f} m",
        show_lidar=False
    )
    plt.tight_layout()
    plt.show()

    return sim, controller, log


if __name__ == "__main__":
    demo_anymal_gait()