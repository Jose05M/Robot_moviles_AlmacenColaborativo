"""
coordinator.py
--------------
Máquina de estados del mini reto completo:

Fase 1:
    Husky despeja el corredor empujando 3 cajas grandes

Fase 2:
    ANYmal cruza el corredor y llega a la zona de trabajo

Fase 3:
    3 PuzzleBots se coordinan para apilar cajas pequeñas
    en orden obligatorio: C abajo, B en medio, A arriba

Archivos requeridos esperados:
    - sim.py
    - husky_pusher.py
    - anymal_gait.py
    - puzzlebot_arm.py

Este coordinador:
    - reutiliza el mismo escenario global (WarehouseSim)
    - ejecuta las fases en secuencia real
    - usa time-slotting para evitar colisiones entre PuzzleBots
    - registra métricas pedidas en el reto
    - puede generar animación final del escenario

Autores: 
Josue Ureña Valencia				IRS | A01738940
César Arellano Arellano			    IRS | A00839373
Jose Eduardo Sanchez Martinez		IRS | A01738476
Rafael André Gamiz Salazar			IRS | A00838280
Curso: TE3002B - Robots Móviles Terrestres
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

from sim import WarehouseSim, distance, wrap_angle, clamp
from husky_pusher import HuskyPusher, plot_husky_phase_results, plot_husky_demo_style
from anymal_gait import ANYmalGaitController, plot_anymal_phase_results
from puzzlebot_arm import PuzzleBotArm, GraspResult


# =============================================================================
# Logging global del coordinador
# =============================================================================

@dataclass
class CoordinatorMetrics:
    """Métricas globales del reto."""
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
    """Bitácora resumida de estados del coordinador."""
    t: List[float] = field(default_factory=list)
    state: List[str] = field(default_factory=list)
    note: List[str] = field(default_factory=list)

    def append(self, t: float, state: str, note: str = "") -> None:
        self.t.append(t)
        self.state.append(state)
        self.note.append(note)


# =============================================================================
# Control simple de PuzzleBot base móvil
# =============================================================================

class PuzzleBotMobile:
    """
    Modelo muy simple de navegación 2D del PuzzleBot dentro del coordinador.
    No reemplaza el archivo de PuzzleBot del curso; solo sirve para orquestar
    la fase 3 sobre el escenario global.
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

        # Parámetros tipo demo/base del PuzzleBot
        self.r = 0.05
        self.L = 0.19

        # Log estilo demo
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
        Da un paso de navegación hacia una pose objetivo.
        Retorna True si la pose ya fue alcanzada.
        Además guarda un log estilo PuzzleBot demo.
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

        # Cinemática inversa estilo PuzzleBot base
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


# =============================================================================
# Coordinador principal
# =============================================================================

def plot_puzzlebot_demo_style(log, title="PuzzleBot - Trayectoria y Actuadores",
                              save_path=None):
    """
    Gráfica estilo demo/base del PuzzleBot.
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

    # --- Trayectoria XY ---
    ax = axes[0, 0]
    ax.plot(x, y, 'b-', linewidth=2, label='Trayectoria')
    if len(x) > 0:
        ax.plot(x[0], y[0], 'go', markersize=10, label='Inicio')
        ax.plot(x[-1], y[-1], 'rs', markersize=10, label='Fin')
        step = max(1, len(t) // 20)
        for i in range(0, len(t), step):
            dx = 0.05 * np.cos(theta[i])
            dy = 0.05 * np.sin(theta[i])
            ax.arrow(x[i], y[i], dx, dy,
                     head_width=0.02, head_length=0.02,
                     fc='orange', ec='orange')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('Trayectoria en el plano XY')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='datalim')

    # --- Actuadores ---
    ax = axes[0, 1]
    ax.plot(t, wR, 'b-', linewidth=2, label=r'$\omega_R$ (rueda der.)')
    ax.plot(t, wL, 'r-', linewidth=2, label=r'$\omega_L$ (rueda izq.)')
    ax.set_xlabel('Tiempo [s]')
    ax.set_ylabel('Velocidad angular [rad/s]')
    ax.set_title('Actuadores: velocidades de ruedas')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Velocidades del cuerpo ---
    ax = axes[1, 0]
    ax2 = ax.twinx()
    l1 = ax.plot(t, v, 'g-', linewidth=2, label='v [m/s]')
    l2 = ax2.plot(t, omega, 'm-', linewidth=2, label=r'$\omega$ [rad/s]')
    ax.set_xlabel('Tiempo [s]')
    ax.set_ylabel('Velocidad lineal v [m/s]', color='g')
    ax2.set_ylabel(r'Velocidad angular $\omega$ [rad/s]', color='m')
    ax.tick_params(axis='y', labelcolor='g')
    ax2.tick_params(axis='y', labelcolor='m')
    ax.set_title('Velocidades del cuerpo')
    lines = l1 + l2
    ax.legend(lines, [l.get_label() for l in lines], loc='best')
    ax.grid(True, alpha=0.3)

    # --- Orientación ---
    ax = axes[1, 1]
    ax.plot(t, np.degrees(theta), 'k-', linewidth=2)
    ax.set_xlabel('Tiempo [s]')
    ax.set_ylabel(r'$\theta$ [deg]')
    ax.set_title('Orientacion del robot')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  -> Figura guardada en {save_path}")
    return fig


class MissionCoordinator:
    """
    Máquina de estados global del reto.

    Estados:
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
        

        # Controladores de las primeras dos fases
        self.husky_controller = HuskyPusher(sim=self.sim, terrain="grass")
        self.anymal_controller = ANYmalGaitController(sim=self.sim)

        self.pb_names = ["pb1", "pb2", "pb3"]

        # Ahora sí se despliegan en la zona de trabajo y pueden moverse por su cuenta
        self.pb_mobile: Dict[str, PuzzleBotMobile] = {
            name: PuzzleBotMobile(sim=self.sim, robot_name=name)
            for name in self.pb_names
        }

        self.pb_arms: Dict[str, PuzzleBotArm] = {
            name: PuzzleBotArm(l1=0.10, l2=0.08, l3=0.06)
            for name in self.pb_names
        }

        # Asignación explícita para cumplir orden C-B-A por time-slotting
        # pb3 -> C (abajo), pb2 -> B (medio), pb1 -> A (arriba)
        self.stack_plan = [
            ("pb3", "C", 0),
            ("pb2", "B", 1),
            ("pb1", "A", 2),
        ]

        # Offset para que cada PuzzleBot se coloque a un lado de la caja antes de hacer grasp
        self.pick_standoff_offset = (-0.22, 0.00)

        # Poses de colocación cercanas a la pila destino
        self.place_standoff_positions = {
            "pb1": (11.20, 2.35, math.radians(-90.0)),   # llega por arriba
            "pb2": (10.75, 1.60, 0.0),                   # llega por la izquierda
            "pb3": (11.20, 0.95, math.radians(90.0)),    # llega por abajo
        }

        self.pb_wait_positions = {
            "pb1": (10.0, 4.0, 0.0),
            "pb2": (10.0, 3.25, 0.0),
            "pb3": (10.0, 2.5, 0.0),
        }

        # Logs por grasp
        self.grasp_logs: Dict[str, List[GraspResult]] = {
            "pb1": [],
            "pb2": [],
            "pb3": [],
        }

        # Altura virtual de apilado por nivel
        self.stack_heights = {
            0: 0.135,   # C abajo
            1: 0.165,   # B en medio
            2: 0.195,   # A arriba
        }
        self.train_collision_model()
        self.avoid_dir = {name: 1 for name in self.pb_names}
        self.pb_status = {name: "IDLE" for name in self.pb_names}

    def train_collision_model(self):
        # Features: [distancia, diferencia_vel, diferencia_angulo]
        X = np.array([
            [0.1, 0.5, 0.2],
            [0.2, 0.4, 0.3],
            [0.5, 0.2, 0.1],
            [1.0, 0.1, 0.05],
            [0.15, 0.6, 0.4],
            [0.8, 0.1, 0.1]
        ])

        # 1 = alto riesgo, 0 = bajo
        y = np.array([1, 1, 0, 0, 1, 0])

        model = LogisticRegression()
        model.fit(X, y)

        self.collision_model = model

    def collision_risk(self, robot1, robot2):

        x1, y1, th1 = self.pb_mobile[robot1].get_pose()
        x2, y2, th2 = self.pb_mobile[robot2].get_pose()

        # distancia
        dist = math.hypot(x1 - x2, y1 - y2)

        # velocidades
        log1 = self.pb_mobile[robot1].motion_log
        log2 = self.pb_mobile[robot2].motion_log

        v1 = log1['v'][-1] if log1['v'] else 0.0
        v2 = log2['v'][-1] if log2['v'] else 0.0

        dv = abs(v1 - v2)

        # orientación
        dtheta = abs(wrap_angle(th1 - th2))

        X = np.array([[dist, dv, dtheta]])
        risk = self.collision_model.predict(X)[0]

        return risk  # 0 o 1

    def _send_other_puzzlebots_to_wait(self, active_robot: str) -> None:
        """
        Mueve a los PuzzleBots que no están trabajando a sus zonas de espera,
        para que no invadan la trayectoria del robot activo.
        """
        for robot_name in self.pb_names:

            if robot_name == active_robot:
                continue

            # 🔥 NO mover robots que ya terminaron
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
        Navega un PuzzleBot a través de una secuencia de waypoints.
        Cada waypoint es (x, y, theta) y theta puede ser None.
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

    # -------------------------------------------------------------------------
    # Logging coordinador
    # -------------------------------------------------------------------------

    def _record(self, state: str, note: str = "") -> None:
        self.log.append(self.sim.time, state, note)
        self.sim.record_state(phase=state.lower(), note=note)

    # -------------------------------------------------------------------------
    # Fase 1: Husky
    # -------------------------------------------------------------------------

    def run_husky_phase(self, verbose: bool = True) -> None:
        """Ejecuta la fase 1 completa."""
        t0 = self.sim.time
        self._record("HUSKY_PHASE", "Inicio fase Husky")

        self.husky_log = self.husky_controller.run(verbose=verbose)
        husky_log = self.husky_log

        self.metrics.husky_time = self.sim.time - t0
        self._record(
            "HUSKY_PHASE",
            f"Fin fase Husky | boxes_cleared={self.sim.all_large_boxes_cleared()}"
        )

        # Guardar figura resumen
        plot_husky_phase_results(
            sim=self.sim,
            log=husky_log,
            title="Fase 1 - Husky: despeje del corredor",
            save_path="coordinator_husky_phase.png"
        )

    # -------------------------------------------------------------------------
    # Fase 2: ANYmal
    # -------------------------------------------------------------------------

    def run_anymal_phase(self, verbose: bool = True) -> None:
        """Ejecuta la fase 2 completa."""
        t0 = self.sim.time
        self._record("ANYMAL_PHASE", "Inicio fase ANYmal")

        self.anymal_log = self.anymal_controller.run(verbose=verbose)
        anymal_log = self.anymal_log

        self.metrics.anymal_time = self.sim.time - t0
        self.metrics.anymal_final_error = self.sim.anymal_goal_error()
        self.metrics.detJ_violations_anymal = len(anymal_log.singularity_events)

        reached = getattr(self.anymal_controller, "reached_goal", False)

        self._record(
            "ANYMAL_PHASE",
            (
                f"Fin fase ANYmal | reached={reached} | "
                f"err={self.metrics.anymal_final_error:.3f} m | "
                f"detJ_viol={self.metrics.detJ_violations_anymal}"
            )
        )

        plot_anymal_phase_results(
            log=anymal_log,
            title="ANYmal Fase 2: Marcha Trote",
            save_path="coordinator_anymal_phase.png"
        )

        if not reached:
            raise RuntimeError(
                f"Fase ANYmal falló: no llegó a p_dest_anymal. "
                f"Error final = {self.metrics.anymal_final_error:.3f} m"
            )


    # -------------------------------------------------------------------------
    # Fase 3: PuzzleBots
    # -------------------------------------------------------------------------


    def _get_box_world_center(self, box_name: str) -> Tuple[float, float]:
        """Centro XY de una caja pequeña en el mundo."""
        return self.sim.boxes[box_name].center()

    def _set_box_world_center(self, box_name: str, cx: float, cy: float) -> None:
        """Reposiciona caja pequeña usando su centro."""
        self.sim.boxes[box_name].set_center(cx, cy)

    def _stack_point_for_level(self, level: int) -> Tuple[float, float]:
        """
        Punto XY de apilado.
        En esta simulación 2D el XY es el mismo; el nivel se maneja virtualmente.
        """
        return self.sim.stack_point
    
    def _navigate_robot_to(self, robot_name: str, gx: float, gy: float, gtheta: Optional[float] = None,
                            phase_note: str = "", max_steps: int = 500) -> None:

        controller = self.pb_mobile[robot_name]
        if self.pb_status.get(robot_name) == "DONE":
            return

        for _ in range(max_steps):

            x, y, theta = controller.get_pose()

            # ======================================================
            # 1. DIRECCIÓN AL OBJETIVO
            # ======================================================
            dx = gx - x
            dy = gy - y

            dist_goal = math.hypot(dx, dy)

            dir_goal_x, dir_goal_y = 0.0, 0.0
            if dist_goal > 1e-6:
                dir_goal_x = dx / dist_goal
                dir_goal_y = dy / dist_goal

            # ======================================================
            # 2. FUERZA DE EVASIÓN (ROBOTS + CAJAS)
            # ======================================================
            avoid_x = 0.0
            avoid_y = 0.0

            # -------- ROBOTS --------
            for other in self.pb_names:
                if other == robot_name:
                    continue

                x2, y2, _ = self.pb_mobile[other].get_pose()

                dist = math.hypot(x - x2, y - y2)

                d_safe = 0.6
                d_crit = 0.25

                if dist < d_safe:
                    # riesgo continuo
                    risk = (d_safe - dist) / (d_safe - d_crit)
                    risk = np.clip(risk, 0.0, 1.0)

                    # dirección de repulsión
                    rx = x - x2
                    ry = y - y2

                    norm = math.hypot(rx, ry)
                    if norm > 1e-6:
                        rx /= norm
                        ry /= norm

                        avoid_x += risk * rx
                        avoid_y += risk * ry

                    print(f"[ML] {robot_name} evita robot {other} | risk={risk:.2f}")

            # -------- CAJAS --------
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

                    print(f"[ML] {robot_name} evita caja {box_name} | risk={risk:.2f}")

            # normalizar evasión
            norm_avoid = math.hypot(avoid_x, avoid_y)
            if norm_avoid > 1e-6:
                avoid_x /= norm_avoid
                avoid_y /= norm_avoid

            # ======================================================
            # 3. MEZCLA (CLAVE)
            # ======================================================
            # si no hay riesgo → puro goal
            if norm_avoid < 1e-6:
                mix_x = dir_goal_x
                mix_y = dir_goal_y
            else:
                # peso dinámico
                alpha = 0.7   # hacia goal

                # si está muy cerca del objetivo → menos evasión
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

            # ======================================================
            # 4. PASO CORTO (evita curvas grandes)
            # ======================================================
            step_size = 0.25

            gx_new = x + mix_x * step_size
            gy_new = y + mix_y * step_size

            # ======================================================
            # 5. MOVIMIENTO
            # ======================================================
            done = controller.step_to_pose(gx_new, gy_new, None)

            self.sim.step(
                phase="puzzlebot",
                note=f"{robot_name} navegando suave | {phase_note}"
            )

            if done:
                break
            
    def _move_box_with_robot(self, robot_name: str, box_name: str,
                            target_x: float, target_y: float,
                            level: int) -> None:
        controller = self.pb_mobile[robot_name]

        # Punto final de aproximación de ese robot a la pila
        gx, gy, gth = self.place_standoff_positions[robot_name]

        # Waypoint intermedio común para entrar ordenadamente al área de la pila
        # y luego aproximarse desde su lado correspondiente
        # posición actual del robot
        x, y, _ = controller.get_pose()

        # punto de stack final
        gx, gy, gth = self.place_standoff_positions[robot_name]

        # vector hacia el stack
        dx = gx - x
        dy = gy - y
        dist = math.hypot(dx, dy)

        if dist > 1e-6:
            dx /= dist
            dy /= dist

        # 🔥 offset lateral (lado de entrada)
        side_offset = 0.35

        # perpendicular
        px = -dy
        py = dx

        # elegir lado según robot
        if robot_name == "pb1":
            side = 1   # arriba
        elif robot_name == "pb2":
            side = -1  # izquierda
        else:
            side = 1   # abajo

        # waypoint dinámico
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

                # La caja acompaña al robot durante el transporte
                attach_x = x + 0.12 * math.cos(theta)
                attach_y = y + 0.12 * math.sin(theta)
                self._set_box_world_center(box_name, attach_x, attach_y)

                self.sim.step(
                    phase="puzzlebot",
                    note=f"{robot_name} transportando {box_name}"
                )
                if done:
                    break

        # Dejar primero la caja cerca de la pila desde su lado de aproximación
        x, y, theta = controller.get_pose()
        preplace_x = target_x - 0.05 * math.cos(theta)
        preplace_y = target_y - 0.05 * math.sin(theta)
        self._set_box_world_center(box_name, preplace_x, preplace_y)

        self.sim.step(
            phase="puzzlebot",
            note=f"{robot_name} acercó {box_name} a la pila desde su lado"
        )

        # Colocación final exacta
        self._set_box_world_center(box_name, target_x, target_y)
        self.sim.boxes[box_name].stacked = True
        self.sim.step(
            phase="puzzlebot",
            note=f"{robot_name} colocó {box_name} en nivel {level}"
        )


    def _execute_single_stack_task(self, robot_name: str, box_name: str, level: int) -> None:
        """
        Ejecuta una tarea individual de apilado para un PuzzleBot ya desplegado
        en la zona de trabajo.
        """
        self.pb_status[robot_name] = "ACTIVE"
        # Mandar a los otros robots a esperar antes de que este empiece
        self._send_other_puzzlebots_to_wait(active_robot=robot_name)

        box = self.sim.boxes[box_name]
        bx, by = box.center()
        

        # 1) Navegar a posición de toma
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

        # 2) Grasp con el brazo
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
            note=f"{robot_name} hizo grasp de {box_name}"
        )

        # 3) Transportar la caja hasta la pila
        sx, sy = self._stack_point_for_level(level)
        self._move_box_with_robot(
            robot_name=robot_name,
            box_name=box_name,
            target_x=sx,
            target_y=sy,
            level=level
        )

        # 4) Colocar en la pila con control de fuerza
        local_place = np.array([0.09, 0.00, self.stack_heights[level]], dtype=float)
        grasp_result_place = arm.grasp_box(
            box_pos=local_place,
            grip_force=4.0,
            n_points=30
        )
        self.grasp_logs[robot_name].append(grasp_result_place)

        self.sim.step(
            phase="puzzlebot",
            note=f"{robot_name} aplicó tau=J^T f para colocar {box_name}"
        )
        self.pb_status[robot_name] = "DONE"

    def run_puzzlebot_phase(self, verbose: bool = True) -> None:
        """
        Ejecuta la fase 3 completa en turnos.
        Esto cumple la recomendación del PDF de usar time-slotting
        como la forma más simple de coordinación.
        """
        t0 = self.sim.time
        self._record("PUZZLEBOT_PHASE", "Inicio fase PuzzleBots")

        # Time-slotting: un robot por vez
        for robot_name, box_name, level in self.stack_plan:
            if verbose:
                print(f"[puzzlebot] {robot_name} -> {box_name} -> nivel {level}")

            self._execute_single_stack_task(robot_name, box_name, level)

        self.metrics.puzzlebot_time = self.sim.time - t0

        # Error final de apilado en XY
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
                f"Fin fase PuzzleBots | stack_success={self.metrics.stack_success} | "
                f"stack_err={self.metrics.stack_final_error:.3f} m"
            )
        )

    # -------------------------------------------------------------------------
    # Misión completa
    # -------------------------------------------------------------------------

    def run_mission(self, verbose: bool = True) -> CoordinatorMetrics:
        """
        Ejecuta toda la misión de forma secuencial.
        """
        self._record("START", "Inicio misión completa")

        self.run_husky_phase(verbose=verbose)
        self.run_anymal_phase(verbose=verbose)
        self.run_puzzlebot_phase(verbose=verbose)

        self.state = "DONE"
        self.metrics.total_time = self.sim.time

        self._record("DONE", "Misión completa finalizada")
        return self.metrics

    # -------------------------------------------------------------------------
    # Reportes y figuras
    # -------------------------------------------------------------------------

    def plot_coordinator_summary(self, save_path: Optional[str] = None):
        """
        Figura resumen del coordinador y métricas globales.
        """
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        fig.suptitle("Resumen global del mini reto", fontsize=15, fontweight="bold")

        # 1) timeline de estados
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
        ax.set_title("Timeline de estados")
        ax.set_xlabel("Tiempo [s]")
        ax.set_ylabel("Estado")
        ax.set_yticks(list(state_to_num.values()))
        ax.set_yticklabels(list(state_to_num.keys()))
        ax.grid(True, alpha=0.3)

        # 2) tiempos por fase
        ax = axes[0, 1]
        labels = ["Husky", "ANYmal", "PuzzleBots"]
        vals = [
            self.metrics.husky_time,
            self.metrics.anymal_time,
            self.metrics.puzzlebot_time,
        ]
        ax.bar(labels, vals)
        ax.set_title("Tiempo por fase")
        ax.set_ylabel("Tiempo [s]")
        ax.grid(True, axis="y", alpha=0.3)

        # 3) métricas numéricas
        ax = axes[1, 0]
        ax.axis("off")
        text = (
            f"Tiempo total: {self.metrics.total_time:.2f} s\n"
            f"Error final ANYmal: {self.metrics.anymal_final_error:.3f} m\n"
            f"Violaciones det(J) ANYmal: {self.metrics.detJ_violations_anymal}\n"
            f"Error final apilado: {self.metrics.stack_final_error:.3f} m\n"
            f"Colisiones PuzzleBots: {self.metrics.puzzlebot_collisions}\n"
            f"Apilado correcto C-B-A: {self.metrics.stack_success}"
        )
        ax.text(
            0.05, 0.95, text,
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=11,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
        )
        ax.set_title("Métricas globales")

        # 4) torques finales de grasp/place por robot
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
        ax.set_title(r"Norma de torques finales ($\tau = J^T f$)")
        ax.set_ylabel("||tau|| [N·m]")
        ax.grid(True, axis="y", alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  -> Figura guardada en {save_path}")
        return fig

    def show_final_world(self, save_path: Optional[str] = None):
        """Muestra snapshot final del mundo."""
        fig, ax = self.sim.draw_world(
            phase="done",
            note="Estado final de la misión",
            show_lidar=False
        )
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  -> Figura guardada en {save_path}")
        return fig

    def save_animation(self, save_path: Optional[str] = None):
        """
        Genera la animación completa desde el log global del mundo.
        """
        return self.sim.animate_log(interval_ms=50, save_path=save_path)


# =============================================================================
# Demo principal
# =============================================================================

def demo_coordinator():
    """
    Demo de la misión completa.
    """
    coordinator = MissionCoordinator(dt=0.04)
    metrics = coordinator.run_mission(verbose=True)

    print("\n" + "=" * 70)
    print("RESUMEN FINAL DEL RETO")
    print("=" * 70)
    print(f"Tiempo Husky:       {metrics.husky_time:.2f} s")
    print(f"Tiempo ANYmal:      {metrics.anymal_time:.2f} s")
    print(f"Tiempo PuzzleBots:  {metrics.puzzlebot_time:.2f} s")
    print(f"Tiempo total:       {metrics.total_time:.2f} s")
    print(f"Error final ANYmal: {metrics.anymal_final_error:.3f} m")
    print(f"Violaciones det(J): {metrics.detJ_violations_anymal}")
    print(f"Error apilado:      {metrics.stack_final_error:.3f} m")
    print(f"Apilado correcto:   {metrics.stack_success}")

    coordinator.plot_coordinator_summary(save_path="coordinator_summary.png")
    coordinator.show_final_world(save_path="coordinator_final_world.png")

    # --- Gráficas estilo base/demo ---
    if coordinator.husky_log is not None:
        plot_husky_demo_style(
            coordinator.husky_log,
            title="Husky Fase 1: Despeje del corredor",
            save_path="coordinator_husky_demo_style.png"
        )

    if coordinator.anymal_log is not None:
        plot_anymal_phase_results(
            log=coordinator.anymal_log,
            title="ANYmal Fase 2: Marcha Trote",
            save_path="coordinator_anymal_demo_style.png"
        )

    for pb_name in coordinator.pb_names:
        pb_log = coordinator.pb_mobile[pb_name].motion_log
        if len(pb_log['t']) > 0:
            plot_puzzlebot_demo_style(
                pb_log,
                title=f"PuzzleBot {pb_name.upper()} Fase 3",
                save_path=f"coordinator_{pb_name}_demo_style.png"
            )

    # Animación opcional
    anim = coordinator.save_animation(save_path=None)

    plt.show()
    return coordinator, metrics, anim


if __name__ == "__main__":
    demo_coordinator()