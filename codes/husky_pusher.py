"""
husky_pusher.py
---------------
Nodo/controlador para la Fase 1 del mini reto:
el Husky A200 debe localizar y empujar 3 cajas grandes
fuera del corredor usando skid-steer + planner local simple.

Este módulo:
- reutiliza el modelo cinemático del Husky
- se integra con WarehouseSim (sim.py)
- usa un LiDAR 2D simulado del escenario
- implementa una lógica simple de:
    1) seleccionar caja objetivo
    2) navegar a un punto de pre-empuje
    3) alinearse con la dirección de empuje
    4) empujar la caja hasta sacarla del corredor
- guarda logs de:
    * v, omega comandados
    * v, omega medidos
    * velocidades de ruedas
    * caja objetivo
    * estado del planner

Autor: Tu equipo
Curso: TE3002B - Robots Móviles Terrestres
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math
import numpy as np
import matplotlib.pyplot as plt

from sim import WarehouseSim, wrap_angle, distance, clamp


# =============================================================================
# Modelo del Husky A200
# =============================================================================

class HuskyA200:
    """
    Modelo cinemático simplificado del Husky A200 (skid-steer de 4 ruedas).

    Modelo:
        avg_R = (wR1 + wR2)/2
        avg_L = (wL1 + wL2)/2

        v     = r/2 * (avg_R + avg_L) * slip
        omega = r/B * (avg_R - avg_L)

    Nota:
    - el factor de slip afecta v
    - mantenemos omega según el modelo base compartido en clase
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
        """Establece el terreno actual."""
        self.terrain = terrain_name

    def get_slip(self) -> float:
        """Retorna el factor de slip del terreno actual."""
        return self.slip_factors.get(self.terrain, 0.8)

    def forward_kinematics(
        self,
        wR1: float,
        wR2: float,
        wL1: float,
        wL2: float
    ) -> Tuple[float, float]:
        """Cinemática directa de 4 ruedas a (v, omega)."""
        avg_R = 0.5 * (wR1 + wR2)
        avg_L = 0.5 * (wL1 + wL2)
        slip = self.get_slip()

        v = self.r * 0.5 * (avg_R + avg_L) * slip
        omega = self.r / self.B * (avg_R - avg_L)
        return v, omega

    def inverse_kinematics(self, v: float, omega: float) -> Tuple[float, float, float, float]:
        """
        Cinemática inversa de (v, omega) a las 4 ruedas.
        Se asume mismo comando en las dos ruedas derechas y mismo en las dos izquierdas.
        """
        v = clamp(v, -self.v_max, self.v_max)
        omega = clamp(omega, -self.omega_max, self.omega_max)

        # Inversa compatible con el modelo base compartido
        wR = (2.0 * v + omega * self.B) / (2.0 * self.r)
        wL = (2.0 * v - omega * self.B) / (2.0 * self.r)

        wR = clamp(wR, -self.wheel_max, self.wheel_max)
        wL = clamp(wL, -self.wheel_max, self.wheel_max)

        return wR, wR, wL, wL


# =============================================================================
# Log de la fase Husky
# =============================================================================

@dataclass
class HuskyLog:
    """Log temporal detallado de la fase Husky."""
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
        """Agrega una muestra temporal al log."""
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
# Planeador local / controlador del Husky
# =============================================================================

class HuskyPusher:
    """
    Controlador de alto nivel para despejar el corredor.

    Estados:
        - SELECT_BOX
        - GO_PREPUSH
        - ALIGN_PUSH
        - PUSH
        - PARK
        - DONE

    Estrategia:
        1) elegir la siguiente caja dentro del corredor
        2) definir un punto de pre-empuje detrás de la caja
        3) llegar a ese punto con control proporcional simple
        4) alinear la orientación hacia la dirección de empuje
        5) empujar hasta que el centro de la caja salga del corredor
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

        # Ganancias y umbrales
        self.k_rho = 0.9
        self.k_alpha = 1.8
        self.k_push_heading = 2.2

        self.pos_tol = 0.14
        self.ang_tol = math.radians(7.0)

        self.push_speed_cmd = 0.55
        self.max_steps = 4000
        # Pose final de estacionamiento del Husky para liberar el corredor
        sx, sy, sw, sh = self.sim.start_zone
        self.park_pose = (
            sx + 0.28 * sw,   # x
            sy + 0.28 * sh,   # y
            math.radians(180.0)  # orientación hacia la izquierda
        )

        # Dirección preferida de expulsión por caja
        # B1 y B3 hacia abajo, B2 hacia arriba
        self.push_dirs = {
            "B1": (0.0, -1.0),
            "B2": (0.0, +1.0),
            "B3": (0.0, -1.0),
        }

    # -------------------------------------------------------------------------
    # Utilidades geométricas del planner
    # -------------------------------------------------------------------------

    def _boxes_still_blocking(self) -> List[str]:
        """Retorna las cajas grandes que siguen dentro del corredor."""
        return [name for name in ("B1", "B2", "B3") if not self.sim.is_box_out_of_corridor(name)]

    def _select_next_box(self) -> Optional[str]:
        """
        Selecciona la siguiente caja objetivo.
        Política simple: la más cercana al Husky entre las que siguen bloqueando.
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
        """Dirección unitaria deseada para expulsar la caja."""
        dx, dy = self.push_dirs[box_name]
        norm = math.hypot(dx, dy)
        return dx / norm, dy / norm

    def _get_prepush_point(self, box_name: str, margin: float = 0.70) -> Tuple[float, float]:
        """
        Punto detrás de la caja respecto a la dirección de empuje.
        Si la caja va hacia arriba, el Husky se coloca abajo de ella; viceversa.
        """
        box = self.sim.boxes[box_name]
        cx, cy = box.center()
        dir_x, dir_y = self._get_push_direction(box_name)

        # Detrás de la caja = centro - dir * margen
        px = cx - dir_x * margin
        py = cy - dir_y * margin
        return px, py
    
    def _get_recontact_point(self, box_name: str, margin: float = 0.42) -> Tuple[float, float]:
        """
        Punto cercano detrás de la caja para recuperar contacto si durante PUSH
        el Husky se separa.
        """
        box = self.sim.boxes[box_name]
        cx, cy = box.center()
        dir_x, dir_y = self._get_push_direction(box_name)

        rx = cx - dir_x * margin
        ry = cy - dir_y * margin
        return rx, ry

    def _get_push_heading(self, box_name: str) -> float:
        """Orientación deseada del Husky durante el empuje."""
        dir_x, dir_y = self._get_push_direction(box_name)
        return math.atan2(dir_y, dir_x)

    def _front_lidar_min(self) -> float:
        """
        Distancia mínima aproximada al frente del Husky usando una ventana angular pequeña.
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
        Verifica si el Husky está suficientemente cerca de una caja como para
        considerarse en contacto de empuje.
        """
        robot = self.sim.robots["husky"]
        box = self.sim.boxes[box_name]

        cx, cy = box.center()
        d = distance((robot.x, robot.y), (cx, cy))

        contact_threshold = robot.radius + 0.5 * math.hypot(box.w, box.h) + extra_margin
        return d <= contact_threshold

    # -------------------------------------------------------------------------
    # Control continuo simple
    # -------------------------------------------------------------------------

    def _go_to_point(self, goal_x: float, goal_y: float) -> Tuple[float, float]:
        """
        Control proporcional simple para navegación pose->punto.
        Retorna (v_cmd, omega_cmd).
        """
        husky = self.sim.robots["husky"]
        dx = goal_x - husky.x
        dy = goal_y - husky.y
        rho = math.hypot(dx, dy)

        desired_heading = math.atan2(dy, dx)
        alpha = wrap_angle(desired_heading - husky.theta)

        v_cmd = self.k_rho * rho
        omega_cmd = self.k_alpha * alpha

        # Reducir avance si todavía está muy mal orientado
        if abs(alpha) > math.radians(35.0):
            v_cmd *= 0.25

        v_cmd = clamp(v_cmd, -0.8, 0.8)
        omega_cmd = clamp(omega_cmd, -1.6, 1.6)

        return v_cmd, omega_cmd

    def _align_to_heading(self, theta_des: float) -> Tuple[float, float]:
        """
        Alineación pura de orientación antes del empuje.
        """
        husky = self.sim.robots["husky"]
        e = wrap_angle(theta_des - husky.theta)

        v_cmd = 0.0
        omega_cmd = clamp(self.k_push_heading * e, -1.2, 1.2)
        return v_cmd, omega_cmd
    
    def _go_to_park_pose(self) -> Tuple[float, float]:
        """
        Control para ir a la pose final de estacionamiento.
        """
        gx, gy, _ = self.park_pose
        return self._go_to_point(gx, gy)
    
    def _park_pose_reached(self) -> bool:
        """
        Verifica si el Husky ya llegó a la zona de estacionamiento.
        """
        husky = self.sim.robots["husky"]
        gx, gy, gtheta = self.park_pose

        pos_ok = distance((husky.x, husky.y), (gx, gy)) < self.pos_tol
        ang_ok = abs(wrap_angle(gtheta - husky.theta)) < self.ang_tol

        return pos_ok and ang_ok

    def _push_command(self, box_name: str) -> Tuple[float, float]:
        """
        Comando durante la fase de empuje.
        Mantiene orientación de empuje y además corrige ligeramente hacia
        el centro actual de la caja para no perder contacto.
        """
        husky = self.sim.robots["husky"]
        box = self.sim.boxes[box_name]
        cx, cy = box.center()

        # orientación ideal de empuje
        theta_push = self._get_push_heading(box_name)

        # orientación hacia el centro actual de la caja
        theta_box = math.atan2(cy - husky.y, cx - husky.x)

        # mezcla de ambas referencias
        theta_des = wrap_angle(0.75 * theta_push + 0.25 * theta_box)

        e = wrap_angle(theta_des - husky.theta)

        v_cmd = self.push_speed_cmd
        omega_cmd = clamp(1.6 * e, -0.9, 0.9)

        if abs(e) > math.radians(20.0):
            v_cmd *= 0.40

        return v_cmd, omega_cmd
    
    def _recover_contact_command(self, box_name: str) -> Tuple[float, float]:
        """
        Comando para volver a acercarse a la caja cuando se perdió contacto
        durante la fase de empuje.
        """
        gx, gy = self._get_recontact_point(box_name)
        v_cmd, omega_cmd = self._go_to_point(gx, gy)

        # más conservador que una navegación normal
        v_cmd = clamp(v_cmd, 0.0, 0.35)
        omega_cmd = clamp(omega_cmd, -1.0, 1.0)
        return v_cmd, omega_cmd

    # -------------------------------------------------------------------------
    # Integración del Husky en el escenario
    # -------------------------------------------------------------------------

    def _apply_motion(self, v_cmd: float, omega_cmd: float) -> Tuple[float, float, float, float, float, float]:
        """
        Convierte (v_cmd, omega_cmd) a ruedas, obtiene (v_meas, omega_meas)
        y actualiza la pose del Husky en el simulador.
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
        """Guarda una muestra en el log local del Husky."""
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
    # Máquina de estados
    # -------------------------------------------------------------------------

    def update(self) -> None:
        """
        Ejecuta un paso del planner/control del Husky.
        """
        v_cmd = 0.0
        omega_cmd = 0.0
        note = ""

        if self.state == "SELECT_BOX":
            self.current_box = self._select_next_box()
            if self.current_box is None:
                self.state = "PARK"
                note = "Corredor despejado; Husky va a estacionarse"
            else:
                self.state = "GO_PREPUSH"
                note = f"Nueva caja objetivo: {self.current_box}"

        elif self.state == "GO_PREPUSH":
            assert self.current_box is not None
            gx, gy = self._get_prepush_point(self.current_box)
            v_cmd, omega_cmd = self._go_to_point(gx, gy)

            husky = self.sim.robots["husky"]
            if distance((husky.x, husky.y), (gx, gy)) < self.pos_tol:
                self.state = "ALIGN_PUSH"
                note = f"Llegó a pre-empuje de {self.current_box}"
            else:
                note = f"Yendo a pre-empuje {self.current_box}"

        elif self.state == "ALIGN_PUSH":
            assert self.current_box is not None
            theta_des = self._get_push_heading(self.current_box)
            v_cmd, omega_cmd = self._align_to_heading(theta_des)

            husky = self.sim.robots["husky"]
            e = wrap_angle(theta_des - husky.theta)
            if abs(e) < self.ang_tol:
                self.state = "PUSH"
                note = f"Alineado para empujar {self.current_box}"
            else:
                note = f"Alineando con {self.current_box}"

        elif self.state == "PUSH":
            assert self.current_box is not None

            # Si ya salió completamente, pasar a la siguiente caja
            if self.sim.is_box_out_of_corridor(self.current_box):
                note = f"{self.current_box} completamente fuera del corredor"
                self.current_box = None
                self.state = "SELECT_BOX"

            else:
                # Verificar si aún hay contacto real con la caja
                in_contact = self._is_in_contact_with_box(self.current_box)

                if in_contact:
                    # Mantener empuje continuo
                    v_cmd, omega_cmd = self._push_command(self.current_box)

                    pushed = self.sim.push_box_if_contact(
                        robot_name="husky",
                        box_name=self.current_box,
                        push_distance=max(0.0, v_cmd * self.sim.dt * 1.10)
                    )

                    note = f"Empujando {self.current_box} con contacto"
                    if not pushed:
                        note += " (contacto marginal)"

                else:
                    # Recuperar contacto antes de seguir empujando
                    v_cmd, omega_cmd = self._recover_contact_command(self.current_box)
                    note = f"Recuperando contacto con {self.current_box}"

        elif self.state == "PARK":
            gx, gy, gtheta = self.park_pose
            husky = self.sim.robots["husky"]

            # Primero acercarse al punto
            if distance((husky.x, husky.y), (gx, gy)) > self.pos_tol:
                v_cmd, omega_cmd = self._go_to_park_pose()
                note = "Husky yendo a la esquina de estacionamiento"

            else:
                # Luego alinear orientación final
                v_cmd, omega_cmd = self._align_to_heading(gtheta)
                note = "Husky alineándose en estacionamiento"

                if self._park_pose_reached():
                    self.state = "DONE"
                    v_cmd = 0.0
                    omega_cmd = 0.0
                    note = "Fase 1 completada: corredor libre y Husky estacionado"           

        elif self.state == "DONE":
            v_cmd = 0.0
            omega_cmd = 0.0
            note = "Fase Husky completada"

        else:
            raise RuntimeError(f"Estado desconocido: {self.state}")

        # Integrar movimiento real del Husky
        wR1, wR2, wL1, wL2, v_meas, omega_meas = self._apply_motion(v_cmd, omega_cmd)

        # Registrar en el log local
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

        # Registrar también en el log global del mundo
        self.sim.step(
            phase="husky",
            note=note
        )

    def run(self, max_steps: Optional[int] = None, verbose: bool = True) -> HuskyLog:
        """
        Corre la fase completa del Husky hasta despejar las 3 cajas
        o hasta alcanzar el límite de pasos.
        """
        if max_steps is None:
            max_steps = self.max_steps

        # Guardar snapshot inicial
        self.sim.record_state(phase="husky_init", note="Inicio fase Husky")

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
# Plots del Husky
# =============================================================================

def plot_husky_phase_results(
    sim: WarehouseSim,
    log: HuskyLog,
    title: str = "Husky - Fase 1: despeje del corredor",
    save_path: Optional[str] = None
):
    """
    Genera una figura resumen de la fase 1 del Husky.

    Subplots:
        1) trayectoria XY + cajas
        2) velocidades de ruedas
        3) v, omega comandados vs medidos
        4) distancia mínima frontal LiDAR
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(title, fontsize=15, fontweight="bold")

    # -----------------------------------------------------------------
    # 1) Trayectoria XY
    # -----------------------------------------------------------------
    ax = axes[0, 0]
    ax.set_title("Trayectoria del Husky y cajas")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlim(sim.world_xmin, sim.world_xmax)
    ax.set_ylim(sim.world_ymin, sim.world_ymax)

    # Dibujar zonas
    sx, sy, sw, sh = sim.start_zone
    ax.add_patch(plt.Rectangle((sx, sy), sw, sh, fill=False, linestyle="--", edgecolor="tab:green"))
    cx, cy, cw, ch = sim.corridor
    ax.add_patch(plt.Rectangle((cx, cy), cw, ch, fill=False, linestyle="-", linewidth=2, edgecolor="dimgray"))
    wx, wy, ww, wh = sim.work_zone
    ax.add_patch(plt.Rectangle((wx, wy), ww, wh, fill=False, linestyle="--", edgecolor="tab:blue"))

    # Trayectoria
    ax.plot(log.x, log.y, color="goldenrod", linewidth=2.5, label="Husky")
    ax.plot(log.x[0], log.y[0], "go", markersize=9, label="Inicio")
    ax.plot(log.x[-1], log.y[-1], "rs", markersize=9, label="Fin")

    # Flechas de orientación
    step = max(1, len(log.t) // 20)
    for i in range(0, len(log.t), step):
        dx = 0.25 * math.cos(log.theta[i])
        dy = 0.25 * math.sin(log.theta[i])
        ax.arrow(log.x[i], log.y[i], dx, dy,
                 head_width=0.08, head_length=0.08,
                 fc="orange", ec="orange", alpha=0.75)

    # Trayectorias de cajas grandes
    colors = {"B1": "#8B5A2B", "B2": "#A0522D", "B3": "#CD853F"}
    for name in ("B1", "B2", "B3"):
        pts = np.array(log.box_positions[name])
        ax.plot(pts[:, 0], pts[:, 1], linestyle="--", linewidth=2,
                color=colors[name], label=f"{name}")
        ax.plot(pts[0, 0], pts[0, 1], "o", color=colors[name], markersize=6)
        ax.plot(pts[-1, 0], pts[-1, 1], "s", color=colors[name], markersize=6)

    ax.legend(loc="best", fontsize=8)

    # -----------------------------------------------------------------
    # 2) Ruedas
    # -----------------------------------------------------------------
    ax = axes[0, 1]
    ax.set_title("Actuadores: 4 ruedas del Husky")
    ax.plot(log.t, log.wR1, "b-", linewidth=2, label=r"$\omega_{R1}$ (FR)")
    ax.plot(log.t, log.wR2, "b--", linewidth=2, label=r"$\omega_{R2}$ (RR)")
    ax.plot(log.t, log.wL1, "r-", linewidth=2, label=r"$\omega_{L1}$ (FL)")
    ax.plot(log.t, log.wL2, "r--", linewidth=2, label=r"$\omega_{L2}$ (RL)")
    ax.set_xlabel("Tiempo [s]")
    ax.set_ylabel("Velocidad angular [rad/s]")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="best")

    # -----------------------------------------------------------------
    # 3) v y omega: comandados vs medidos
    # -----------------------------------------------------------------
    ax = axes[1, 0]
    ax2 = ax.twinx()

    l1 = ax.plot(log.t, log.v_cmd, "g--", linewidth=2, label="v_cmd [m/s]")
    l2 = ax.plot(log.t, log.v_meas, "g-", linewidth=2, label="v_meas [m/s]")

    l3 = ax2.plot(log.t, log.omega_cmd, "m--", linewidth=2, label=r"$\omega$_cmd [rad/s]")
    l4 = ax2.plot(log.t, log.omega_meas, "m-", linewidth=2, label=r"$\omega$_meas [rad/s]")

    ax.set_title("Velocidades del cuerpo: comandadas vs medidas")
    ax.set_xlabel("Tiempo [s]")
    ax.set_ylabel("Velocidad lineal v [m/s]", color="g")
    ax2.set_ylabel(r"Velocidad angular $\omega$ [rad/s]", color="m")
    ax.tick_params(axis="y", labelcolor="g")
    ax2.tick_params(axis="y", labelcolor="m")
    ax.grid(True, alpha=0.3)

    lines = l1 + l2 + l3 + l4
    labels = [ln.get_label() for ln in lines]
    ax.legend(lines, labels, loc="best", fontsize=9)

    # -----------------------------------------------------------------
    # 4) LiDAR frontal
    # -----------------------------------------------------------------
    ax = axes[1, 1]
    ax.set_title("LiDAR 2D simulado: distancia frontal mínima")
    ax.plot(log.t, log.lidar_front_min, color="darkorange", linewidth=2)
    ax.set_xlabel("Tiempo [s]")
    ax.set_ylabel("Distancia [m]")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  -> Figura guardada en {save_path}")

    return fig

def husky_log_to_demo_dict(log):
    """
    Convierte el HuskyLog del reto a un diccionario tipo demo/base.
    Si ya viene como dict, lo deja igual.
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

def plot_husky_demo_style(log, title="Husky A200 - Trayectoria y Actuadores",
                          save_path=None):
    """
    Gráfica con el mismo estilo visual del código base del curso.
    """
    d = husky_log_to_demo_dict(log)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # --- Subplot 1: trayectoria XY ---
    ax = axes[0, 0]
    ax.plot(d['x'], d['y'], 'b-', linewidth=2, label='Trayectoria')
    ax.plot(d['x'][0], d['y'][0], 'go', markersize=10, label='Inicio')
    ax.plot(d['x'][-1], d['y'][-1], 'rs', markersize=10, label='Fin')
    step = max(1, len(d['t']) // 20)
    for i in range(0, len(d['t']), step):
        dx = 0.3 * np.cos(d['theta'][i])
        dy = 0.3 * np.sin(d['theta'][i])
        ax.arrow(d['x'][i], d['y'][i], dx, dy,
                 head_width=0.08, head_length=0.08, fc='orange', ec='orange')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('Trayectoria en el plano XY')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='datalim')

    # --- Subplot 2: actuadores ---
    ax = axes[0, 1]
    ax.plot(d['t'], d['wR1'], 'b-', linewidth=2, label=r'$\omega_{R1}$ (FR)')
    ax.plot(d['t'], d['wR2'], 'b--', linewidth=2, label=r'$\omega_{R2}$ (RR)')
    ax.plot(d['t'], d['wL1'], 'r-', linewidth=2, label=r'$\omega_{L1}$ (FL)')
    ax.plot(d['t'], d['wL2'], 'r--', linewidth=2, label=r'$\omega_{L2}$ (RL)')
    ax.set_xlabel('Tiempo [s]')
    ax.set_ylabel('Velocidad angular [rad/s]')
    ax.set_title('Actuadores: 4 ruedas del Husky')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Subplot 3: velocidades del cuerpo ---
    ax = axes[1, 0]
    ax2 = ax.twinx()
    l1 = ax.plot(d['t'], d['v'], 'g-', linewidth=2, label='v [m/s]')
    l2 = ax2.plot(d['t'], d['omega'], 'm-', linewidth=2, label=r'$\omega$ [rad/s]')
    ax.set_xlabel('Tiempo [s]')
    ax.set_ylabel('Velocidad lineal v [m/s]', color='g')
    ax2.set_ylabel(r'Velocidad angular $\omega$ [rad/s]', color='m')
    ax.tick_params(axis='y', labelcolor='g')
    ax2.tick_params(axis='y', labelcolor='m')
    ax.set_title('Velocidades del cuerpo')
    lines = l1 + l2
    ax.legend(lines, [l.get_label() for l in lines], loc='best')
    ax.grid(True, alpha=0.3)

    # --- Subplot 4: orientación ---
    ax = axes[1, 1]
    ax.plot(d['t'], np.degrees(d['theta']), 'k-', linewidth=2)
    ax.set_xlabel('Tiempo [s]')
    ax.set_ylabel(r'$\theta$ [deg]')
    ax.set_title('Orientacion del robot')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  -> Figura guardada en {save_path}")
    return fig


# =============================================================================
# Demo local
# =============================================================================

def demo_husky_pusher():
    """
    Demo de la fase Husky dentro del escenario del mini reto.
    """
    sim = WarehouseSim(dt=0.05)
    controller = HuskyPusher(sim=sim, terrain="grass")

    log = controller.run(verbose=True)

    print("\nResumen fase Husky:")
    print(f"  Cajas fuera del corredor: {sim.all_large_boxes_cleared()}")
    print(f"  Estado final del planner: {controller.state}")
    print(f"  Tiempo total: {sim.time:.2f} s")

    for name in ("B1", "B2", "B3"):
        print(f"  {name} fuera del corredor: {sim.is_box_out_of_corridor(name)}")

    plot_husky_phase_results(
        sim=sim,
        log=log,
        title="Husky - Fase 1: despeje del corredor",
        save_path="husky_phase1_pusher.png"
    )

    # Snapshot final
    sim.draw_world(
        phase="husky_done",
        note="Fase 1 completada",
        show_lidar=True,
        lidar_robot_name="husky"
    )
    plt.tight_layout()
    plt.show()

    return sim, controller, log


if __name__ == "__main__":
    demo_husky_pusher()