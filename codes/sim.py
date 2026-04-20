"""
sim.py
------
Simulador 2D del escenario completo del mini reto de Robots Móviles.

Este módulo define:
- El mapa del almacén
- El corredor bloqueado
- Las cajas grandes que empuja el Husky
- Las cajas pequeñas A, B, C para apilar
- Los estados de los robots en 2D
- Utilidades geométricas, colisiones básicas y logging
- Visualización estática y animación en matplotlib

Diseñado para integrarse con:
    1) husky_pusher.py
    2) anymal_gait.py
    3) puzzlebot_arm.py
    4) coordinator.py

Autores: 
Josue Ureña Valencia				IRS | A01738940
César Arellano Arellano			    IRS | A00839373
Jose Eduardo Sanchez Martinez		IRS | A01738476
Rafael André Gamiz Salazar			IRS | A00838280
Curso: TE3002B - Robots Móviles Terrestres
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch
from matplotlib.animation import FuncAnimation


# =============================================================================
# Utilidades geométricas
# =============================================================================

def wrap_angle(theta: float) -> float:
    """Normaliza un ángulo al intervalo [-pi, pi]."""
    return math.atan2(math.sin(theta), math.cos(theta))


def distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Distancia euclidiana 2D."""
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def clamp(value: float, low: float, high: float) -> float:
    """Satura un valor en [low, high]."""
    return max(low, min(high, value))


def point_in_rect(
    px: float,
    py: float,
    rx: float,
    ry: float,
    rw: float,
    rh: float
) -> bool:
    """Indica si un punto está dentro de un rectángulo axis-aligned."""
    return (rx <= px <= rx + rw) and (ry <= py <= ry + rh)


def rect_center(x: float, y: float, w: float, h: float) -> Tuple[float, float]:
    """Centro geométrico de un rectángulo."""
    return (x + w / 2.0, y + h / 2.0)


# =============================================================================
# Entidades del mundo
# =============================================================================

@dataclass
class Box2D:
    """
    Caja rectangular en 2D.

    Atributos:
        name: identificador
        x, y: esquina inferior izquierda [m]
        w, h: dimensiones [m]
        kind: "large" o "small"
        color: color para dibujar
        movable: si puede moverse
        stacked: si ya forma parte de la pila final
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
        """Retorna el centro de la caja."""
        return rect_center(self.x, self.y, self.w, self.h)

    def as_patch(self, alpha: float = 0.85) -> Rectangle:
        """Genera el patch de matplotlib para dibujar la caja."""
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
        """Desplaza la caja."""
        if self.movable:
            self.x += dx
            self.y += dy

    def set_center(self, cx: float, cy: float) -> None:
        """Reposiciona la caja usando su centro."""
        self.x = cx - self.w / 2.0
        self.y = cy - self.h / 2.0


@dataclass
class RobotState:
    """
    Estado simplificado de un robot móvil en 2D.

    Atributos:
        name: nombre del robot
        x, y, theta: pose en el plano
        radius: radio de colisión aproximado
        color: color para dibujarlo
        active: si debe mostrarse
        payload: texto descriptivo opcional
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
        """Retorna la pose actual."""
        return (self.x, self.y, self.theta)

    def set_pose(self, x: float, y: float, theta: float) -> None:
        """Actualiza la pose del robot."""
        self.x = x
        self.y = y
        self.theta = wrap_angle(theta)


@dataclass
class WorldLog:
    """
    Estructura de logging del escenario.

    Guarda snapshots temporales del mundo para graficación y animación.
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
        """Agrega un snapshot del estado actual."""
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
# Escenario principal del reto
# =============================================================================

class WarehouseSim:
    """
    Simulador 2D del almacén colaborativo.

    El mundo incluye:
    - zona de inicio
    - corredor (rectángulo de 6x2 m)
    - zona de trabajo
    - tres cajas grandes bloqueando el corredor
    - tres cajas pequeñas A, B, C
    - punto destino de apilado

    Coordenadas del mundo:
        X hacia la derecha
        Y hacia arriba
    """

    def __init__(self, dt: float = 0.05):
        self.dt = dt
        self.time = 0.0

        # Dimensiones generales del mundo
        self.world_xmin = -1.0
        self.world_xmax = 13.5
        self.world_ymin = -1.5
        self.world_ymax = 5.5

        # Rectángulos de referencia
        self.start_zone = (0.0, 1.2, 1.8, 2.0)         # x, y, w, h
        self.corridor = (2.0, 1.2, 6.0, 2.0)           # rectángulo de 6x2 m
        self.work_zone = (9.0, 0.6, 3.5, 3.6)

        # Punto destino ANYmal
        self.anymal_goal = (11.0, 3.6)

        # Punto destino de apilado C-B-A
        self.stack_point = (11.4, 1.6)

        # Robots del escenario
        self.robots: Dict[str, RobotState] = {}
        self._init_robots()

        # Cajas grandes y pequeñas
        self.boxes: Dict[str, Box2D] = {}
        self._init_boxes()

        # Historial temporal del mundo
        self.log = WorldLog()

        # Configuración visual
        self.title = "Mini Reto - Almacén colaborativo"

    # -------------------------------------------------------------------------
    # Inicialización
    # -------------------------------------------------------------------------

    def _init_robots(self) -> None:
        """Inicializa los robots del escenario."""
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
        """Inicializa las cajas grandes del corredor y las pequeñas del área de trabajo."""
        # Cajas grandes que bloquean el corredor
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

        # Cajas pequeñas del apilado
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
    # Utilidades de escenario
    # -------------------------------------------------------------------------

    def reset(self) -> None:
        """Reinicia por completo el escenario a su estado inicial."""
        self.time = 0.0
        self.log = WorldLog()
        self._init_robots()
        self._init_boxes()

    def step(self, n: int = 1, phase: str = "idle", note: str = "") -> None:
        """
        Avanza el tiempo del mundo y guarda snapshots.

        No integra dinámica por sí mismo; se usa como reloj central.
        """
        for _ in range(n):
            self.time += self.dt
            self.record_state(phase=phase, note=note)

    def record_state(self, phase: str = "idle", note: str = "") -> None:
        """Guarda el estado actual del mundo en el log."""
        self.log.append(
            t=self.time,
            robots=self.robots,
            boxes=self.boxes,
            phase=phase,
            note=note
        )

    def set_robot_pose(self, name: str, x: float, y: float, theta: float) -> None:
        """Actualiza la pose de un robot por nombre."""
        if name not in self.robots:
            raise KeyError(f"Robot '{name}' no existe.")
        self.robots[name].set_pose(x, y, theta)

    def move_robot_by(self, name: str, dx: float, dy: float, dtheta: float = 0.0) -> None:
        """Desplaza incrementalmente un robot."""
        if name not in self.robots:
            raise KeyError(f"Robot '{name}' no existe.")
        robot = self.robots[name]
        robot.set_pose(robot.x + dx, robot.y + dy, robot.theta + dtheta)

    def robot_to_box_distance(self, robot_name: str, box_name: str) -> float:
        """Distancia entre el centro del robot y el centro de una caja."""
        robot = self.robots[robot_name]
        box = self.boxes[box_name]
        return distance((robot.x, robot.y), box.center())

    def is_box_out_of_corridor(self, box_name: str) -> bool:
        """
        Verifica si una caja quedó completamente fuera del rectángulo del corredor.

        La caja solo se considera fuera si su rectángulo ya no intersecta
        en absoluto con el rectángulo del corredor.
        """
        if box_name not in self.boxes:
            raise KeyError(f"Caja '{box_name}' no existe.")

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
        """Indica si B1, B2 y B3 ya están fuera del corredor."""
        targets = ["B1", "B2", "B3"]
        return all(self.is_box_out_of_corridor(name) for name in targets)

    def anymal_goal_error(self) -> float:
        """Error euclidiano del ANYmal respecto al objetivo final."""
        anymal = self.robots["anymal"]
        return distance((anymal.x, anymal.y), self.anymal_goal)

    def stack_order_is_correct(self, tol_xy: float = 0.05, tol_z_virtual: float = 1e-9) -> bool:
        """
        Verificación simplificada del apilado C-B-A.

        En esta simulación 2D no modelamos altura física real en el mundo.
        En vez de eso, consideramos que:
        - las cajas deben quedar centradas en stack_point
        - y cada una debe marcarse como 'stacked'
        - coordinator.py llevará el conteo de niveles

        Esta función valida la proyección XY.
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
        Despliega los 3 PuzzleBots cerca del ANYmal cuando llega a la zona de trabajo.
        La idea es que se vean como si se bajaran del robot, no que aparezcan
        mágicamente en posiciones lejanas.
        """
        anymal = self.robots["anymal"]

        # Offsets locales alrededor del ANYmal para "bajarse"
        # pb1: frente-derecha
        # pb2: frente-izquierda
        # pb3: atrás
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
        Coloca los 3 PuzzleBots montados sobre el ANYmal usando offsets 2D
        respecto al centro del robot.

        offsets:
            diccionario con desplazamientos locales (dx, dy) en el marco del ANYmal.
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
    # LiDAR 2D simulado para el Husky
    # -------------------------------------------------------------------------

    def simulate_lidar_2d(
        self,
        robot_name: str = "husky",
        n_beams: int = 181,
        max_range: float = 8.0,
        fov_deg: float = 180.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simula un LiDAR 2D muy simplificado.

        Retorna:
            angles: ángulos relativos al robot [rad]
            ranges: distancia estimada [m]

        Modelo simplificado:
        - solo detecta intersección aproximada con cajas grandes
        - usa el centro de cada caja y un radio equivalente
        - suficiente para planner local simple del reto
        """
        if robot_name not in self.robots:
            raise KeyError(f"Robot '{robot_name}' no existe.")

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

                # Distancia perpendicular del centro de caja al rayo
                perp = abs(vx * dy_beam - vy * dx_beam)

                # Radio equivalente de caja como círculo de colisión
                eq_radius = 0.5 * math.hypot(box.w, box.h)

                if perp <= eq_radius:
                    hit_range = max(0.0, proj - eq_radius)
                    if hit_range < ranges[i]:
                        ranges[i] = hit_range

        return angles, ranges

    # -------------------------------------------------------------------------
    # Contacto simple robot-caja
    # -------------------------------------------------------------------------

    def push_box_if_contact(
        self,
        robot_name: str,
        box_name: str,
        push_distance: float
    ) -> bool:
        """
        Modelo simplificado de empuje.

        Si el robot está suficientemente cerca de una caja, la empuja en la dirección
        de su orientación actual.

        Retorna:
            True si hubo contacto y movimiento de caja
            False en caso contrario
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
    # Visualización
    # -------------------------------------------------------------------------

    def _draw_robot(self, ax, robot: RobotState) -> None:
        """Dibuja un robot como círculo con flecha de orientación."""
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
        """Dibuja zonas fijas del escenario."""
        # Zona de inicio
        x, y, w, h = self.start_zone
        ax.add_patch(Rectangle((x, y), w, h, fill=False, linestyle="--",
                               linewidth=2.0, edgecolor="tab:green"))
        ax.text(x + 0.05, y + h + 0.08, "ZONA DE INICIO", color="tab:green",
                fontsize=10, fontweight="bold")

        # Corredor
        x, y, w, h = self.corridor
        ax.add_patch(Rectangle((x, y), w, h, fill=False, linestyle="-",
                               linewidth=2.5, edgecolor="dimgray"))
        ax.text(x + 0.05, y + h + 0.08, "CORREDOR (6x2 m)", color="dimgray",
                fontsize=10, fontweight="bold")

        # Zona de trabajo
        x, y, w, h = self.work_zone
        ax.add_patch(Rectangle((x, y), w, h, fill=False, linestyle="--",
                               linewidth=2.0, edgecolor="tab:blue"))
        ax.text(x + 0.05, y + h + 0.08, "ZONA DE TRABAJO", color="tab:blue",
                fontsize=10, fontweight="bold")

        # Objetivo ANYmal
        gx, gy = self.anymal_goal
        ax.plot(gx, gy, marker="*", markersize=14, color="tab:red")
        ax.text(gx + 0.1, gy + 0.08, "p_dest ANYmal", color="tab:red",
                fontsize=9, fontweight="bold")

        # Punto de apilado
        sx, sy = self.stack_point
        ax.plot(sx, sy, marker="s", markersize=10, color="black")
        ax.text(sx + 0.1, sy + 0.08, "pila destino", color="black",
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
        Dibuja el estado actual del mundo.

        Parámetros:
            ax: eje existente o None
            phase: texto de fase actual
            note: nota descriptiva
            show_lidar: dibuja haces del LiDAR
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
        ax.set_title(f"{self.title} | Fase: {phase}")

        self._draw_static_map(ax)

        # Dibujar cajas
        for box in self.boxes.values():
            patch = box.as_patch()
            ax.add_patch(patch)
            cx, cy = box.center()
            ax.text(cx, cy, box.name, ha="center", va="center",
                    fontsize=9, fontweight="bold", color="white")

        # Dibujar robots
        for robot in self.robots.values():
            self._draw_robot(ax, robot)

        # LiDAR opcional
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

        # Texto informativo
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
    # Animación desde el log
    # -------------------------------------------------------------------------

    def animate_log(
        self,
        interval_ms: int = 60,
        save_path: Optional[str] = None
    ) -> FuncAnimation:
        """
        Genera una animación matplotlib a partir del log del mundo.

        Requisitos:
            primero haber llenado self.log con snapshots.
        """
        if len(self.log.t) == 0:
            raise RuntimeError("No hay datos en el log. Llama record_state() durante la simulación.")

        fig, ax = plt.subplots(figsize=(12, 6))

        def update(frame_idx: int):
            ax.clear()

            # Restaurar snapshot
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
            # Requiere ffmpeg o pillow según el formato
            anim.save(save_path, dpi=120)

        return anim


# =============================================================================
# Demo local de prueba del simulador
# =============================================================================

def demo_sim() -> WarehouseSim:
    """
    Demo mínima para verificar que el mapa y el log funcionan.
    No resuelve el reto; solo prueba el escenario.
    """
    sim = WarehouseSim(dt=0.1)

    # Guardar estado inicial
    sim.record_state(phase="init", note="Estado inicial del escenario")

    # Mover ligeramente al Husky
    for _ in range(10):
        sim.move_robot_by("husky", dx=0.08, dy=0.0, dtheta=0.01)
        sim.step(phase="demo_husky", note="Prueba básica de movimiento")

    # Activar PuzzleBots en la zona de trabajo
    sim.activate_puzzlebots_at_work_zone()
    sim.record_state(phase="demo_deploy", note="Despliegue de PuzzleBots")

    # Mostrar snapshot final
    sim.draw_world(phase="demo_final", note="Snapshot final del demo", show_lidar=True)
    plt.tight_layout()
    plt.show()

    return sim


if __name__ == "__main__":
    demo_sim()