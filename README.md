# Mini Reto – Almacén Colaborativo

Simulación 2D de un sistema multi-robot donde:

* Un **Husky** despeja un corredor empujando cajas
* Un **ANYmal** cruza el corredor
* Tres **PuzzleBots** apilan cajas en orden **C → B → A**

---

# Estructura del proyecto

```
.
├── sim.py                # Simulador del mundo (escenario, robots, cajas)
├── coordinator.py       # Orquestador de toda la misión
├── husky_pusher.py      # Control del Husky
├── anymal_gait.py       # Control de ANYmal
├── puzzlebot_arm.py     # Brazo manipulador PuzzleBot
```

---

# Descripción general

## sim.py

Este archivo define el entorno completo:

* Mapa del almacén
* Robots (Husky, ANYmal, PuzzleBots)
* Cajas (grandes y pequeñas)
* Física simplificada (empuje, distancias)
* Simulación de LiDAR
* Visualización y animación

---

## coordinator.py

Este archivo coordina toda la misión en 3 fases:

1. **HUSKY_PHASE**

   * Despeja el corredor

2. **ANYMAL_PHASE**

   * Cruza el corredor

3. **PUZZLEBOT_PHASE**

   * Apila cajas en orden: C (abajo), B (medio), A (arriba)

---

# Requisitos

Instala dependencias:

```bash
pip install numpy matplotlib
```

---

# Cómo ejecutar

Ejecuta directamente:

```bash
python coordinator.py
```

Esto hará:

* Correr toda la misión automáticamente
* Mostrar gráficas
* Mostrar animación final

---

# Qué se genera

* Gráficas por fase:

  * Husky
  * ANYmal
  * PuzzleBots
* Resumen global del sistema
* Animación del escenario completo

---

# Modulos de software

## husky_pusher.py

Controla:

* Navegación del Husky
* Empuje de cajas grandes

---

## anymal_gait.py

Controla:

* Movimiento del ANYmal (marcha tipo trote)
* Cinemática / trayectoria

---

## puzzlebot_arm.py

Controla:

* Brazo manipulador de 3 DOF
* Grasp y colocación de cajas

---