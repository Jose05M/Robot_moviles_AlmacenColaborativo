# Mini Challenge – Collaborative Warehouse

2D simulation of a multi-robot system where:

* A **Husky** clears a corridor by pushing boxes
* An **ANYmal** crosses the corridor
* Three **PuzzleBots** stack boxes in order **C → B → A**

---

# Project structure

```
.
├── src/
│   ├── sim.py                # World simulator (scenario, robots, boxes)
│   ├── coordinator.py        # Orchestrator for the whole mission
│   ├── husky_pusher.py       # Husky control
│   ├── anymal_gait.py        # ANYmal control
│   └── puzzlebot_arm.py      # PuzzleBot manipulator arm
├── plots/                    # Reference output plots
└── requirements.txt
```

---

# Overview

## sim.py

This file defines the complete environment:

* Warehouse map
* Robots (Husky, ANYmal, PuzzleBots)
* Boxes (large and small)
* Simplified physics (pushing, distances)
* LiDAR simulation
* Visualization and animation

---

## coordinator.py

This file coordinates the whole mission in 3 phases:

1. **HUSKY_PHASE**

   * Clears the corridor

2. **ANYMAL_PHASE**

   * Crosses the corridor

3. **PUZZLEBOT_PHASE**

   * Stacks boxes in order: C (bottom), B (middle), A (top)

---

# Requirements

Install dependencies:

```bash
pip install numpy matplotlib
```

---

# How to run

Run directly:

```bash
python src/coordinator.py
```

This will:

* Run the whole mission automatically
* Show plots
* Show the final animation

---

# Output

* Per-phase plots:

  * Husky
  * ANYmal
  * PuzzleBots
* Global system summary
* Full scenario animation

---

# Software modules

## husky_pusher.py

Controls:

* Husky navigation
* Pushing large boxes

---

## anymal_gait.py

Controls:

* ANYmal locomotion (trot gait)
* Kinematics / trajectory

---

## puzzlebot_arm.py

Controls:

* 3-DOF manipulator arm
* Grasping and placing boxes

---
