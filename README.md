# Mini Challenge – Collaborative Warehouse

2D simulation of a multi-robot system where:

* A **Husky** clears a corridor by pushing boxes
* An **ANYmal** crosses the corridor
* Three **PuzzleBots** stack boxes in order **C → B → A**

> **⚠️ Branch status:** `feature/ml-collision-camera` is a work in progress, not a finished/stable version. It adds an ML-based collision-avoidance model for the PuzzleBots and a synthetic RGB camera renderer, both still under development (the collision model is trained on a small placeholder dataset, and the camera demo is experimental).

---

# Team

Course: **TE3002B – Ground Mobile Robots**

* Josue Ureña Valencia — IRS | A01738940
* César Arellano Arellano — IRS | A00839373
* Jose Eduardo Sanchez Martinez — IRS | A01738476
* Rafael André Gamiz Salazar — IRS | A00838280

---

# Project structure

```
.
├── sim.py                # World simulator (scenario, robots, boxes)
├── coordinator.py       # Orchestrator for the whole mission
├── husky_pusher.py      # Husky control
├── anymal_gait.py       # ANYmal control
├── puzzlebot_arm.py     # PuzzleBot manipulator arm
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
* Synthetic RGB camera rendering
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
   * Collision avoidance between PuzzleBots uses a logistic regression model

---

# Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

or manually:

```bash
pip install numpy matplotlib scikit-learn opencv-python
```

---

# How to run

Run directly:

```bash
python coordinator.py
```

This will:

* Run the whole mission automatically
* Show plots
* Show the final animation

You can also run the standalone simulator demo (live matplotlib view + synthetic camera window):

```bash
python sim.py
```

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
