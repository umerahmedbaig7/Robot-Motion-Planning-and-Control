<div align="center">

# 🤖 Robot Motion Planning and Control
### MSc Robotics and Artificial Intelligence — Course Repository

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Robotics Toolbox](https://img.shields.io/badge/Robotics%20Toolbox-1.0+-FF6B35?style=for-the-badge&logo=ros&logoColor=white)](https://github.com/petercorke/robotics-toolbox-python)
[![NumPy](https://img.shields.io/badge/NumPy-1.21+-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.4+-11557C?style=for-the-badge&logo=python&logoColor=white)](https://matplotlib.org/)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=for-the-badge)](https://github.com/umerahmedbaig7)
[![Field](https://img.shields.io/badge/Field-Robotics%20%26%20AI-blueviolet?style=for-the-badge&logo=ros&logoColor=white)]()

<br>

> *"A robot that cannot model its own dynamics cannot move with precision. A robot that cannot plan its path cannot move with purpose. Mastering both — from Newton-Euler inverse dynamics to closed-loop computed torque control — is what separates a manipulator from a machine."*

<br>

**Author:** Umer Ahmed Baig Mughal <br>
**Programme:** MSc Robotics and Artificial Intelligence <br>
**Specialization:** Machine Learning · Computer Vision · Human-Robot Interaction · Autonomous Systems · Robotic Motion Control <br>
**Institution:** ITMO University — Faculty of Control Systems and Robotics

</div>

---

## 📋 Table of Contents

- [📖 About This Repository](#-about-this-repository)
- [🗂️ Repository Structure](#️-repository-structure)
- [🔬 Course Overview](#-course-overview)
- [🧪 Lab Summaries](#-lab-summaries)
  - [🔴 Lab 1 — Dynamic Model of a Multi-Link Manipulator (Stanford Arm)](#-lab-1--dynamic-model-of-a-multi-link-manipulator-stanford-arm)
  - [🔵 Lab 2 — Forward and Inverse Kinematics, Workspace Analysis, and Trajectory Planning](#-lab-2--forward-and-inverse-kinematics-workspace-analysis-and-trajectory-planning)
  - [🟠 Lab 3 — Multi-Point Motion Trajectory Planning for the Stanford Manipulator](#-lab-3--multi-point-motion-trajectory-planning-for-the-stanford-manipulator)
  - [🟣 Lab 4 — Inverse Dynamics PD Control for the Stanford Manipulator](#-lab-4--inverse-dynamics-pd-control-for-the-stanford-manipulator)
- [🔗 Progressive Learning Pathway](#-progressive-learning-pathway)
- [⚙️ Common Platform — The Stanford Arm Model](#️-common-platform--the-stanford-arm-model)
- [🚀 Quick Start](#-quick-start)
- [📊 Results at a Glance](#-results-at-a-glance)
- [🧰 Tech Stack](#-tech-stack)
- [👤 Author](#-author)
- [📄 License](#-license)

---

## 📖 About This Repository

This repository contains the complete implementation of all four laboratory assignments from the **Robot Motion Planning and Control (RMPC)** course, part of the MSc in Robotics and Artificial Intelligence at ITMO University. The labs form a tightly coupled progressive series — beginning with the physical modelling of a 6-DOF manipulator and culminating in a fully closed-loop trajectory tracking control system — covering the essential theoretical and engineering pillars of modern robotic motion control.

The course advances along a deliberate three-stage hierarchy: **Stage 1** (Lab 1) constructs the complete dynamic model of the Stanford Arm, establishing the Newton-Euler inverse dynamics framework and extracting all fundamental equation components. **Stage 2** (Labs 2 and 3) extends this model into full kinematic analysis and trajectory planning — progressing from two-point joint-space trajectories with workspace construction and IK solving, to multi-waypoint Cartesian pipeline design. **Stage 3** (Lab 4) closes the control loop: a computed torque controller wraps the Lab 1 dynamic model and the Lab 2 LSPB trajectory into a full inverse dynamics PD control architecture, simulated with an RK45 integrator and validated under a 200 kg end-effector payload.

All four notebooks are implemented from first principles in Python using Peter Corke's Robotics Toolbox for Python. Every component — the DH parameter table, the physical dynamic parameterisation, the Newton-Euler solver, the Gauss-Newton and Levenberg-Marquardt IK solvers, the piecewise trajectory concatenation, and the `PD_regulator` callback — is constructed and analysed explicitly rather than called as a black box, ensuring depth of understanding at every level of the robotics stack.

### 🎯 What You Will Find Here

| 📁 Lab | 🏷️ Topic | 🧠 Core Concept | 🛠️ Key Tools |
|:------:|:--------:|:---------------:|:-------------:|
| Lab 1 | Dynamic Modelling | Newton-Euler Inverse Dynamics · M(q) · C(q,q̇) · G(q) | Robotics Toolbox · `robot.rne()` |
| Lab 2 | Kinematics & Trajectory Planning | FK · Workspace · Gauss-Newton IK · jtraj / LSPB / Quintic | Robotics Toolbox · SpatialMath |
| Lab 3 | Multi-Point Trajectory Planning | 4-Waypoint Pipeline · LM-IK · Piecewise Cubic · Torque Feasibility | Robotics Toolbox · SpatialMath |
| Lab 4 | Inverse Dynamics Control | Computed Torque Control · PD Feedback · `fdyn()` RK45 · Payload Robustness | Robotics Toolbox · NumPy |

---

## 🗂️ Repository Structure

```
📦 Robot-Motion-Planning-and-Control/
│
├── 📁 Lab_1/
│   ├── 📄 README.md                                           # Lab 1 full documentation
│   ├── 📁 src/
│   │   └── 📓 Stanford_Arm_Inverse_Dynamics.ipynb             # Newton-Euler dynamics · M, C, G · torque plots
│   └── 📁 results/
│       └── 🖼️  Joint_Torques.png                             # 6-panel joint torque/force — 3 scenarios
│
├── 📁 Lab_2/
│   ├── 📄 README.md                                           # Lab 2 full documentation
│   ├── 📁 src/
│   │   └── 📓 Stanford_Arm_Kinematics_Trajectory.ipynb        # FK · Workspace · IK · 3 trajectory methods
│   └── 📁 results/
│       ├── 🖼️  Workspace.png                                  # 3D reachable workspace — 27,000 FK points
│       ├── 🖼️  Joint_Positions.png                            # Position profiles — jtraj / LSPB / Quintic
│       ├── 🖼️  Joint_Velocities.png                           # Velocity profiles — method comparison
│       └── 🖼️  Joint_Accelerations.png                        # Acceleration profiles — method comparison
│
├── 📁 Lab_3/
│   ├── 📄 README.md                                           # Lab 3 full documentation
│   ├── 📁 src/
│   │   └── 📓 Stanford_Arm_Multi_Point_Trajectory.ipynb       # 4-waypoint pipeline · LM-IK · torque analysis
│   └── 📁 results/
│       ├── 🖼️  Workspace.png                                  # Workspace for waypoint selection
│       ├── 🖼️  Joint_States_Comprehensive.png                 # 24-panel: pos · vel · acc · torque × 6 joints
│       ├── 🖼️  EE_Trajectory_Simple.png                       # Simple 3D end-effector path
│       └── 🖼️  EE_Trajectory_MultiView.png                    # Speed-coloured 3D + 3 ortho projections
│
├── 📁 Lab_4/
│   ├── 📄 README.md                                           # Lab 4 full documentation
│   ├── 📁 src/
│   │   └── 📓 Stanford_Arm_Inverse_Dynamics_Control.ipynb     # CTC · fdyn() · RK45 · payload robustness
│   └── 📁 results/
│       ├── 🖼️  Position_Tracking_No_Payload.png               # Desired vs actual joint positions
│       ├── 🖼️  Velocity_Tracking_No_Payload.png               # Desired vs actual joint velocities
│       ├── 🖼️  Torques_No_Payload.png                         # Control torques — all 6 joints
│       ├── 🖼️  Tracking_Errors_No_Payload.png                 # Position error time histories
│       ├── 🖼️  Position_Tracking_With_Payload.png             # Position tracking — 200 kg payload
│       ├── 🖼️  Velocity_Tracking_With_Payload.png             # Velocity tracking — 200 kg payload
│       ├── 🖼️  Torques_With_Payload.png                       # Elevated torque profiles with payload
│       └── 🖼️  Tracking_Errors_With_Payload.png               # Error profiles — loaded vs unloaded
│
└── 📄 README.md                                               # ← You are here
```

---

## 🔬 Course Overview

The **Robot Motion Planning and Control** course develops the complete theoretical and computational toolkit required to model, plan for, and control the motion of robotic systems. The curriculum spans rigid-body dynamics, kinematic analysis, workspace characterisation, trajectory generation, and closed-loop model-based control — all grounded in both mathematical rigour and hands-on computational implementation.

The four labs advance in a deliberate three-stage progression unified by a single physical system — the **Stanford Arm**:

**Stage 1 — Dynamic Modelling (Lab 1):** The course opens with the hardest question in robotics: *what forces does the arm actually need to produce?* Lab 1 constructs the complete physical dynamic model of the Stanford Arm and solves the inverse dynamics problem via the Newton-Euler recursive algorithm, extracting the mass matrix M(q), Coriolis matrix C(q,q̇), and gravity vector G(q) as independent equation components.

**Stage 2 — Kinematic Analysis and Trajectory Planning (Labs 2 and 3):** With the dynamic model in place, Labs 2 and 3 address the geometric and temporal questions: *where can the arm reach, how does it get there, and through how many intermediate points?* Lab 2 establishes FK, workspace construction, and two-point trajectory planning with method comparison. Lab 3 scales this to a full 4-waypoint Cartesian pipeline with Levenberg-Marquardt IK, piecewise cubic trajectory construction, and comprehensive torque feasibility analysis.

**Stage 3 — Closed-Loop Control (Lab 4):** The course culminates by closing the feedback loop. Lab 4 implements computed torque control — wrapping the Lab 1 inverse dynamics model around a PD feedback law to theoretically linearise and decouple the closed-loop manipulator dynamics. The controller tracks a Lab 2 LSPB reference trajectory, is simulated with an RK45 forward dynamics integrator, and is validated against a 200 kg end-effector payload with RMS error quantification.

```
  Dynamic                Kinematic &            Multi-Point           Closed-Loop
  Modelling              Trajectory             Trajectory            Control
  ─────────────          ──────────────         ──────────────        ──────────────
   ┌─────────┐            ┌─────────┐            ┌─────────┐           ┌─────────┐
   │  LAB 1  │──────────► │  LAB 2  │──────────► │  LAB 3  │─────────► │  LAB 4  │
   └─────────┘            └─────────┘            └─────────┘           └─────────┘
   Stanford Arm           FK · Workspace         4-Waypoint            Computed Torque
   Newton-Euler RNE       Gauss-Newton IK        LM-IK Pipeline        PD Controller
   M(q) · C(q,q̇) · G(q)   jtraj / LSPB           Piecewise jtraj       fdyn() + RK45
   3 Scenarios            Trajectory Comparison  Torque Profiles       Payload Robust.

                               LSPB reference ──────────────────────────────────────► Lab 4
                               Dynamic model  ──────────────────────────────────────► Lab 4
                               LM-IK upgrade  ──────────────────────────────────────► Lab 3
```

---

## 🧪 Lab Summaries

---

### 🔴 Lab 1 — Dynamic Model of a Multi-Link Manipulator (Stanford Arm)

<div align="center">

[![Lab1](https://img.shields.io/badge/Lab%201-Dynamic%20Modelling-C0392B?style=flat-square&logo=python&logoColor=white)]()
[![Robot](https://img.shields.io/badge/Robot-Stanford%20Arm%206--DOF-red?style=flat-square)]()
[![Method](https://img.shields.io/badge/Method-Newton--Euler%20RNE-red?style=flat-square)]()
[![Output](https://img.shields.io/badge/Output-M(q)%20·%20C(q%2Cq̇)%20·%20G(q)-red?style=flat-square)]()

</div>

#### 📌 Task Description

> Given the **Stanford Arm** — a classical 6-DOF RRPRRR manipulator (five revolute joints + one prismatic) — fully parameterise its dynamic model with physically motivated link masses, inertia tensors, gear ratios, and friction coefficients. Plan a smooth joint-space trajectory and solve the **inverse dynamics problem** for three distinct motion scenarios using the Newton-Euler recursive algorithm, then extract and interpret the three fundamental dynamic equation components.

This lab establishes the **complete rigid-body dynamic model** that underpins all subsequent labs in the series, answering the foundational question: *"Given a desired joint-space trajectory, what torques and forces must every actuator produce to execute it?"*

**What the task requires:**
- Load the Stanford Arm via `rtb.models.DH.Stanford()`, read and interpret its **DH parameter table** (joint angle θ, offset d, link length a, twist α), and overwrite the toolbox defaults with physically motivated parameters across all six links: link masses, centres of mass, **diagonal inertia tensors** [Ixx, Iyy, Izz, Ixy, Iyz, Ixz], motor inertias Jm, viscous friction B, Coulomb friction pairs [Tc⁺, Tc⁻], and gear ratios G.
- Specify arbitrary initial and final joint configurations, visualise both via `robot.plot()`, and plan a smooth minimum-jerk **joint-space trajectory** using `rtb.jtraj()` — a quintic polynomial interpolator guaranteeing zero velocity and acceleration at both endpoints.
- Solve inverse dynamics using `robot.rne()` for three scenarios that isolate different subsets of τ = Mq̈ + Cq̇ + G + F: **full dynamics** (q̇ ≠ 0, q̈ ≠ 0), **quasi-static** (q̇ ≠ 0, q̈ = 0), and **static hold** (q̇ = 0, q̈ = 0) — each revealing the active dynamic contributions.
- Extract and numerically interpret the **three fundamental dynamic matrices** at the trajectory midpoint: `robot.inertia()` → M(q), `robot.coriolis()` → C(q,q̇), `robot.gravload()` → G(q).
- Plot **joint torque and force time histories** for all six joints across a 15-second trajectory for all three scenarios, comparing profiles to identify inertial, Coriolis, gravitational, and friction contributions.

#### 🔑 Key Concepts

| Concept | Description |
|---------|-------------|
| 🦾 DH Parameters | θ (joint angle), d (offset), a (link length), α (twist) — defines the kinematic chain for all 6 joints |
| ⚖️ Newton-Euler RNE | Recursive two-pass algorithm: forward velocity/acceleration propagation → backward force/torque propagation |
| 📐 Mass Matrix M(q) | Symmetric positive-definite inertia matrix — configuration-dependent, reflects effective inertia of each joint |
| 🌀 Coriolis Matrix C(q,q̇) | Velocity-product matrix capturing Coriolis and centrifugal effects — zero when q̇ = 0 |
| 🌍 Gravity Vector G(q) | Configuration-dependent gravitational load on each joint — the sole active term in static hold |
| 🔩 Prismatic Joint (J3) | Joint 3 output is a **force in Newtons** (not N·m) — limits in metres, not radians |
| 📈 Quintic Trajectory | `rtb.jtraj()` — fifth-order polynomial guaranteeing zero velocity and acceleration at endpoints |
| 📊 3-Scenario Comparison | Full / Quasi-static / Static: isolates M·q̈, C·q̇, and G contributions in torque plots |

#### 📤 Key Dynamic Results

```
Stanford Arm — 6-DOF RRPRRR  |  Trajectory: 15 s, 150 steps, quintic polynomial
Gravity vector: g = [0, 0, -9.81] m/s²  |  Dynamic parameters: physically motivated

Mass Matrix M(q) at trajectory midpoint — selected diagonal entries:
┌────────────────────────────────────────────────────────────────────────┐
│  M₁₁ = 3.46   M₂₂ = 3.50   M₃₃ = 7.11   M₄₄ = 0.12   M₆₆ = 0.028       │
│  Largest element: M₃₃ = 7.11  (prismatic joint — highest inertia)      │
│  Largest coupling: M₁₂ = M₂₁ = -0.446  (base-shoulder interaction)     │
└────────────────────────────────────────────────────────────────────────┘

Gravity Vector G(q) at midpoint — key values:
  J1 (Base):    ≈ 0 N·m        (rotation axis aligned with gravity vector)
  J3 (Elbow):   ≈ 63.47 N      (prismatic joint supports full wrist assembly weight)
  J5 (Wrist 2): ≈ 0.08 N·m    (small gravity offset)

Peak Coriolis coupling: C₂₃ = 0.5323, C₃₂ = −0.5267  (shoulder–elbow coupling)
```

📂 **[→ View Lab 1 Full Documentation](https://github.com/umerahmedbaig7/Robot-Motion-Planning-and-Control/blob/main/Lab_1/Readme.md)**

---

### 🔵 Lab 2 — Forward and Inverse Kinematics, Workspace Analysis, and Trajectory Planning

<div align="center">

[![Lab2](https://img.shields.io/badge/Lab%202-Kinematics%20%26%20Trajectory-0078D7?style=flat-square&logo=python&logoColor=white)]()
[![Task](https://img.shields.io/badge/Task-FK%20·%20Workspace%20·%20IK-lightblue?style=flat-square)]()
[![Solver](https://img.shields.io/badge/IK%20Solver-Gauss--Newton-lightblue?style=flat-square)]()
[![Trajectories](https://img.shields.io/badge/Methods-jtraj%20·%20LSPB%20·%20Quintic-lightblue?style=flat-square)]()

</div>

#### 📌 Task Description

> Extending the Lab 1 dynamic model, solve the **forward kinematics** to determine end-effector pose in SE(3), construct the complete **3D reachable workspace** by grid sampling over 27,000 joint configurations, solve the **inverse kinematics problem** numerically using the Gauss-Newton method, and plan and compare **three distinct trajectory types** — visualising position, velocity, and acceleration profiles for all six joints.

This lab addresses the geometric layer of the robotics stack: *"Where can the arm reach, how does it get to a specific point, and how does the choice of trajectory method affect the smoothness and actuator stress of the resulting motion?"*

**What the task requires:**
- Solve **forward kinematics** using `robot.fkine()` — interpreting the SE(3) result as a 3×3 rotation submatrix (orientation) and a 3×1 translation vector (Cartesian end-effector position in the base frame).
- Construct the full **reachable workspace** by sweeping the first three joint variables across their limits in a 30³ grid, computing FK at all 27,000 configurations, and plotting the resulting 3D point cloud to reveal the toroidal shell geometry.
- Solve the **inverse kinematics** problem using `robot.ikine_GN()` — the iterative Gauss-Newton method — with a `spatialmath.base.transl()` pure-translation SE(3) target for position-only IK.
- Implement and compare three trajectory planning methods between the IK-solved configurations: (1) **`rtb.jtraj()`** — quintic polynomial with smooth bell-shaped velocity; (2) **`rtb.mtraj(rtb.trapezoidal, ...)`** — LSPB with linear ramp, constant-velocity cruise, ramp-down; (3) **`rtb.mtraj(rtb.quintic, ...)`** — per-joint scalar quintic via the multi-axis wrapper.
- Produce 6-panel **position, velocity, and acceleration time-history figures** for all three methods, identifying the characteristic signatures of each trajectory type in the profiles.

#### 🔑 Key Concepts

| Concept | Description |
|---------|-------------|
| 🗺️ SE(3) Forward Kinematics | Product of 6 successive DH joint transforms → T₀₆ ∈ SE(3): rotation R + translation p |
| 🌐 Workspace Grid Sampling | 30³ = 27,000 FK evaluations over (J1, J2, J3) joint limits → toroidal shell point cloud |
| 🔁 Gauss-Newton IK | Iterative Jacobian pseudoinverse updates minimising SE(3) pose error: q ← q + J⁺·Δx |
| 📏 transl() IK Target | Position-only target: `sb.transl(px, py, pz)` — orientation unconstrained, 3-DOF task |
| 📈 jtraj | Quintic polynomial: zero vel + acc at endpoints, smooth bell-shaped velocity profile |
| 🔷 LSPB | Trapezoidal velocity: linear ramp + constant cruise + linear ramp — maximises throughput |
| 📐 mtraj/quintic | Per-joint scalar quintic: same boundary conditions as jtraj, independent per axis |
| ⚡ Acceleration Comparison | jtraj/quintic: smooth continuous curves; LSPB: piecewise-constant with step discontinuities |

#### 📤 Key Kinematic and Trajectory Results

```
Forward Kinematics — q_start = [0, −π/4, 0.2, 0, 0, 0]:
  End-effector position: p = [−0.1414, 0.1337, 0.5534] m

Workspace Construction:
  Grid resolution: 30 × 30 × 30 = 27,000 FK evaluations
  Geometry: Toroidal shell — central void from min prismatic extension (0 m)
  Outer boundary: max prismatic reach (0.5 m) + full J1 range (±π)

IK Verification (Gauss-Newton):
  Target: p = [−0.1414, 0.1337, 0.5534] m  |  Solver: ikine_GN()
  Result: Convergence confirmed — FK(IK solution) ≈ target within numerical tolerance

Trajectory Method Comparison — 5 s, 50 steps:
┌─────────────────┬─────────────────────────────┬───────────────────────────────────────┐
│ Method          │ Velocity Profile            │ Acceleration Profile                  │
├─────────────────┼─────────────────────────────┼───────────────────────────────────────┤
│ jtraj (quintic) │ Smooth bell — zero at bounds│ Smooth curves — zero at both endpoints│
│ LSPB            │ Trapezoidal — flat cruise   │ Piecewise constant — step at blends   │
│ mtraj/quintic   │ Smooth, near-identical jtraj│ Smooth — same boundary conditions     │
└─────────────────┴─────────────────────────────┴───────────────────────────────────────┘
Conclusion: Quintic methods optimal for smooth actuation; LSPB for time-critical tasks
```

📂 **[→ View Lab 2 Full Documentation](https://github.com/umerahmedbaig7/Robot-Motion-Planning-and-Control/blob/main/Lab_2/Readme.md)**

---

### 🟠 Lab 3 — Multi-Point Motion Trajectory Planning for the Stanford Manipulator

<div align="center">

[![Lab3](https://img.shields.io/badge/Lab%203-Multi--Point%20Trajectory-E67E22?style=flat-square&logo=python&logoColor=white)]()
[![Waypoints](https://img.shields.io/badge/Waypoints-4%20Cartesian%20Points-orange?style=flat-square)]()
[![Solver](https://img.shields.io/badge/IK%20Solver-Levenberg--Marquardt-orange?style=flat-square)]()
[![Segments](https://img.shields.io/badge/Segments-3%20×%203s%20Cubic%20Polynomial-orange?style=flat-square)]()

</div>

#### 📌 Task Description

> Develop a complete **multi-waypoint trajectory planning pipeline** for the Stanford Arm: define four Cartesian end-effector positions distributed across the workspace, solve their joint-space representations via Levenberg-Marquardt IK, construct cubic polynomial trajectory segments connecting consecutive waypoints, and analyse the resulting motion through position, velocity, acceleration, and torque profiles across all six joints — with a professional multi-panel end-effector path visualisation.

This lab scales the two-point trajectory framework of Lab 2 to a full production-grade multi-waypoint pipeline, answering the engineering question: *"How do you specify a task-space motion through multiple Cartesian goals and translate it into a dynamically feasible joint-space execution plan?"*

**What the task requires:**
- Maintain the full **Stanford Arm dynamic model** from Labs 1–2 as the consistent parameterised foundation, verifying parameters via `robot.links[0].dyn()`.
- Select **four Cartesian end-effector positions** distributed across the reachable workspace volume — at different heights, lateral extents, and depths — to produce a geometrically diverse and mechanically challenging trajectory.
- Solve **IK for all four waypoints** using `robot.ikine_LM()` — the Levenberg-Marquardt solver, which combines gradient descent and Gauss-Newton methods for robust convergence from a fixed initial guess across geometrically diverse targets. Verify all four solutions before proceeding.
- Construct **three jtraj segments** over t = [0,3], [3,6], [6,9] s — independently solving the quintic polynomial boundary conditions for each consecutive waypoint pair — and concatenate position, velocity, and acceleration arrays into a continuous 9-second time-series.
- Produce a **24-panel comprehensive joint state figure** (4 rows × 6 columns): position with ±0.02 rad uncertainty band, velocity with zero-reference line, acceleration with zero-reference line, and estimated torque from `robot.rne()` — across all six joints.
- Visualise the **Cartesian end-effector path** through a 6-panel professional layout: speed-coloured 3D trajectory with waypoint markers, XY / XZ / YZ orthogonal projections, and a waypoint information panel.

#### 🔑 Key Concepts

| Concept | Description |
|---------|-------------|
| 🏹 LM IK Solver | `ikine_LM()` — adaptive damping combines GN stability and gradient descent convergence robustness |
| 🔗 Piecewise jtraj | 3 independent quintic segments concatenated: q(t) piecewise over [0–3], [3–6], [6–9] s |
| 🛑 Zero-Velocity Waypoints | Each segment endpoint has zero boundary velocity — robot decelerates to full stop at P2, P3 |
| 📐 Torque Feasibility | `robot.rne()` confirms smooth bounded torque curves — no impulsive spikes across 9 s trajectory |
| 🗺️ Curved Cartesian Path | Joint-space quintic ≠ straight-line Cartesian — FK nonlinearity produces curved EE path |
| 🎨 Speed-Coloured 3D | `Line3DCollection` + `ListedColormap` — trajectory colour encodes instantaneous segment speed |
| 📊 24-Panel Figure | GridSpec layout: position/vel/acc/torque × 6 joints with uncertainty bands and zero references |
| 🆚 LM vs GN Upgrade | LM handles 4 diverse waypoints from fixed initial guess more reliably than Gauss-Newton (Lab 2) |

#### 📤 Key Multi-Point Trajectory Results

```
Stanford Arm — 4-Waypoint Pipeline  |  Trajectory: 9 s total, 3 × 3 s segments, 300 total points
IK Solver: Levenberg-Marquardt (ikine_LM)  |  Reference: jtraj quintic per segment

IK Convergence — All 4 waypoints solved successfully:
  Structural observation: J4 = ±π/2 (±1.5708 rad) consistent across all solutions —
  wrist aligns EE frame with base frame orientation under position-only IK target

Trajectory Quality:
  ✅ Position: smooth quintic within each segment, correct waypoint interpolation
  ✅ Velocity: bell-shaped per segment; zero at all 4 waypoints (piecewise boundary condition)
  ✅ Acceleration: continuous within segments, zero at boundaries — correct jtraj parameterisation
  ✅ Torque: smooth, bounded throughout — dynamic feasibility confirmed via rne()

End-Effector Path:
  Curved Cartesian trajectory (expected: joint-space → nonlinear FK → curved EE path)
  All 4 waypoints confirmed within workspace volume across 3 orthogonal projections
  Peak speed coloured on 3D trajectory — dark blue (slow) to deepskyblue (fast)
```

📂 **[→ View Lab 3 Full Documentation](https://github.com/umerahmedbaig7/Robot-Motion-Planning-and-Control/blob/main/Lab_3/Readme.md)**

---

### 🟣 Lab 4 — Inverse Dynamics PD Control for the Stanford Manipulator

<div align="center">

[![Lab4](https://img.shields.io/badge/Lab%204-Inverse%20Dynamics%20Control-8E44AD?style=flat-square&logo=python&logoColor=white)]()
[![Architecture](https://img.shields.io/badge/Architecture-Computed%20Torque%20Control-purple?style=flat-square)]()
[![Simulation](https://img.shields.io/badge/Simulation-fdyn()%20%2B%20RK45-purple?style=flat-square)]()
[![Payload](https://img.shields.io/badge/Payload%20Test-200%20kg%20EE%20Load-purple?style=flat-square)]()

</div>

#### 📌 Task Description

> Implement a complete **closed-loop trajectory tracking control system** for the Stanford Arm using an inverse dynamics PD controller (computed torque control). A `PD_regulator` callback — combining PD feedback with full inverse dynamics feedforward — is passed to `robot.fdyn()`, which integrates the closed-loop equations of motion using an RK45 ODE solver. Tracking performance is quantified via RMS error metrics and then stress-tested by repeating the full experiment with a **200 kg end-effector payload**.

This lab closes the feedback loop on the entire course, answering the ultimate control question: *"How do you make a real manipulator actually follow a desired trajectory, and how robust is that control to a massive, unexpected change in payload?"*

**What the task requires:**
- Load the Stanford Arm with the full Lab 1 dynamic parameterisation and use the **LSPB reference trajectory** from Lab 2 (`rtb.mtraj(rtb.trapezoidal, ...)`) — directly connecting this lab's control design to the trajectory planning work of the earlier course.
- Implement the **inverse dynamics PD control law** (computed torque control): τ = Kp·(q_des − q_act) + Kd·(q̇_des − q̇_act) + M(q_act)·q̈_des + C(q_act, q̇_act)·q̇_des + G(q_act) — where the feedforward term theoretically cancels all nonlinear dynamics, linearising and decoupling the closed-loop system to six independent double integrators.
- Tune **diagonal PD gain matrices** Kp and Kd per joint, reflecting differing inertia and load characteristics: high gains (Kp = 500–1000) for heavy proximal joints (J1–J3), lower gains (Kp = 50–200) for lightweight distal wrist joints (J4–J6).
- Implement the `PD_regulator` callback that is invoked at each RK45 step by `robot.fdyn()` — performing trajectory index lookup, extracting desired state, computing error signals, evaluating `robot.inertia()`, `robot.coriolis()`, and `robot.gravload()` from the actual state, and returning the control torque vector.
- Simulate **closed-loop forward dynamics** using `robot.fdyn()` with the RK45 integrator, recovering velocity profiles via `np.gradient()` numerical differentiation, and plotting position, velocity, torque, and error profiles for all six joints.
- Repeat the full experiment with `robot.payload(200)` — adding a **200 kg end-effector load** — and quantify the residual RMS tracking error change to demonstrate the feedforward's inherent payload compensation.

#### 🔑 Key Concepts

| Concept | Description |
|---------|-------------|
| 🎯 Computed Torque Control | τ = PD feedback + full inverse dynamics feedforward — linearises and decouples closed-loop dynamics |
| 🔁 fdyn() + RK45 | `robot.fdyn()` integrates forward dynamics under control law; RK45 ODE solver for state evolution |
| 📡 PD_regulator Callback | Per-step callback: time lookup → desired state extraction → error computation → model evaluation → τ |
| 🔧 Diagonal Gain Matrices | Kp, Kd ∈ ℝ⁶ˣ⁶ diagonal — per-joint tuning: J1–J3 high gains, J4–J6 lower gains |
| 💪 Feedforward Compensation | M(q_act)·q̈_des + C(q_act,q̇_act)·q̇_des + G(q_act) — cancels all nonlinear dynamics if model is exact |
| 📦 Payload Robustness | `robot.payload(200)` modifies M, C, G in-place — feedforward re-evaluates automatically at every step |
| 📉 RMS Error Quantification | Per-joint RMS position error computed for unloaded and loaded conditions — direct performance metric |
| 🔗 Cross-Lab Integration | LSPB reference from Lab 2 + dynamic model from Lab 1 — synthesises the full course pipeline |

#### 📤 Key Control and Robustness Results

```
Stanford Arm — Computed Torque Control  |  Reference: LSPB (Lab 2)  |  Gains: diagonal Kp, Kd
Simulation: robot.fdyn() + RK45 integrator  |  Output: tg.q joint positions

RMS Position Tracking Errors — No Payload vs 200 kg End-Effector Payload:
┌────────────────────────────┬────────────────────┬──────────────────────────┬──────────┐
│ Joint                      │ No Payload (rad)   │ With 200 kg Payload (rad)│  Change  │
├────────────────────────────┼────────────────────┼──────────────────────────┼──────────┤
│ J1 — Base                  │     0.027229       │         0.027162         │  −0.25%  │
│ J2 — Shoulder              │     0.021500       │         0.021373         │  −0.59%  │
│ J3 — Elbow (prismatic)     │     0.001659       │         0.001708         │  +2.95%  │
│ J4 — Wrist 1               │     0.017986       │         0.017909         │  −0.43%  │
│ J5 — Wrist 2               │     0.010281       │         0.010241         │  −0.39%  │
│ J6 — End-Effector          │     0.015041       │         0.014958         │  −0.55%  │
└────────────────────────────┴────────────────────┴──────────────────────────┴──────────┘
Key result: RMS errors change by < 3% across all joints despite a 200 kg payload —
direct demonstration of inverse dynamics feedforward's inherent payload compensation.

J3 (prismatic) achieves the smallest absolute error (0.0017 rad) — linear joints
are inherently simpler to control due to absence of trigonometric configuration dependence.
```

📂 **[→ View Lab 4 Full Documentation](https://github.com/umerahmedbaig7/Robot-Motion-Planning-and-Control/blob/main/Lab_4/Readme.md)**

---

## 🔗 Progressive Learning Pathway

The four labs are not independent exercises — they form a tightly coupled engineering progression in which each lab builds directly on the foundations established by its predecessors. Understanding this chain is essential for interpreting any individual lab's design decisions.

```
LAB 1 ─────────────────────────────────────────────────────────────────────► LAB 4
       Provides: Stanford Arm dynamic model (M, C, G, friction, gear ratios)
       Used in: Labs 2, 3, 4 — identical robot object with identical parameters

LAB 2 ─────────────────────────────────────────────────────────────────────► LAB 4
       Provides: LSPB reference trajectory (rtb.mtraj with trapezoidal profile)
       Used in: Lab 4 as the controller's reference input signal

LAB 2 ──────────────────────────────────────────────────────────────────────► LAB 3
       Provides: Gauss-Newton IK methodology and jtraj two-point framework
       Upgraded in: Lab 3 — GN → Levenberg-Marquardt; 2-point → 4-waypoint
```

**Technology escalation across the Stanford Arm thread:**

| Capability | Lab 1 | Lab 2 | Lab 3 | Lab 4 |
|:----------:|:-----:|:-----:|:-----:|:-----:|
| Robot Model | ✅ Established | ✅ Inherited | ✅ Inherited | ✅ Inherited |
| IK Solver | — | Gauss-Newton | Levenberg-Marquardt | — |
| Trajectory Points | 2 (jtraj) | 2 (3 methods) | 4 waypoints (3 segments) | LSPB from Lab 2 |
| Control Loop | Open-loop | Open-loop | Open-loop | **Closed-loop** |
| Dynamics Used | Inverse (RNE) | — | RNE feasibility | Forward (fdyn + RNE) |
| Simulation | — | — | — | RK45 integrator |

---

## ⚙️ Common Platform — The Stanford Arm Model

### 🦾 The Shared Robot Model

Labs 1, 2, 3, and 4 operate on a single shared **Stanford Arm** instance — a 6-DOF RRPRRR manipulator with one prismatic joint — loaded from Peter Corke's Robotics Toolbox and then overwritten with a complete set of physically motivated dynamic parameters. This parameterisation is established once in Lab 1 and reproduced identically in all subsequent labs, ensuring full continuity of the physical model across the laboratory series.

```
Joint Configuration:    R   R   P   R   R   R
                        J1  J2  J3  J4  J5  J6
                       Base Shldr Elbow Wr1 Wr2  EE

DH Parameters (standard modified DH convention):
┌───────┬──────────────────┬────────────────┬──────────┬────────────────┬───────┐
│ Joint │       θ (rad)    │     d (m)      │  a (m)   │   α (rad)      │ Type  │
├───────┼──────────────────┼────────────────┼──────────┼────────────────┼───────┤
│  J1   │       q₁         │    0.412       │  0.000   │    −π/2        │   R   │
│  J2   │       q₂         │    0.154       │  0.000   │    +π/2        │   R   │
│  J3   │       0          │    q₃          │  0.000   │    0           │   P   │
│  J4   │       q₄         │    0.000       │  0.000   │    −π/2        │   R   │
│  J5   │       q₅         │    0.000       │  0.000   │    +π/2        │   R   │
│  J6   │       q₆         │    0.000       │  0.000   │    0           │   R   │
└───────┴──────────────────┴────────────────┴──────────┴────────────────┴───────┘
Note: J3 is prismatic — limits in metres, dynamics output in Newtons (not N·m)
```

### 📐 Shared Dynamic Parameter Specification

The full dynamic parameterisation — applied identically in Labs 1, 2, 3, and 4 — covers all six links:

| Parameter | Description | Labs Applied |
|-----------|-------------|:------------:|
| **Link masses** m [kg] | Physically motivated per-link masses (J1–J6) | 1, 2, 3, 4 |
| **Centre of mass** r [m] | 3-vector relative to link frame origin | 1, 2, 3, 4 |
| **Inertia tensor** I [kg·m²] | Diagonal form [Ixx, Iyy, Izz, Ixy, Iyz, Ixz] | 1, 2, 3, 4 |
| **Motor inertia** Jm | Reflected rotor inertia | 1, 2, 3, 4 |
| **Viscous friction** B | Linear velocity-proportional damping | 1, 2, 3, 4 |
| **Coulomb friction** Tc | [Tc⁺, Tc⁻] — asymmetric static friction pair | 1, 2, 3, 4 |
| **Gear ratio** G | Transmission ratio — highest (80) at J3 | 1, 2, 3, 4 |
| **Joint limits** qlim | Revolute: radians; Prismatic J3: metres | 1, 2, 3, 4 |

### 🔁 Shared Robotics Toolbox API Pattern

The core API pattern used across all four labs:

```python
import roboticstoolbox as rtb
import numpy as np

# Load and parameterise — Lab 1, then inherited in Labs 2, 3, 4
robot = rtb.models.DH.Stanford()
robot.links[i].m   = ...    # link mass
robot.links[i].r   = ...    # centre of mass
robot.links[i].I   = ...    # inertia tensor
robot.links[i].Jm  = ...    # motor inertia
robot.links[i].B   = ...    # viscous friction
robot.links[i].Tc  = ...    # Coulomb friction
robot.links[i].G   = ...    # gear ratio
robot.links[i].qlim = ...   # joint limits

# Trajectory planning — Lab 2 pattern, reused in Labs 3, 4
traj = rtb.jtraj(q_start, q_end, t_array)            # quintic polynomial
traj = rtb.mtraj(rtb.trapezoidal, q_start, q_end, t) # LSPB

# Inverse dynamics — established Lab 1, used in Labs 3, 4
tau  = robot.rne(q, qd, qdd)           # Newton-Euler torques/forces
M    = robot.inertia(q)                 # mass matrix
C    = robot.coriolis(q, qd)            # Coriolis matrix
G    = robot.gravload(q)                # gravity vector

# Inverse kinematics — Lab 2 (GN), upgraded to LM in Lab 3
sol  = robot.ikine_GN(T_target)        # Gauss-Newton (Lab 2)
sol  = robot.ikine_LM(T_target)        # Levenberg-Marquardt (Lab 3)

# Closed-loop simulation — Lab 4 only
tg   = robot.fdyn(T, q0, torqfun=PD_regulator)  # RK45 forward dynamics
```

---

## 🚀 Quick Start

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/umerahmedbaig7/Robot-Motion-Planning-and-Control.git
cd Robot-Motion-Planning-and-Control
```

### 2️⃣ Create a Virtual Environment and Install Dependencies

```bash
# Create and activate virtual environment
python -m venv rmpc_env
source rmpc_env/bin/activate        # Windows: rmpc_env\Scripts\activate

# Install all dependencies (covers all 4 labs)
pip install roboticstoolbox-python spatialmath-python numpy matplotlib jupyter
```

> 📌 `spatialmath-python` is an automatic dependency of `roboticstoolbox-python` — it is installed automatically but listed explicitly for clarity.

Verify the installation:

```python
import roboticstoolbox as rtb
import numpy as np
import matplotlib
print("All dependencies installed successfully.")
print(f"  roboticstoolbox: {rtb.__version__}")
print(f"  numpy: {np.__version__}")
print(f"  matplotlib: {matplotlib.__version__}")
```

### 3️⃣ Launch Jupyter and Run Each Lab

```bash
jupyter notebook
```

Then open the notebooks in sequence:

```
# 🔴 Lab 1 — Dynamic Modelling (~2–5 min)
Open:  Lab_1/src/Stanford_Arm_Inverse_Dynamics.ipynb

# 🔵 Lab 2 — Kinematics & Trajectory Planning (~3–6 min)
Open:  Lab_2/src/Stanford_Arm_Kinematics_Trajectory.ipynb

# 🟠 Lab 3 — Multi-Point Trajectory Planning (~3–6 min)
Open:  Lab_3/src/Stanford_Arm_Multi_Point_Trajectory.ipynb

# 🟣 Lab 4 — Inverse Dynamics Control (~5–15 min, RK45 integration)
Open:  Lab_4/src/Stanford_Arm_Inverse_Dynamics_Control.ipynb
```

| 📁 Lab | Notebook | Estimated Runtime |
|:------:|----------|:-----------------:|
| Lab 1 | `Stanford_Arm_Inverse_Dynamics.ipynb` | ~2–5 min |
| Lab 2 | `Stanford_Arm_Kinematics_Trajectory.ipynb` | ~3–6 min |
| Lab 3 | `Stanford_Arm_Multi_Point_Trajectory.ipynb` | ~3–6 min |
| Lab 4 | `Stanford_Arm_Inverse_Dynamics_Control.ipynb` | ~5–15 min |

> ⚠️ **Lab 4 runtime note:** The `robot.fdyn()` RK45 integration is the most computationally intensive step in the series. Runtime scales with the number of trajectory steps and the RK45 internal step count. The no-payload and payload experiments each run a full forward dynamics simulation — total runtime may vary by hardware.

> ⚠️ **Lab 4 payload ordering:** After `robot.payload(200)` is called, the robot model is modified in-place for all subsequent computations. If re-running individual cells out of order, restart the kernel and re-run from the top to ensure a clean model state.

### 4️⃣ Visualisation Backend Note

`robot.plot()` renders using either the **Swift browser-based 3D viewer** (if `swift-sim` is installed) or the **Matplotlib backend**. Both produce equivalent configuration visualisations. The lab notebooks use the Matplotlib backend by default. To use Swift:

```bash
pip install swift-sim
```

---

## 📊 Results at a Glance

### 🔴 Lab 1 — Dynamic Equation Components

| Component | Key Value | Physical Interpretation |
|:---------:|:---------:|------------------------|
| M₃₃ (mass matrix) | **7.11 kg·m²** | Largest diagonal entry — prismatic joint has the highest effective inertia |
| G₃ (gravity vector) | **63.47 N** | Largest load — prismatic actuator supports full wrist assembly against gravity |
| G₁ (gravity vector) | **≈ 0 N·m** | Machine-precision zero — base axis aligned with gravity in this configuration |
| M₁₂ coupling | **−0.446** | Largest off-diagonal — significant base-shoulder inertial interaction |
| C₂₃ Coriolis | **+0.5323** | Peak coupling — shoulder–prismatic elbow velocity interaction |

### 🔵 Lab 2 — Kinematic and Trajectory Planning

| Metric | Value | Notes |
|:------:|:-----:|-------|
| Workspace grid points | **27,000** (30³) | J1 × J2 × J3 grid sweep |
| Workspace geometry | **Toroidal shell** | Central void: min prismatic extension = 0 m |
| IK solver | **Gauss-Newton** | Converges to target within numerical tolerance |
| Best trajectory (smoothness) | **jtraj / mtraj-quintic** | Zero-boundary vel + acc, smooth profiles |
| Best trajectory (throughput) | **LSPB** | Maximises constant-velocity time; preferred for pick-and-place |

### 🟠 Lab 3 — Multi-Point Trajectory

| Metric | Value | Notes |
|:------:|:-----:|-------|
| Waypoints | **4** Cartesian positions | Distributed across workspace volume |
| Segments | **3** (3 s each) | Total trajectory: 9 s, 300 points |
| IK solver | **Levenberg-Marquardt** | Upgraded from Lab 2 GN — more robust for diverse targets |
| Torque feasibility | **✅ Confirmed** | Smooth bounded torques via rne() — no impulsive spikes |
| Cartesian path | **Curved** (expected) | Joint-space quintic + nonlinear FK → curved EE path |

### 🟣 Lab 4 — Inverse Dynamics Control

| Joint | No Payload RMS (rad) | 200 kg Payload RMS (rad) | Error Change |
|:-----:|:--------------------:|:------------------------:|:------------:|
| J1 — Base | 0.027229 | 0.027162 | −0.25% |
| J2 — Shoulder | 0.021500 | 0.021373 | −0.59% |
| J3 — Elbow (P) | 0.001659 | 0.001708 | **+2.95%** |
| J4 — Wrist 1 | 0.017986 | 0.017909 | −0.43% |
| J5 — Wrist 2 | 0.010281 | 0.010241 | −0.39% |
| J6 — End-Effector | 0.015041 | 0.014958 | −0.55% |

> **Key takeaway:** Adding a 200 kg payload changes RMS tracking error by **less than 3% across all joints** — a direct quantitative demonstration of the computed torque controller's inherent payload compensation through the inverse dynamics feedforward term.

---

## 🧰 Tech Stack

<div align="center">

| 🛠️ Tool | 🔖 Version | 🎯 Role in This Course | 🧪 Used In |
|:-------:|:---------:|:---------------------:|:----------:|
| ![Python](https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white) | 3.11+ | Core language — all notebooks, robot modelling, algorithms, visualisation | All |
| ![Robotics Toolbox](https://img.shields.io/badge/-Robotics%20Toolbox-FF6B35?logo=ros&logoColor=white) | ≥ 1.0 | Stanford Arm model, `rne()`, `inertia()`, `coriolis()`, `gravload()`, `fkine()`, `ikine_GN/LM()`, `jtraj()`, `mtraj()`, `fdyn()`, `payload()` | All |
| ![SpatialMath](https://img.shields.io/badge/-SpatialMath-7D3C98?logo=python&logoColor=white) | ≥ 1.0 | `SE3`, `SO3`, `transl()` — SE(3) homogeneous transforms, IK target construction | Labs 2, 3 |
| ![NumPy](https://img.shields.io/badge/-NumPy-013243?logo=numpy&logoColor=white) | ≥ 1.21 | Array maths, `np.gradient()`, `np.diag()`, `np.vstack()`, `np.linalg.norm()` | All |
| ![Matplotlib](https://img.shields.io/badge/-Matplotlib-11557C?logo=python&logoColor=white) | ≥ 3.4 | All figures: torque subplots, 3D workspace scatter, trajectory profiles, `plot_surface()`, `GridSpec`, `Line3DCollection` | All |
| ![Jupyter](https://img.shields.io/badge/-Jupyter-F37626?logo=jupyter&logoColor=white) | Any | Interactive notebook environment for all four labs | All |

</div>

**No pre-built motion planning libraries. No control design toolboxes. No black-box trajectory optimisers.** Every algorithm in this course — from the Newton-Euler recursive two-pass dynamics solver, to the Gauss-Newton and Levenberg-Marquardt IK iteration schemes, to the PD_regulator computed torque callback — is implemented and analysed explicitly, ensuring complete comprehension of the mathematics at every stage of the robotics control stack.

---

## 👤 Author

<div align="center">

### Umer Ahmed Baig Mughal

🎓 **MSc Robotics and Artificial Intelligence** — ITMO University <br>
🏛️ *Faculty of Control Systems and Robotics* <br>
🔬 *Specialization: Machine Learning · Computer Vision · Human-Robot Interaction · Autonomous Systems · Robotic Motion Control*

[![GitHub](https://img.shields.io/badge/GitHub-umerahmedbaig7-181717?style=for-the-badge&logo=github)](https://github.com/umerahmedbaig7)

</div>

---

## 📄 License

This repository is intended for **academic and research use**. All work was developed as part of the *Robot Motion Planning and Control* course within the MSc Robotics and Artificial Intelligence program at ITMO University. Redistribution, modification, and use in derivative academic work are permitted with appropriate attribution to the original author.

---

<div align="center">

*Robot Motion Planning and Control — MSc Robotics and Artificial Intelligence | ITMO University*

⭐ *If this repository helped you understand robotic dynamics, motion planning, or model-based control, consider giving it a star!* ⭐

</div>
