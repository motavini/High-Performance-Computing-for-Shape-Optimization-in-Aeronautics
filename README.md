# 2SC7692 – High Performance Computing for Shape Optimization and Drag Reduction in Aeronautics ✈️

**Authors:** Vinícius da Mata e Mota, Hugo Gandhi, Sushant Patil  
**Course:** CentraleSupélec – 2SC7692

---

## 📘 Overview

This project focuses on **high-performance computing (HPC)** for **aerodynamic optimization**, leveraging **Computational Fluid Dynamics (CFD)** and **Finite Element Methods (FEM)** to solve large-scale simulations efficiently.  
Using **OpenMPI** for distributed parallelization and running simulations on the DCE cluster at Centrale 
Supélec, we accelerate PDE solvers, analyze lift and drag behavior under varying conditions, and perform 3D airflow simulations around a Boeing 767 to investigate how geometry, discretization, and meshing strategies influence aerodynamic performance.
## 🛠️ Tools & Technologies

- **FEniCSx + MPI (OpenMPI)** – Parallel finite element solver for large-scale PDE simulations  
- **Gmsh / Rhino 8 / MeshMixer** – Geometry & meshing  
- **VTK (ParaView)** – Visualization

---

## 📑 Highlights

### 🧪 Worksheets
- **Worksheet 1:** Poisson solver, FEM basics, convergence analysis  
- **Worksheet 2:** Stokes flow, pressure & velocity fields, drag/lift computation  
- **Worksheet 3:** 2D NACA airfoil simulation with varying angle of attack & camber

### ✈️ Boeing 767 CFD
- 3D simulation with multiple mesh refinements  
- Parallel execution on HPC cluster using MPI  
- Velocity, pressure, drag, and lift field visualization  
- Runtime: ~35 s (coarse) → ~8 min (fine mesh)

---

