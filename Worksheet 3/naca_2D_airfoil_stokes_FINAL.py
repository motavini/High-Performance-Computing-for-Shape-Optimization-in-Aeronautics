import gmsh
import numpy as np
import matplotlib.pyplot as plt
from dolfinx.io.gmshio import model_to_mesh
from dolfinx.io import XDMFFile
from mpi4py import MPI

import matplotlib.pyplot as plt
from petsc4py import PETSc
from petsc4py.PETSc import ScalarType

from dolfinx import fem, io, mesh, plot
from dolfinx.fem.petsc import LinearProblem
from dolfinx.fem import (Expression, Function, functionspace,
                         assemble_scalar, dirichletbc, form, locate_dofs_topological, locate_dofs_geometrical)
from basix.ufl import element, mixed_element

import ufl
from ufl import TrialFunction, TestFunction, TrialFunctions, TestFunctions
from ufl import inner, dot, grad, dx, ds, div
from ufl import Identity, FacetNormal, sym, as_vector, Measure

from codetiming import Timer

def rotate_2D(theta, x, y):
    theta_rad = np.radians(theta) # converts to radians
    x_r = x * np.cos(theta_rad) - y * np.sin(theta_rad)
    y_r = x * np.sin(theta_rad) + y * np.cos(theta_rad)
    return x_r, y_r

def generate_naca4_profile(four_digits, chord_length, n, angle_of_attack):

    def camber_line(x, m, p, c):
        yc = np.where(x < p * c,  
              m * (x / np.power(p, 2)) * (2 * p - (x / c)),  
              m * ((c - x) / np.power(1-p, 2)) * (1 + (x / c) - 2 * p))
        return yc
    
    def thickness(x, t, c):
        term1 =  0.2969 * (np.sqrt(x/c))
        term2 = -0.1260 * (x/c)
        term3 = -0.3516 * np.power(x/c,2)
        term4 =  0.2843 * np.power(x/c,3)
        term5 = -0.1015 * np.power(x/c,4)
        return 5 * t * c * (term1 + term2 + term3 + term4 + term5)
    
    def dyc_over_dx(x, m, p, c):
        return np.where(x < p * c,
                        2*m / np.power(p, 2) * (p - x / c),
                        2*m / np.power(1-p, 2) * (p - x / c))
    
    m = int(four_digits[0]) / 100 # maximum camber
    p = int(four_digits[1]) / 10 # position of maximum camber
    t = int(four_digits[2:]) / 100 # maximum thickness

    x = np.linspace(0, 1, n)
    x = (0.5 * (1 - np.cos(np.pi * x))) * chord_length

    
    dyc_dx = dyc_over_dx(x, m, p, chord_length)
    theta = np.arctan(dyc_dx)
    yt = thickness(x, t, chord_length)
    yc = camber_line(x, m, p, chord_length)

    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

    # Rotation of the frame of reference
    xu, yu = rotate_2D(angle_of_attack, xu, yu)
    xl, yl = rotate_2D(angle_of_attack, xl, yl)


    x_profile = np.concatenate([xu[:-1], xl[::-1]])[:-1]
    y_profile = np.concatenate([yu[:-1], yl[::-1]])[:-1]       

    return x_profile, y_profile


def create_NACA_mesh(theta, camber):

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("Naca mesh")

    # defining the geometry
    L = 0.5  # characteristic length 

    # add points for the rectangle
    p1 = gmsh.model.geo.addPoint(-15, -6, 0, L)
    p2 = gmsh.model.geo.addPoint(15, -6, 0, L)
    p3 = gmsh.model.geo.addPoint(15, 6, 0, L)
    p4 = gmsh.model.geo.addPoint(-15, 6, 0, L)

    # add lines for the rectangle
    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p1)

    x_center_coord = 0
    y_center_coord = 0
    x, y = generate_naca4_profile(camber, chord_length=6.0, n=100, angle_of_attack=theta)
    x, y = x + x_center_coord, y + y_center_coord

    airfoil_profile_points = []
    for i, j in zip(x, y):
        mesh_point = gmsh.model.geo.addPoint(i, j, 0, L/10)
        airfoil_profile_points.append(mesh_point)
    
    if not np.allclose(airfoil_profile_points[0], airfoil_profile_points[-1]):
        airfoil_profile_points.append(airfoil_profile_points[0])

    airfoil_spline = gmsh.model.geo.addSpline(airfoil_profile_points)

    # create curve loops and plane surfaces
    far_field_loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    airfoil_loop = gmsh.model.geo.addCurveLoop([airfoil_spline])
    plane_surface = gmsh.model.geo.addPlaneSurface([far_field_loop, airfoil_loop])

    # synchronizing
    gmsh.model.geo.synchronize()

    # defining physical groups for boundary conditions
    fluid = gmsh.model.addPhysicalGroup(2, [plane_surface], 0)  # Fluid
    gmsh.model.setPhysicalName(2, fluid, "Fluid")

    inflow = gmsh.model.addPhysicalGroup(1, [l4], 1)  # Inflow
    gmsh.model.setPhysicalName(1, inflow, "Inflow")

    outflow = gmsh.model.addPhysicalGroup(1, [l2], 2)  # Outflow
    gmsh.model.setPhysicalName(1, outflow, "Outflow")

    walls = gmsh.model.addPhysicalGroup(1, [l3, l1], 3)  # Walls
    gmsh.model.setPhysicalName(1, walls, "Walls")

    obstacle = gmsh.model.addPhysicalGroup(1, [airfoil_spline], 4)  # Obstacle
    gmsh.model.setPhysicalName(1, obstacle, "Obstacle")

    # generating the mesh
    gmsh.model.mesh.generate(2)

    # visualizing and converting to FEM mesh
    # gmsh.fltk.run()

    mesh, cell_tags, facet_tags = model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)

    # Finalize Gmsh
    gmsh.finalize()

    return mesh, facet_tags

def run_stokes(theta, camber):

    msh, facet_tags = create_NACA_mesh(theta, camber)
    epsilon = 1e-6 

    # Define function spaces for velocity (V) and pressure (Q)
    V = element("Lagrange", msh.basix_cell(), 2, shape=(msh.geometry.dim,))
    Q = element("Lagrange", msh.basix_cell(), 1)
    VQ = mixed_element([V, Q])
    W = fem.functionspace(msh, VQ)
    
    # Collapse the function spaces
    V_collapse, _ = W.sub(0).collapse()
    Q_collapse, _ = W.sub(1).collapse()

    # Get boundary facets
    facets_inflow = facet_tags.find(1)
    facets_outflow = facet_tags.find(2)
    facets_wall = facet_tags.find(3)
    facets_obstacle = facet_tags.find(4)

    # Create boundary conditions for the walls, obstacle, inflow and outflow
    fdim  = msh.topology.dim - 1
    def u_nonslip(x): return np.zeros((msh.topology.dim, x.shape[1]), dtype=PETSc.ScalarType)  
    u_walls = fem.Function(V_collapse) 
    u_walls.interpolate(u_nonslip)

    #Walls
    dofs_walls = fem.locate_dofs_topological((W.sub(0), V_collapse), fdim, facets_wall) # No dofs_walls 
    bc_wall = dirichletbc(value = u_walls, dofs=dofs_walls, V= W.sub(0)) # do i need to collapse W? -> V=V_collapse

    #Airfoil 
    dofs_obstacle = fem.locate_dofs_topological((W.sub(0), V_collapse), fdim, facets_obstacle)
    bc_obstacle = dirichletbc(value = u_walls, dofs=dofs_obstacle, V= W.sub(0))

    #inflow
    def inflow_expression(x): return np.array([-(1/24) * (x[1] - 6) * (x[1] + 6), np.zeros_like(x[1])], dtype=PETSc.ScalarType)
    dofs_inflow = fem.locate_dofs_topological((W.sub(0), V_collapse), fdim, facets_inflow)
    inflow_profile = fem.Function(V_collapse)
    inflow_profile.interpolate(inflow_expression)
    bc_inflow = dirichletbc(value = inflow_profile, dofs=dofs_inflow, V=W.sub(0))

    # Outflow 
    def p_zero(x): return np.zeros((1, x.shape[1]), dtype=PETSc.ScalarType) 
    p_out = fem.Function(Q_collapse) 
    p_out.interpolate(p_zero)
    dofs_outflow = fem.locate_dofs_topological((W.sub(1), Q_collapse), fdim, facets_outflow)
    bc_outflow = dirichletbc(value = p_out, dofs=dofs_outflow, V= W.sub(1))

    # Next, the variational problem is defined:
    (u,p) = TrialFunctions(W)
    (v,q) = TestFunctions(W)

    # ufl expression
    x = ufl.SpatialCoordinate(msh)
    a = (inner(grad(u), grad(v)) - p * div(v) + q * div(u) + p*q*epsilon) * dx 
    L = fem.Constant(msh, PETSc.ScalarType(0)) * (q) * dx

    bcs = [bc_inflow, bc_outflow, bc_wall, bc_obstacle]
    problem = LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    wh = problem.solve()
    p = wh.sub(1).collapse() 
    uh = wh.sub(0).collapse() 

    # stress
    I = Identity(2) 
    J_eqn = 0.5*inner(grad(uh), grad(uh))*dx
    J = assemble_scalar(form(J_eqn))

    # stress = -p*I + 2*sym(grad(uh))
    stress = -p * Identity(len(uh)) + grad(uh) + grad(uh).T

    # Normal faces
    n = FacetNormal(msh)

    df = as_vector([1, 0])  # Drag direction
    dp = as_vector([0, 1])  # Lift direction

    # Obstacles faces
    fdim = msh.topology.dim - 1
    obstacle_marker_id = 4
    facet_values = np.full(len(facets_obstacle), obstacle_marker_id, dtype=np.int32) 
    facet_tags = mesh.meshtags(msh, msh.topology.dim - 1, facets_obstacle, facet_values) 
    dObs = Measure("ds", domain=msh, subdomain_data=facet_tags)(obstacle_marker_id)

    # Calculate drag and lift forces
    Fd_equ = dot(stress * n, df)*dObs
    Fl_equ = dot(stress * n, dp)*dObs
    drag = assemble_scalar(form(Fd_equ))
    lift = assemble_scalar(form(Fl_equ))

    return lift, drag, J, uh, p, msh


def run_simulations(cambers, angles):
    """
    Run simulations for different camber values and angles.
    """
    dataset = []
    for camber in cambers:
        j_values = []
        fd_values = []
        fl_values = []
        for angle in angles:
            fl, fd, J, uh, p, msh = run_stokes(angle, camber)
            j_values.append(J)
            fd_values.append(fd)
            fl_values.append(fl)
            #print(f"Camber: {camber}, Angle: {angle}")
        data = {'j': j_values, 'fl': fl_values, 'fd': fd_values, 'camber': camber, 'angle': angles}
        dataset.append(data)
    return dataset

def plot_results(dataset):
    """
    Plot the results of the simulation.

    """

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
    
    plt.subplot(2, 2, 1)
    for data in dataset:
        plt.plot(data["angle"], data["fd"], label=data["camber"])
    plt.title("Fd")
    plt.legend()

    plt.subplot(2, 2, 2)
    for data in dataset:
        plt.plot(data["angle"], data["fl"], label=data["camber"])
    plt.title("Fl")
    plt.legend()


    

    plt.subplot(2, 2, 3)
    for data in dataset:
        coeffs = []
        for (lift, drag) in zip(data["fl"],data["fd"]):
            coeffs.append(lift/drag)

        plt.plot(data["angle"], coeffs, label=data["camber"])
    plt.title("Fl / Fd")
    plt.legend()

    plt.subplot(2, 2, 4)
    for data in dataset:
        plt.plot(data["angle"], data["j"], label=data["camber"])
    plt.title("J")
    plt.legend()

    fig.suptitle("All Cambers")
    plt.show()

def save_results(theta, camber):
    """
    Prints the values for the drag, lift and objective function

    Saves the veolocity and presure fields in vtk
    """
    F_d, F_l, J, uh, p, msh = run_stokes(theta, camber)

    # Assigning names
    uh.name = "velocity"
    p.name = "pressure"

    
    with io.VTKFile(MPI.COMM_WORLD, "velocity.pvd", "w") as vtk:
        vtk.write_function(uh)

    with io.VTKFile(MPI.COMM_WORLD, "pressure.pvd", "w") as vtk:
        vtk.write_function(p)


    print("J =", J)
    print("-------------------------------")
    print("Drag force Fd =", F_d, "N")
    print("-------------------------------")
    print("Lift force Fl =", F_l, "N")


def pareto(n_samples, camber_range=(0, 0.09)):
    """
    params: 
    n_samples (int): number of sampled angles and cambers. Grid size will be n_samples x n_samples
    camber_range (tuple): (min_camber, max_camber) as decimal fractions

    returns:
    tuple: (drag_forces, lift_forces, J_values, optimal_params)

    Performs Pareto front analysis for airfoil optimization

    """

    alpha_values = np.linspace(0, 180, n_samples)
    camber_values = np.linspace(camber_range[0], camber_range[1], n_samples)

    alpha_grid, camber_grid = np.meshgrid(alpha_values, camber_values)
    param_pairs = np.vstack((alpha_grid.ravel(), camber_grid.ravel())).T

    total_pairs = len(param_pairs)

    J_values = np.zeros(total_pairs)       # To store J values
    drag_forces = np.zeros(total_pairs)    # To store drag forces F_d
    lift_forces = np.zeros(total_pairs)    # To store lift forces F_l

    for i, (alpha, camber) in enumerate(param_pairs):
        print(f"Simulating pair {i+1}/{total_pairs}...", end="\n ", flush=True)
        m = int(round(camber * 100))
        four_digits = f"{m}412"
        fl, fd, J_val, *_ = run_stokes(alpha, four_digits)

        lift_forces[i] = fl               # Store lift force
        drag_forces[i] = -fd               # Store drag force
        J_values[i] = J_val               # Store objective function J
    
    # Compute Pareto front indices
    pareto_indices = compute_pareto_front(lift_forces, drag_forces)

    # Create a scatter plot of Drag vs Lift for the Pareto front

    plt.figure(figsize=(8, 6))
    plt.scatter(drag_forces, lift_forces, marker='x', color='blue', label='Solutions')

    # Highlight Pareto optimal solutions
    plt.scatter(drag_forces[pareto_indices], lift_forces[pareto_indices], 
                marker='o', color='red', label='Pareto Front')
    
    plt.xlabel("Drag Force ($F_d$)")
    plt.ylabel("Lift Force ($F_l$)")
    plt.title("Pareto Front: Drag vs. Lift for 100 (α, camber) pairs")
    plt.legend()
    plt.grid(True)

    plt.savefig('pareto.png', dpi=300, bbox_inches='tight')

    optimal_idx = np.argmax(lift_forces - drag_forces) # Takes the pairs with maximum Lift - Drag
    optimal_params = param_pairs[optimal_idx]

    optimal_alpha, optimal_camber = optimal_params

    print(f"Optimal (α, camber): α = {optimal_alpha:.3f} degrees, camber = {optimal_camber:.3f}")
    print(f"Optimal Forces: F_d = {drag_forces[optimal_idx]:.3f}, F_l = {lift_forces[optimal_idx]:.3f}, J = {J_values[optimal_idx]:.3f}")

    return drag_forces, lift_forces, J_values, optimal_params

def compute_pareto_front(lift, drag):
    """
    Compute the indices of the Pareto front.
    Returns a list of indices corresponding to non-dominated solutions.
    """
    n = len(lift)
    is_dominated = np.zeros(n, dtype=bool)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # Check if j dominates i
            if (lift[j] >= lift[i] and drag[j] <= drag[i]) and (lift[j] > lift[i] or drag[j] < drag[i]):
                is_dominated[i] = True
                break

    pareto_indices = np.where(~is_dominated)[0]
    return pareto_indices


if __name__ == "__main__":
   drag_forces, lift_forces, J_values, optimal_params = pareto(n_samples=10)