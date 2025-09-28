import gmsh
import numpy as np
import matplotlib.pyplot as plt
from dolfinx.io.gmshio import model_to_mesh
from mpi4py import MPI
import pandas as pd

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

def rotate_3D(coord, alpha_rad):
    """
    Rotates a 3D coordinate around the X-axis by a given angle.

    :param coord: A list or array of 3D coordinates [x, y, z].
    :param alpha_rad: The rotation angle in radians.
    :return: A list of rotated coordinates [x, y', z'].
    """
    cos_a = np.cos(alpha_rad)
    sin_a = np.sin(alpha_rad)
    x, y, z = coord
    y_r = y * cos_a - z * sin_a
    z_r = y * sin_a + z * cos_a
    return [x, y_r, z_r]


def stokes_solver(mesh, facets, inflow, angle):
    """
    Solves the Stokes problem for the given mesh and facets.

    :param mesh: The mesh to solve the problem on.
    :param facets: Facets of the mesh.
    :param inflow: Function defining the inflow velocity.
    :param angle: Angle of attack in degrees.
    :return: None. Outputs results to files and prints forces and coefficients.
    """
    lift, drag, J, uh, p, msh = run_stokes(mesh, facets, inflow, angle)
    
    j_val = np.array(J, dtype='float64')
    fd_val = np.array(drag, dtype='float64')
    fl_val = np.array(lift, dtype='float64')

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    global_j = np.empty_like(j_val)
    global_fd = np.empty_like(fd_val)
    global_fl = np.empty_like(fl_val)

    comm.Reduce([j_val, MPI.DOUBLE], [global_j, MPI.DOUBLE], op=MPI.SUM, root=0)
    comm.Reduce([fd_val, MPI.DOUBLE], [global_fd, MPI.DOUBLE], op=MPI.SUM, root=0)
    comm.Reduce([fl_val, MPI.DOUBLE], [global_fl, MPI.DOUBLE], op=MPI.SUM, root=0)

    if rank == 0:
        print("J =", global_j)
        print("-------------------------------")
        print("Drag force Fd =", global_fd, "N")
        print("-------------------------------")
        print("Lift force Fl =", global_fl, "N")


    p.name = "pressure" 
    file1 = io.VTXWriter(comm, "pressure_out.bp", p, engine ="BP4") 
    file1.write(0.0)
    file1.close() 

    uh.name = "velocity"
    file2 = io.VTXWriter(comm, "velocity_out.bp", uh, engine="BP4") 
    file2.write(0.0)
    file2.close()

    """
    with io.VTKFile(comm, "test.pvd", "w") as file_vtk:
        file_vtk.write_mesh(msh)
        file_vtk.write_function(uh, t=0.0)
        file_vtk.write_function(p, t=0.0)
   
    # writer = io.XDMFFile(comm,"test.xdmf", 'w')
    # writer.write_mesh(msh)
    # writer.write_function(uh)
    # writer.write_function(p)
    # writer.close()
    """

def solve_and_save_stokes_in_range_angle(angles, model_num, inflow_speed):
    """
    Solves and saves the results of the Stokes problem over a range of angles of attack.

    :param angles: List of angles of attack to solve for.
    :param model_num: Integer representing the mesh model number (1 to 4).
    :param inflow_speed: Speed of the inflow in the simulation.
    :return: None. Saves results to files and prints forces and coefficients.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    j_values_ls = []
    fd_values_ls = []
    fl_values_ls = []
    angle_ls = []

    for angle in angles:
        #print(f"angle is {angle}")
        inflow = np.array([0,0,-1])
        inflow_rotated = inflow_speed * np.array(rotate_3D(inflow, np.deg2rad(angle))) # rotate to match the box xyz coordinate system
        inflow = lambda x: (np.stack((np.zeros(x.shape[1]), inflow_rotated[1] * np.ones(x.shape[1]), inflow_rotated[2] * np.ones(x.shape[1]))))
        
        mesh, _, facets = io.gmshio.read_from_msh(f"boeing_{model_num}_msh/boeing_{model_num}_{angle}.msh", MPI.COMM_WORLD, 0, gdim=3)
        fl, fd, J, u, p, mesh = run_stokes(mesh, facets, inflow, angle)

        j_val = np.array(J, dtype='float64')
        fd_val = np.array(fd, dtype='float64')
        fl_val = np.array(fl, dtype='float64')

        global_j = np.empty_like(j_val)
        global_fd = np.empty_like(fd_val)
        global_fl = np.empty_like(fl_val)

        comm.Reduce([j_val, MPI.DOUBLE], [global_j, MPI.DOUBLE], op=MPI.SUM, root=0)
        comm.Reduce([fd_val, MPI.DOUBLE], [global_fd, MPI.DOUBLE], op=MPI.SUM, root=0)
        comm.Reduce([fl_val, MPI.DOUBLE], [global_fl, MPI.DOUBLE], op=MPI.SUM, root=0)

        if rank == 0: # only append on the first processor
            j_values_ls.append(global_j.item())
            fd_values_ls.append(global_fd.item())
            fl_values_ls.append(global_fl.item())
            angle_ls.append(angle)
            print("-------------------------------")
            print(f"Results for angle {angle}:")
            print("-------------------------------")
            print("J =", global_j)
            print("-------------------------------")
            print("Drag force Fd =", global_fd, "N")
            print("-------------------------------")
            print("Lift force Fl =", global_fl, "N")

        p.name = "pressure" 
        file1 = io.VTXWriter(comm, f"outputs/boeing_{model_num}_simulation_results/{angle}/pressure_boeing_{model_num}_{angle}.bp", p, engine ="BP4") 
        file1.write(0.0)
        file1.close() 

        u.name = "velocity"
        file2 = io.VTXWriter(comm, f"outputs/boeing_{model_num}_simulation_results/{angle}/velocity_boeing_{model_num}_{angle}.bp", u, engine="BP4") 
        file2.write(0.0)
        file2.close()

    if rank == 0:
        data = {'angle': angle_ls, 'j': j_values_ls, 'fl': fl_values_ls, 'fd': fd_values_ls}
        df = pd.DataFrame(data)
        df.to_csv(f"outputs/boeing_results_{model_num}.csv")
        plot_stokes_results(data)


def plot_stokes_results(data):
    """
    Plots the results of the Stokes problem.

    :param data: Dictionary containing results with keys 'angle', 'j', 'fl', and 'fd'.
    :return: None. Displays plots of the results.
    """
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))

    plt.subplot(2, 2, 1)
    plt.plot(data["angle"], data["fd"])
    plt.title("Fd")

    plt.subplot(2, 2, 2)
    plt.plot(data["angle"], data["fl"])
    plt.title("Fl")

    plt.subplot(2, 2, 3)
    coeffs = []
    for (lift, drag) in zip(data["fl"],data["fd"]):
        coeffs.append(lift/drag)
    plt.plot(data["angle"], coeffs)
    plt.title("Fl / Fd")

    plt.subplot(2, 2, 4)
    plt.plot(data["angle"], data["j"])
    plt.title("J")

    plt.show()

def flow_field_mesh(surface_file, angle_of_attack, mesh_size_min=None, mesh_size_max=None, bounding_box=(-150, -150, -150, 300, 300, 300)):
    """
    Generates a 3D volume mesh around the airplane surface.

    :param surface_file: Path to the airplane surface file.
    :param angle_of_attack: Angle of attack in degrees.
    :param mesh_size_min: Minimum mesh size (optional).
    :param mesh_size_max: Maximum mesh size (optional).
    :param bounding_box: Bounding box dimensions as (xmin, ymin, zmin, dx, dy, dz).
    :return: Tuple containing the mesh and facet tags.
    """
    
    # Initialize the model
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("AirplaneModel")

    # Create box around airplane
    xmin, ymin, zmin, dx, dy, dz = bounding_box
    channel = gmsh.model.occ.addBox(xmin, ymin, zmin, dx, dy, dz)
    gmsh.model.occ.synchronize()

    # Coordinates of the CoM for each box surface
    inlet_coord = [dx/2 + xmin, dy/2 + ymin, zmin + dz]
    outlet_coord = [dx/2 + xmin, dy/2 + ymin, zmin]
    left_wall_coord = [xmin, dy/2 + ymin, dz/2 + zmin]
    right_wall_coord = [dx + xmin, dy/2 + ymin, dz/2 + zmin]
    upper_wall_coord = [dx/2 + xmin, dy + ymin, dz/2 + zmin]
    lower_wall_coord = [dx/2 + xmin, ymin, dz/2 + zmin]


    angle = np.radians(angle_of_attack)  # Convert degrees to radians
    axis_point = [0, 0, 0]
    axis_direction = [1, 0, 0]  # Direction vector of the X-axis

    # Rotating box coordinates
    inlet_coord = rotate_3D(inlet_coord, angle)
    outlet_coord = rotate_3D(outlet_coord, angle)
    left_wall_coord = rotate_3D(left_wall_coord, angle)
    right_wall_coord = rotate_3D(right_wall_coord, angle)
    upper_wall_coord = rotate_3D(upper_wall_coord, angle)
    lower_wall_coord = rotate_3D(lower_wall_coord, angle)


    gmsh.model.occ.rotate([(3, 1)], *axis_point, *axis_direction, angle)

    # Remove box volume
    gmsh.model.occ.remove([(3, 1)])
    gmsh.model.occ.synchronize()

    # Merge airplane surface
    airplane = gmsh.merge(surface_file)
    gmsh.model.geo.synchronize()

    # Create surface loops
    box_surface_loop = gmsh.model.geo.addSurfaceLoop([6, 1, 3, 5, 4, 2])
    airplane_surface_loop = gmsh.model.geo.addSurfaceLoop([7])

    # Create volume
    gmsh.model.geo.addVolume([box_surface_loop, airplane_surface_loop])
    gmsh.model.geo.synchronize()
    #gmsh.fltk.run()

    ##### DEBUG
    entities = gmsh.model.getEntities(dim=3)  # Get all 3D entities
    print("Entities in the model:", entities)
    #####

    volume = gmsh.model.getEntities(dim=3)

    # Define physical groups
    gmsh.model.addPhysicalGroup(volume[0][0], [volume[0][1]], 0)
    gmsh.model.setPhysicalName(volume[0][0], 0, "Fluid")

    surfaces = gmsh.model.occ.getEntities(dim=2)
    inlet_marker, outlet_marker, wall_marker, obstacle_marker = 1, 2, 3, 4
    walls = []

    inlet = None  # Initialize inlet
    for surface in surfaces:
        CoM = gmsh.model.occ.getCenterOfMass(surface[0], surface[1])
        if np.allclose(CoM, inlet_coord):
            inlet = surface[1]
            gmsh.model.addPhysicalGroup(surface[0], [surface[1]], inlet_marker)
            gmsh.model.setPhysicalName(surface[0], inlet_marker, "Inflow")
        elif np.allclose(CoM, outlet_coord):
            gmsh.model.addPhysicalGroup(surface[0], [surface[1]], outlet_marker)
            gmsh.model.setPhysicalName(surface[0], outlet_marker, "Outflow")
        elif np.logical_or(np.allclose(CoM, left_wall_coord), np.allclose(CoM, right_wall_coord)):
            walls.append(surface[1])
        elif np.logical_or(np.allclose(CoM, lower_wall_coord), np.allclose(CoM, upper_wall_coord)):
            walls.append(surface[1])

    
    gmsh.model.addPhysicalGroup(2, walls, wall_marker)
    gmsh.model.setPhysicalName(2, wall_marker, "walls")
    gmsh.model.addPhysicalGroup(2, [7], obstacle_marker)
    gmsh.model.setPhysicalName(2, obstacle_marker, "airplane")

    ##### DEBUG
    #gmsh.fltk.run()

    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(3)

    #gmsh.fltk.run()

    mesh, cell_tags, facet_tags = model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=3)
    gmsh.write("Boeing_mesh_facets.msh")

    with io.VTKFile(mesh.comm, "test_volume.pvd", "w") as file_vtk:
        file_vtk.write_mesh(mesh)
    
    gmsh.finalize()

    return mesh, facet_tags


def run_stokes(msh, facet_tags, inflow_expression, angle):

    comm = MPI.COMM_WORLD
    #print(f"Rank {comm.rank}: Ghost cells (global numbering): {msh.topology.index_map(msh.topology.dim).ghosts}")

    epsilon = 1e-6 
    # Define function spaces for velocity (V) and pressure (Q)
    V = element("Lagrange", msh.basix_cell(), 2, shape=(msh.geometry.dim,))
    Q = element("Lagrange", msh.basix_cell(), 1)
    VQ = mixed_element([V, Q])
    W = fem.functionspace(msh, VQ)
    
    # Collapse the function spaces
    V_collapse, _ = W.sub(0).collapse()
    Q_collapse, _ = W.sub(1).collapse()

    # Wall amd obstacle Profile
    def u_nonslip(x): return np.zeros((msh.topology.dim, x.shape[1]), dtype=PETSc.ScalarType)  
    u_walls = fem.Function(V_collapse) 
    u_walls.interpolate(u_nonslip)

    # Inflow Profile
    inflow_profile = fem.Function(V_collapse)
    inflow_profile.interpolate(inflow_expression)

    # Outflow Profile 
    def p_zero(x): return np.zeros((1, x.shape[1]), dtype=PETSc.ScalarType) 
    p_out = fem.Function(Q_collapse) 
    p_out.interpolate(p_zero)

    # Boundary conditions
    fdim  = msh.topology.dim - 1
    bc_inflow = create_bc(W.sub(0), V_collapse, facet_tags, inflow_profile, 1, fdim)
    bc_outflow = create_bc(W.sub(1), Q_collapse, facet_tags, p_out, 2, fdim)
    bc_wall = create_bc(W.sub(0), V_collapse, facet_tags, u_walls, 3, fdim)
    bc_obstacle = create_bc(W.sub(0), V_collapse, facet_tags, u_walls, 4, fdim)     

    # Define the variational problem is defined:
    (u,p) = TrialFunctions(W)
    (v,q) = TestFunctions(W)
    #ufl expression
    x = ufl.SpatialCoordinate(msh)
    a = (inner(grad(u), grad(v)) - p * div(v) + q * div(u) + p*q*epsilon) * dx 
    L = fem.Constant(msh, PETSc.ScalarType(0)) * (q) * dx

    bcs = [bc_inflow, bc_outflow, bc_wall, bc_obstacle]
    #print("beginning solve")
    problem = LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    wh = problem.solve()
    p = wh.sub(1).collapse() 
    uh = wh.sub(0).collapse() 

    J_eqn = 0.5*inner(grad(uh), grad(uh))*dx
    J = assemble_scalar(form(J_eqn))
    stress = -p * Identity(len(uh)) + grad(uh) + grad(uh).T

    # Normal faces
    n = FacetNormal(msh)

    df_flat = np.array([0, 0, 1])  # Drag direction
    dp_flat = np.array([0, -1, 0])  # Lift direction
    df = as_vector(rotate_3D(df_flat, np.deg2rad(angle))) # rotate so drag is normal to inflow face
    dp = as_vector(rotate_3D(dp_flat, np.deg2rad(angle))) # rotate so lift is perpendicular to flow

    # Obstacles faces
    fdim = msh.topology.dim - 1
    obstacle_marker_id = 4
    #facet_values = np.full(len(facets_obstacle), obstacle_marker_id, dtype=np.int32) 
    #facet_tags = mesh.meshtags(msh, msh.topology.dim - 1, facets_obstacle, facet_values) 
    dObs = Measure("ds", domain=msh, subdomain_data=facet_tags)(obstacle_marker_id)

    # Calculate drag and lift forces
    Fd_equ = dot(stress * n, df)*dObs
    Fl_equ = dot(stress * n, dp)*dObs
    drag = assemble_scalar(form(Fd_equ))
    lift = assemble_scalar(form(Fl_equ))

    return lift, drag, J, uh, p, msh

def create_bc(W_subspace, vector_space, facet_tags, func, tag, fdim):
    dofs = fem.locate_dofs_topological((W_subspace, vector_space), fdim, facet_tags.find(tag))
    return dirichletbc(func, dofs, W_subspace)

if __name__ == "__main__":

    inflow_speed = 500
    angles = np.arange(-60, 61, 20)
    model_num = 4 # integer from 1 to 4, where 1 represents the coarsest mesh
    solve_and_save_stokes_in_range_angle(angles, model_num, inflow_speed)
    
    # angle = -40
    # inflow = np.array([0,0,-1])
    # inflow_rotated = inflow_speed * np.array(rotate_3D(inflow, np.deg2rad(angle)))
    # inflow = lambda x: (np.stack((np.zeros(x.shape[1]), inflow_rotated[1] * np.ones(x.shape[1]), inflow_rotated[2] * np.ones(x.shape[1]))))
    # msh, ct, facets = io.gmshio.read_from_msh(f"boeing_{model_num}_msh/boeing_{model_num}_{angle}.msh", MPI.COMM_WORLD, 0, gdim=3)
    # stokes_solver(msh, facets, inflow, angle)
    

