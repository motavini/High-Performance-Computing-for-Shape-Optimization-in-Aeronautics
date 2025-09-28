# Note that it is important to first `from mpi4py import MPI` to
# ensure that MPI is correctly initialised.
from mpi4py import MPI
from petsc4py.PETSc import ScalarType  # type: ignore
import matplotlib.pyplot as plt

# +
import numpy as np

from dolfinx import fem, io, mesh, plot
from dolfinx.fem.petsc import LinearProblem
from petsc4py import PETSc
from dolfinx.fem import (Expression, Function, functionspace,
                         assemble_scalar, dirichletbc, form, locate_dofs_topological, locate_dofs_geometrical)
from basix.ufl import element, mixed_element


import ufl
from ufl import TrialFunction, TestFunction, TrialFunctions, TestFunctions
from ufl import inner, dot, grad, dx, ds, div

from codetiming import Timer

import meshio
import gmsh
import pygmsh

def u_ex(mod):
    return lambda x: ufl.sin(ufl.pi * x[0])*ufl.sin(ufl.pi * x[1])

u_numpy = u_ex(np)
u_ufl = u_ex(ufl)

resolution = 0.01
L = 2.2
H = 0.41
c = [0.2, 0.2, 0]
r = 0.05

geometry = pygmsh.geo.Geometry()
model = geometry.__enter__()
circle = model.add_circle(c, r, mesh_size=resolution)


points = [
    model.add_point((0, 0), mesh_size=resolution),
    model.add_point((L, 0), mesh_size=5 * resolution),
    model.add_point((L, H), mesh_size=5 * resolution),
    model.add_point((0, H), mesh_size=resolution),
]

channel_lines = [
    model.add_line(points[i], points[i + 1]) for i in range(-1, len(points) - 1)
]

# Create a line loop and plane surface for meshing
channel_loop = model.add_curve_loop(channel_lines)
plane_surface = model.add_plane_surface(channel_loop, holes=[circle.curve_loop])

# Call gmsh kernel before add physical entities
model.synchronize()


volume_marker = 6
model.add_physical([plane_surface], "Volume")
model.add_physical([channel_lines[0]], "Inflow")
model.add_physical([channel_lines[2]], "Outflow")
model.add_physical([channel_lines[1], channel_lines[3]], "Walls")
model.add_physical(circle.curve_loop.curves, "Obstacle")

geometry.generate_mesh(dim=2)
gmsh.write("mesh.msh")
gmsh.clear()
geometry.__exit__()



def create_mesh(mesh, cell_type, prune_z=False):
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    points = mesh.points[:, :2] if prune_z else mesh.points
    out_mesh = meshio.Mesh(
        points=points, cells={cell_type: cells}, cell_data={"name_to_read": [cell_data]}
    )
    return out_mesh

#mesh_from_file = meshio.read("mesh.msh")

#line_mesh = create_mesh(mesh_from_file, "line", prune_z=True)
#triangle_mesh = create_mesh(mesh_from_file, "triangle", prune_z=True)

# meshio.write("facet_mesh.xdmf", line_mesh)
# triangle_mesh = create_mesh(mesh_from_file, "triangle", prune_z=True)
#meshio.write("mesh.vtk", triangle_mesh)
#meshio.write("facet_mesh.vtk", line_mesh)

with io.XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "w") as xdmf:
        msh, ct, _ = io.gmshio.read_from_msh("mesh.msh", MPI.COMM_WORLD, 0, gdim=2)
        #xdmf.write_mesh(msh)

def inflow_boundary(x):
    return np.isclose(x[0], 0)

def wall_boundaries(x):
    return np.logical_or(np.isclose(x[1], 0), np.isclose(x[1], H))

def outflow_boundary(x):
    return np.isclose(x[0], L)

def circle_boundary(x):
    return np.isclose((x[0] - c[0])**2 + (x[1]-c[1])**2 - r**2, 0)



def stokes_solver(N, msh, degree=1):
    #boundaries u =0 
    V = element("Lagrange", msh.basix_cell(), 2, shape=(msh.geometry.dim,))
    Q = element("Lagrange", msh.basix_cell(), 1)
    VQ = mixed_element([V, Q])
    W = fem.functionspace(msh, VQ)
    V_collapse, _ = W.sub(0).collapse()
    Q_collapse, _ = W.sub(1).collapse()

    #u_noslip = fem.Function(W.sub(0).collapse()[0])
    #u_noslip.x.array[:] = 0
    
    #u_nonslip = np.array((0,) * msh.geometry.dim, dtype=PETSc.ScalarType)

    wall_profile = fem.Function(V_collapse)

    def u_nonslip(x):
        return np.zeros((2, x.shape[1]), dtype=PETSc.ScalarType)

    wall_profile.interpolate(u_nonslip)

    #V, _ = W.sub(0).collapse()
 
    def zero(x):     
        return np.zeros((msh.topology.dim, x.shape[1]), dtype=PETSc.ScalarType) 
    u_walls = fem.Function(V_collapse) 
    u_walls.interpolate(zero)

    #Walls
    fdim  = msh.topology.dim - 1
    facets_wall = mesh.locate_entities_boundary(msh, fdim, wall_boundaries)
    dofs_walls = fem.locate_dofs_topological((W.sub(0), V_collapse), fdim, facets_wall) # No dofs_walls 
    bc_wall = dirichletbc(value = u_walls, dofs=dofs_walls, V= W.sub(0)) # do i need to collapse W? -> V=V_collapse

    #Circle 
    u_nonslip = np.array((0,) * msh.geometry.dim, dtype=PETSc.ScalarType)
    facets_circle = mesh.locate_entities_boundary(msh, fdim, circle_boundary)
    dofs_circle = fem.locate_dofs_topological((W.sub(0), V_collapse), 1, facets_circle)
    bc_circle = dirichletbc(value = u_walls, dofs=dofs_circle, V= W.sub(0))

    #inflow
    fdim  = msh.topology.dim - 1
    facets_inflow = mesh.locate_entities_boundary(msh, fdim, inflow_boundary)
    dofs_inflow = fem.locate_dofs_topological((W.sub(0), V_collapse), fdim, facets_inflow)
    #print(dofs_inflow)
    #inflow_profile = fem.Function(W)
    inflow_profile = fem.Function(V_collapse)

    class InletVelocity():

        def __call__(self, x):
            values = np.zeros((2, x.shape[1]), dtype=PETSc.ScalarType)
            values[0] = -(1/24) * (x[1] - H) * (x[1])
            return values

    def inflow_expression(x):
        #values = np.zeros((2, x.shape[1]), dtype=PETSc.ScalarType)
        #values[0] = -(1/24) * (x[1] - 6) * (x[1] + 6)
        #return values
        #return np.array([-(1/24) * (x[1] - 6) * (x[1] + 6), np.zeros_like(x[1])], dtype=PETSc.ScalarType)
        return np.array([(100) * (H - x[1]) * x[1], np.zeros_like(x[1])], dtype=PETSc.ScalarType)

    #inflow_profile.interpolate(inflow_expression)
    #inflow = InletVelocity()
    inflow_profile.interpolate(inflow_expression)
    #print(inflow_profile)
    bc_inflow = dirichletbc(value = inflow_profile, dofs=dofs_inflow, V=W.sub(0))

    # Outflow 
    def zero_(x):     
        return np.zeros((1, x.shape[1]), dtype=PETSc.ScalarType) 
    p_out = fem.Function(Q_collapse) 
    p_out.interpolate(zero_)
    facets_outflow = mesh.locate_entities_boundary(msh, fdim, outflow_boundary)
    dofs_outflow = fem.locate_dofs_topological((W.sub(1), Q_collapse), fdim, facets_outflow)
    #p0 = np.array((0,) * msh.geometry.dim, dtype=PETSc.ScalarType)
    bc_outflow = dirichletbc(value = p_out, dofs=dofs_outflow, V= W.sub(1))

    # Next, the variational problem is defined:
    # u = TrialFunction(V)
    # v = TestFunction(V)

    (u,p) = TrialFunctions(W)
    (v,q) = TestFunctions(W)
    #ufl expression
    x = ufl.SpatialCoordinate(msh)
    a = (inner(grad(u), grad(v)) - p * div(v) + q * div(u)) * dx
    
    L = fem.Constant(msh, ScalarType(0)) * (q) * dx

    bcs = [bc_wall, bc_inflow, bc_circle, bc_outflow]
    problem = LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    wh = problem.solve()
    return wh, msh # u_ufl(x) #, #u_ufl(SpatialCoordinate(msh))

wh, msh = stokes_solver(16, msh)

#f = fem.Function(combinedSpace) 
p = wh.sub(1).collapse() 
p.name = "pressure" 
file1 = io.VTXWriter(msh.comm, "output1.bp", p, "BP4") 
file1.write(0.0)
file1.close() 

uh = wh.sub(0).collapse() 
uh.name = "velocity"
file2 = io.VTXWriter(msh.comm, "output2.bp", uh, engine="BP4") 
file2.write(0.0)
file2.close()


#uh, p = wh.split()
print("Task 2.1")

"""
with io.XDMFFile(msh.comm, "out_poisson/poisson.xdmf", "w") as file:
    #file.parameters["flush_output"] = True
    file.write_mesh(msh)
    file.write_function(uh)
    file.write_function(p)

time = 0
writer = io.VTXWriter(msh.comm,"out_poisson/poisson_vtk.pvd",[uh, p])
writer.write(time)
writer.close()
"""

# -

# and displayed using [pyvista](https://docs.pyvista.org/).

# +
# try:
#     import pyvista

#     cells, types, x = plot.vtk_mesh(V)
#     grid = pyvista.UnstructuredGrid(cells, types, x)
#     grid.point_data["u"] = uh.x.array.real
#     grid.set_active_scalars("u")
#     plotter = pyvista.Plotter()
#     plotter.add_mesh(grid, show_edges=True)
#     warped = grid.warp_by_scalar()
#     plotter.add_mesh(warped)
#     if pyvista.OFF_SCREEN:
#         pyvista.start_xvfb(wait=0.1)
#         plotter.screenshot("uh_poisson.png")
#     else:
#         plotter.show()
    
#     #error_L2 = errornorm(u_e, u, "L2")
#     #error_H1 = errornorm(u_e, u, "H1")

# except ModuleNotFoundError:
#     print("'pyvista' is required to visualise the solution")
#     print("Install 'pyvista' with pip: 'python3 -m pip install pyvista'")
# # -
