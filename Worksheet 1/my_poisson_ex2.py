# Note that it is important to first `from mpi4py import MPI` to
# ensure that MPI is correctly initialised.
from mpi4py import MPI
from petsc4py.PETSc import ScalarType  # type: ignore
import matplotlib.pyplot as plt

# +
import numpy as np

from dolfinx import fem, io, mesh, plot
from dolfinx.fem.petsc import LinearProblem
from dolfinx.fem import (Expression, Function, functionspace,
                         assemble_scalar, dirichletbc, form, locate_dofs_topological, locate_dofs_geometrical)

import ufl
from ufl import TrialFunction, TestFunction
from ufl import inner, dot, grad, dx, ds

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

# A = np.array([0,0])
# B = np.array([L,0])
# C = np.array([L,H])
# D = np.array([0, H])

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

mesh_from_file = meshio.read("mesh.msh")




def create_mesh(mesh, cell_type, prune_z=False):
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    points = mesh.points[:, :2] if prune_z else mesh.points
    out_mesh = meshio.Mesh(
        points=points, cells={cell_type: cells}, cell_data={"name_to_read": [cell_data]}
    )
    return out_mesh

line_mesh = create_mesh(mesh_from_file, "line", prune_z=True)
triangle_mesh = create_mesh(mesh_from_file, "triangle", prune_z=True)


# meshio.write("facet_mesh.xdmf", line_mesh)
# triangle_mesh = create_mesh(mesh_from_file, "triangle", prune_z=True)
meshio.write("mesh.vtk", triangle_mesh)
meshio.write("facet_mesh.vtk", line_mesh)

with io.XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "w") as xdmf:
        msh, ct, _ = io.gmshio.read_from_msh("mesh.msh", MPI.COMM_WORLD, 0, gdim=2)
        xdmf.write_mesh(msh)

def inflow_boundary(x):
    return np.isclose(x[0], 0)


def other_boundaries(x):
    return np.logical_or(np.logical_or(np.isclose(x[0], L), np.isclose(x[1], 0)), np.isclose(x[1], H))

def circle_boundary(x):
    return np.isclose((x[0] - c[0])**2 + (x[1]-c[1])**2 - r**2, 0)





def poisson_solver(N, msh, degree=1):

    #boundaries u =0 
    V = fem.functionspace(msh, ("Lagrange", degree))
    dofs_D = mesh.locate_entities_boundary(msh, 1, other_boundaries)
    #u_bc = np.array([0.0], dtype=ScalarType)
    bc = dirichletbc(value = ScalarType(0), dofs=fem.locate_dofs_topological(V, 1, dofs_D), V=V)
    dofs_D = mesh.locate_entities_boundary(msh, 1, circle_boundary)
    #u_bc = np.array([0.0], dtype=ScalarType)
    bc_circle = dirichletbc(value = ScalarType(0), dofs=fem.locate_dofs_topological(V, 1, dofs_D), V=V)

    #inflow u =1 
    dofs_D = mesh.locate_entities_boundary(msh, 1, inflow_boundary)
   # u_bc = np.array([1.0], dtype=ScalarType)
    bc_inflow = dirichletbc(value=ScalarType(1), dofs = fem.locate_dofs_topological(V, 1, dofs_D), V=V)



    # Next, the variational problem is defined:
    u = TrialFunction(V)
    v = TestFunction(V)
    #ufl expression
    x = ufl.SpatialCoordinate(msh)
    u_e = ufl.sin(ufl.pi * x[0])*ufl.sin(ufl.pi * x[1])
    f = 2*ufl.pi**2*u_e


    a = inner(grad(u), grad(v)) * dx
    L = inner(f, v) * dx

    # A {py:class}object is
    # created that brings together the variational problem, the Dirichlet
    # boundary condition, and which specifies the linear solver. In this
    # case an LU solver is used. The {py:func}`solve
    # <dolfinx.fem.petsc.LinearProblem.solve>` computes the solution.

    # We collect the variational problem and its boundary conditions
    # into a `LinearProblem <dolfinx.fem.petsc.LinearProblem>`
    # and we specify the solver
    problem = LinearProblem(a, L, bcs=[bc, bc_inflow, bc_circle], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve()
    return uh, msh, V # u_ufl(x) #, #u_ufl(SpatialCoordinate(msh))

uh, msh, V  = poisson_solver(16, msh)

print("Task 2.1")


with io.XDMFFile(msh.comm, "out_poisson/poisson.xdmf", "w") as file:
    #file.parameters["flush_output"] = True
    file.write_mesh(msh)
    file.write_function(uh)

time = 0
writer = io.VTXWriter(msh.comm,"out_poisson/poisson_vtk.pvd",[uh])
writer.write(time)
writer.close()

# -

# and displayed using [pyvista](https://docs.pyvista.org/).

# +
try:
    import pyvista

    cells, types, x = plot.vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(cells, types, x)
    grid.point_data["u"] = uh.x.array.real
    grid.set_active_scalars("u")
    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, show_edges=True)
    warped = grid.warp_by_scalar()
    plotter.add_mesh(warped)
    if pyvista.OFF_SCREEN:
        pyvista.start_xvfb(wait=0.1)
        plotter.screenshot("uh_poisson.png")
    else:
        plotter.show()
    
    #error_L2 = errornorm(u_e, u, "L2")
    #error_H1 = errornorm(u_e, u, "H1")

except ModuleNotFoundError:
    print("'pyvista' is required to visualise the solution")
    print("Install 'pyvista' with pip: 'python3 -m pip install pyvista'")
# -
