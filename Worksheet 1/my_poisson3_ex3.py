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

from basix.ufl import element, mixed_element
import ufl
from ufl import TrialFunction, TestFunction, TrialFunctions, TestFunctions
from ufl import inner, dot, grad, dx, ds


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
        msh, ct, facets = io.gmshio.read_from_msh("mesh.msh", MPI.COMM_WORLD, 0, gdim=2)
        #xdmf.write_mesh(msh)

def inflow_boundary(x):
    return np.isclose(x[0], 0)


def other_boundaries(x):
    return np.logical_or(np.logical_or(np.isclose(x[0], L), np.isclose(x[1], 0)), np.isclose(x[1], H))

def circle_boundary(x):
    return np.isclose((x[0] - c[0])**2 + (x[1]-c[1])**2 - r**2, 0)

def poisson_solver(N, msh, facets, degree=1):
    # create a rectangular mesh [0,1]x[0,1] with n subdivisions (ex. n=(16,16) means 16 elements in x-direction and 16 elements in y-direction) of triangles
    
    k = 1
    Q_el = element("Lagrange", msh.basix_cell(), k)
    P_el = element("Lagrange", msh.basix_cell(), k)
    V_el = mixed_element([Q_el, P_el])
    V = fem.functionspace(msh, V_el)

    #Inflow
    dofs_D = mesh.locate_entities_boundary(msh, 1, inflow_boundary)
    dofs_inflow = fem.locate_dofs_topological(V.sub(0), 1, dofs_D)
    bc_inflow = dirichletbc(value = ScalarType(1), dofs = dofs_inflow , V=V.sub(0))
    # dofs_inflow = fem.locate_dofs_topological(V.sub(1), 1, dofs_D)
    # bc_inflow_p = dirichletbc(value = ScalarType(0), dofs = dofs_inflow , V=V.sub(1))

    #Walls
    dofs_D = mesh.locate_entities_boundary(msh, 1, other_boundaries)
    dofs_else = fem.locate_dofs_topological(V.sub(0), 1, dofs_D)
    bc_0 = dirichletbc(value = ScalarType(0), dofs=dofs_else, V=V.sub(0))
    dofs_D = mesh.locate_entities_boundary(msh, 1, other_boundaries)
    dofs_else = fem.locate_dofs_topological(V.sub(1), 1, dofs_D)
    bc_1 = dirichletbc(value = ScalarType(1), dofs=dofs_else, V=V.sub(1))
    # dofs_else = fem.locate_dofs_topological(V.sub(1), 1, dofs_D)
    # bc_q = dirichletbc(value = ScalarType(0), dofs = dofs_else , V=V.sub(1))

    #Circle
    dofs_D = mesh.locate_entities_boundary(msh, 1, circle_boundary)
    dofs_circle = fem.locate_dofs_topological(V.sub(0), 1, dofs_D)
    bc_circle = dirichletbc(value = ScalarType(0), dofs=dofs_circle, V=V.sub(0))
    # dofs_circle = fem.locate_dofs_topological(V.sub(1), 1, dofs_D)
    # bc_circle_q = dirichletbc(value = ScalarType(0), dofs = dofs_circle , V=V.sub(1))
   
    (u,p) = TrialFunctions(V)
    (v,q) = TestFunctions(V)
    # u = TrialFunction(V)
    # v = TestFunction(V)
    # p = TrialFunction(V)
    # q = TestFunction(V)    #ufl expression
    x = ufl.SpatialCoordinate(msh)
    u_e1 = ufl.sin(ufl.pi * x[0])*ufl.sin(ufl.pi * x[1])
    f_1 = 2*ufl.pi**2*u_e1
    f_2 = 1
    a = (inner(grad(u), grad(v)) + inner(grad(p), grad(q)))  * dx
    L = (inner(f_1, v)+ inner(f_2, q)) * dx

    # We collect the variational problem and its boundary conditions
    # into a `LinearProblem <dolfinx.fem.petsc.LinearProblem>`
    # and we specify the solver
    #bcs = [bc, bc_inflow, bc_circle, bc_inflow_p, bc_q, bc_circle_q]
    problem = LinearProblem(a, L, bcs=[bc_0, bc_1], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve()
    return uh, msh, V # u_ufl(x) #, #u_ufl(SpatialCoordinate(msh))

wh, msh, V  = poisson_solver(16, msh)
sigma_h, uh = wh.split()

with io.XDMFFile(msh.comm, "out_poisson/poisson.xdmf", "w") as file:
    #file.parameters["flush_output"] = True
    file.write_mesh(msh)
    file.write_function(uh)
    file.write_function(sigma_h)

time = 0
writer = io.VTXWriter(msh.comm,"out_poisson/poisson_vtk.pvd",[uh, sigma_h])
writer.write(time)
writer.close()

