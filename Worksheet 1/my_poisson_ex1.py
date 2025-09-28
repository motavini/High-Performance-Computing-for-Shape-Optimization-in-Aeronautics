# Note that it is important to first `from mpi4py import MPI` to
# ensure that MPI is correctly initialised.
from mpi4py import MPI
from petsc4py.PETSc import ScalarType 
import matplotlib.pyplot as plt
import numpy as np

from dolfinx import fem, io, mesh, plot
from dolfinx.fem.petsc import LinearProblem
from dolfinx.fem import (Expression, Function, functionspace,
                         assemble_scalar, dirichletbc, form, locate_dofs_topological)

import ufl
from ufl import TrialFunction, TestFunction
from ufl import inner, dot, grad, dx, ds

from codetiming import Timer

def u_ex(mod):
    return lambda x: ufl.sin(ufl.pi * x[0])*ufl.sin(ufl.pi * x[1])

u_numpy = u_ex(np)
u_ufl = u_ex(ufl)

def error_L2(uh, u_ex, degree_raise=3):
    # Create higher order function space
    degree = uh.function_space.ufl_element().degree
    family = uh.function_space.ufl_element().family_name
    mesh = uh.function_space.mesh
    W = functionspace(mesh, (family, degree + degree_raise))
    # Interpolate approximate solution
    u_W = Function(W)
    u_W.interpolate(uh)

    # Interpolate exact solution, special handling if exact solution
    # is a ufl expression or a python lambda function
    u_ex_W = Function(W)
    if isinstance(u_ex, ufl.core.expr.Expr):
        u_expr = Expression(u_ex, W.element.interpolation_points())
        u_ex_W.interpolate(u_expr)
    else:
        u_ex_W.interpolate(u_ex)

    # Compute the error in the higher order function space
    e_W = Function(W)
    e_W.x.array[:] = u_W.x.array - u_ex_W.x.array

    # Integrate the error
    error = form(ufl.inner(e_W, e_W) * ufl.dx)
    error_local = assemble_scalar(error)
    error_global = mesh.comm.allreduce(error_local, op=MPI.SUM)
    return np.sqrt(error_global)

def poisson_solver(N, degree=1):
    # create a rectangular mesh [0,1]x[0,1] with n subdivisions (ex. n=(16,16) means 16 elements in x-direction and 16 elements in y-direction) of triangles
    msh = mesh.create_rectangle(
    comm=MPI.COMM_WORLD,
    points=((0.0, 0.0), (1.0, 1.0)),
    n=(N, N), # is this right
    cell_type=mesh.CellType.triangle,
    )
    V = fem.functionspace(msh, ("Lagrange", degree))

    facets = mesh.locate_entities_boundary(
    msh,
    dim=(msh.topology.dim - 1),
    marker=lambda x: np.logical_or(np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[0], 1.0)),np.logical_or(np.isclose(x[1], 0.0), np.isclose(x[1], 1.0))),
    )
    # We now find the degrees-of-freedom that are associated with the
    # boundary facets using <dolfinx.fem.locate_dofs_topological>
    dofs = fem.locate_dofs_topological(V=V, entity_dim=1, entities=facets)

    # And set Dirichlet boundary conditions on the boundary degrees of freedom with
    # Here, the first argument specifies the value to set on the boundary (in this case 0)
    # the second argument are the degrees of freedom and the third is the function space
    bc = fem.dirichletbc(value=ScalarType(0), dofs=dofs, V=V)

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
    problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve()
    return uh, msh, V, u_ufl(x) #, #u_ufl(SpatialCoordinate(msh))

print("Task 1")
#Task 2
print("-------------------------------")
print("Task 2")
uh, msh, V, u_ex = poisson_solver(16)
comm = uh.function_space.mesh.comm
error = form((uh - u_ex)**2 * ufl.dx) # uh vs u_ex
E = np.sqrt(comm.allreduce(assemble_scalar(error), MPI.SUM))
if comm.rank == 0:
   print(f"L2-error: {E:.2e}")

eh = uh - u_ex
error_H10 = form(dot(grad(eh), grad(eh)) * dx)
E_H10 = np.sqrt(comm.allreduce(assemble_scalar(error_H10), op=MPI.SUM))
if comm.rank == 0:
    print(f"H01-error: {E_H10:.2e}")


#Task 3 
print("-------------------------------")
print("Task 3")
Ns = [5, 10, 20, 40]
Es = np.zeros(len(Ns))
H0s = np.zeros(len(Ns))
hs = np.zeros(len(Ns), dtype=np.float64)

for i, N in enumerate(Ns):
    print(N)
    uh, msh, V, u_ex = poisson_solver(N)
    #error = (uh - u_ex)**2 * ufl.dx
    #comm = uh.function_space.mesh.comm
    comm = uh.function_space.mesh.comm
    eh = uh - u_ex
    error = form((eh)**2 * ufl.dx) # uh vs u_ex
    Es[i] = np.sqrt(comm.allreduce(assemble_scalar(error), MPI.SUM))
    
    error_H10 = form(dot(grad(eh), grad(eh)) * dx)
    H0s[i] = np.sqrt(comm.allreduce(assemble_scalar(error_H10), op=MPI.SUM))
    hs[i] = 1. / Ns[i] 

    if comm.rank == 0:
        print(f"h: {hs[i]:.2e} Error: {Es[i]:.2e}")
    #error = form((uh - u_ex)**2 * ufl.dx)
    #E = np.sqrt(comm.allreduce(assemble_scalar(error), MPI.SUM))

    #if comm.rank == 0:
    #    print(f"L2-error: {E:.2e}")

rates = np.log(Es[1:] / Es[:-1]) / np.log(hs[1:] / hs[:-1])
if comm.rank == 0:
    print(f"L2 Norm Rates: {rates}")

rates = np.log(H0s[1:] / H0s[:-1]) / np.log(hs[1:] / hs[:-1])
if comm.rank == 0:
    print(f"H0 Norm Rates: {rates}")


plt.loglog(hs, Es, label="L2 Norm")
plt.loglog(hs, H0s, label ="H0 Norm")
mL = 4
plt.loglog(hs, mL*hs, label=f"y={mL}x")
#print(mL*hs)
#print(hs)
mH = 0.25
plt.plot(hs, mH*hs, label=f"y={mH}x")
plt.legend()
plt.xlabel("1/N")
plt.ylabel("error")
plt.show()
print("-------------------------------")


#Task 4
print("Task 4")
uh4, msh4, V4, u_ex4 = poisson_solver(16, 2)
comm = uh4.function_space.mesh.comm
error = form((uh4 - u_ex4)**2 * ufl.dx) # uh vs u_ex
E = np.sqrt(comm.allreduce(assemble_scalar(error), MPI.SUM))
if comm.rank == 0:
   print(f"L2-error: {E:.2e}")

eh4 = uh4 - u_ex4
error_H10 = form(dot(grad(eh4), grad(eh4)) * dx)
E_H10 = np.sqrt(comm.allreduce(assemble_scalar(error_H10), op=MPI.SUM))
if comm.rank == 0:
    print(f"H01-error: {E_H10:.2e}")
print("-------------------------------")


#Task 5
print("Task 5")
Ts = np.zeros(len(Ns))
for i, N in enumerate(Ns):
    t0 = Timer("Solve Poisson")
    t0.start()
    uh, msh, V, u_ex = poisson_solver(N)
    elapsed_time = t0.stop()
    Ts[i] = elapsed_time

plt.title("Task 5 for degree = 1")
plt.plot(Ns, Ts)
plt.xlabel("N")
plt.ylabel("Computational Time (s)")
plt.show()
print("-------------------------------")


# The solution can be written to a XDMF File
# for visualization with ParaView or VisIt:

# +
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
