import gmsh
import numpy as np
from dolfinx.io.gmshio import model_to_mesh
from mpi4py import MPI
from dolfinx import io


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


def flow_field_mesh(surface_file, angle_of_attack, mesh_size_min=None, mesh_size_max=None, bounding_box=(-150, -150, -150, 300, 300, 300), file_output_name = "Boeing_mesh", model_num = 1):
    """
    Generates a 3D flow field mesh around an airplane surface.

    :param surface_file: Path to the airplane surface file (STL format).
    :param angle_of_attack: Angle of attack in degrees.
    :param mesh_size_min: Minimum mesh size (optional).
    :param mesh_size_max: Maximum mesh size (optional).
    :param bounding_box: Bounding box dimensions as (xmin, ymin, zmin, dx, dy, dz).
    :param file_output_name: Name of the output mesh file (without extension).
    :param model_num: Integer representing the mesh model number.
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
    gmsh.write(f"outputs/boeing_{model_num}_msh/{file_output_name}.msh")
    gmsh.write(f"outputs/boeing_{model_num}_vtk/volume_mesh_{file_output_name}.vtk")
    gmsh.finalize()

    return mesh, facet_tags


if __name__ == "__main__":
    """
    Main script to generate flow field meshes for a range of angles of attack.
    .msh files can be generated before solving on DCE cluster with parallel computation.
    """
    surface_file = "boeing_767/76682_Boeing_767_hopeful.stl" 
    bounding_box=(-50, -50, -50, 100, 100, 100)
    model_num = 4 # integer from 1 to 4, where 1 represents the coarsest mesh
    angles = np.arange(-60, 61, 20) # positive angle points the plane up 
    for angle in angles:
        flow_field_mesh(surface_file, angle,  bounding_box=bounding_box, file_output_name=f"boeing_{model_num}_{angle}", model_num = model_num)
    