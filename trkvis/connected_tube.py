from __future__ import division

from vispy.visuals.mesh import MeshVisual
import numpy as np
from numpy.linalg import norm
from vispy.util.transforms import rotate
from vispy.color import ColorArray

import collections


class ConnectedTubeVisual(MeshVisual):
    """Displays a tube around a piecewise-linear path.

    The tube mesh is corrected following its Frenet curvature and
    torsion such that it varies smoothly along the curve, including if
    the tube is closed.

    MSC: Optimized constructor with Numpy bulk operations and add 'connect'
    parameter as per LineVisual

    Parameters
    ----------
    points : ndarray
        An array of (x, y, z) points describing the path along which the
        tube will be extruded.
    radius : float | ndarray
        The radius of the tube. Use array of floats as input to set radii of
        points individually. Defaults to 1.0.
    closed : bool
        Whether the tube should be closed, joining the last point to the
        first. Defaults to False.
    color : Color | ColorArray
        The color(s) to use when drawing the tube. The same color is
        applied to each vertex of the mesh surrounding each point of
        the line. If the input is a ColorArray, the argument will be
        cycled; for instance if 'red' is passed then the entire tube
        will be red, or if ['green', 'blue'] is passed then the points
        will alternate between these colours. Defaults to 'purple'.
    tube_points : int
        The number of points in the circle-approximating polygon of the
        tube's cross section. Defaults to 8.
    shading : str | None
        Same as for the `MeshVisual` class. Defaults to 'smooth'.
    vertex_colors: ndarray | None
        Same as for the `MeshVisual` class.
    face_colors: ndarray | None
        Same as for the `MeshVisual` class.
    mode : str
        Same as for the `MeshVisual` class. Defaults to 'triangles'.

    """

    def __init__(self, points, radius=1.0,
                 closed=False,
                 color='purple',
                 tube_points=8,
                 shading='smooth',
                 vertex_colors=None,
                 face_colors=None,
                 mode='triangles',
                 connect=None):

        # make sure we are working with floats
        points = np.array(points).astype(float) # [NPOINTS, 3]

        if connect is None:
            connect = np.ones(len(points), dtype=bool)
        else:
            connect = np.array(connect, dtype=bool)
        if len(connect) != len(points):
            raise ValueError("Connection array must be the same length as points")
        if not closed:
            connect[-1] = False
        tube_ends = np.where(np.logical_not(connect))[0]

        tangents, normals, binormals = _frenet_frames(points, closed)

        segments = len(points) - 1

        # if single radius, convert to list of radii
        if not isinstance(radius, collections.abc.Iterable):
            radius = [radius] * len(points)
        elif len(radius) != len(points):
            raise ValueError('Length of radii list must match points.')
        radius = np.array(radius, dtype=np.float32)

        # get the positions of each vertex
        #print("creating positions")
        #import time
        #print(time.perf_counter())
        vertices = np.arange(tube_points, dtype=np.float32) / tube_points * 2 * np.pi # [TUBE_POINTS]
        cxs = -1. * radius[..., np.newaxis] * np.cos(vertices) # [NPOINTS, TUBE_POINTS]
        cys = radius[..., np.newaxis] * np.sin(vertices) # [NPOINTS, TUBE_POINTS]
        grid = points[:, np.newaxis, :] + cxs[..., np.newaxis] * normals[:, np.newaxis, :] + cys[..., np.newaxis] * binormals[:, np.newaxis, :] # [NPOINTS, TUBE_POINTS, 3]
        vertices = grid.reshape(grid.shape[0]*grid.shape[1], 3)
        #print(time.perf_counter())
        #print("DONE creating positions")

        # construct the mesh
        #print("creating mesh")
        #print(time.perf_counter())
        indices2 = np.zeros((segments*tube_points*2, 3), dtype=np.uint32)
        indices2[0::2, 0] = np.arange(segments*tube_points, dtype=np.uint32)
        indices2[1::2, 0] = np.arange(segments*tube_points, dtype=np.uint32) + tube_points
        indices2[0::2, 1] = np.arange(segments*tube_points, dtype=np.uint32) + tube_points
        indices2[1::2, 1] = np.arange(segments*tube_points, dtype=np.uint32) + tube_points + 1
        indices2[0::2, 2] = np.arange(segments*tube_points, dtype=np.uint32) + 1
        indices2[1::2, 2] = np.arange(segments*tube_points, dtype=np.uint32) + 1
        indices2[tube_points*2-2::2*tube_points, 2] -= tube_points 
        indices2[tube_points*2-1::2*tube_points, 2] -= tube_points
        indices2[tube_points*2-1::2*tube_points, 1] -= tube_points

        tube_start1, tube_start2 = 0, 0
        indices = np.zeros((len(indices2) - (len(tube_ends)-1)*2*tube_points, 3), dtype=np.uint32)
        for idx in range(len(tube_ends)):
            tube_end2 = tube_ends[idx]
            tube_end1 = tube_end2 - idx
            indices[tube_start1*tube_points*2:tube_end1*tube_points*2, :] = indices2[tube_start2*tube_points*2:tube_end2*tube_points*2, :]
            tube_start1 = tube_end1
            tube_start2 = tube_end2 + 1

        #print(time.perf_counter())
        #print("DONE creating mesh")

        color = ColorArray(color)
        if vertex_colors is None:
            point_colors = np.resize(color.rgba,
                                     (len(points), 4))
            vertex_colors = np.repeat(point_colors, tube_points, axis=0)

        MeshVisual.__init__(self, vertices, indices,
                            vertex_colors=vertex_colors,
                            face_colors=face_colors,
                            shading=shading,
                            mode=mode)

def _batch_rotate(angle, axis, dtype=None):
    """3x3 rotation matrices for rotation about a batch of angles/vectors.

    Parameters
    ----------
    angle : ndarray [NBATCH]
        The angles of rotation, in radians.
    axis : ndarray [NBATCH, 3]
        The x, y, z coordinates of the axis direction UNIT vector.

    Returns
    -------
    M : ndarray [NBATCH, 3, 3]
        Transformation matrix describing the rotation.
    """
    nbatch = angle.shape[0]
    assert angle.shape[0] == axis.shape[0]
    assert axis.shape[1] == 3
    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2] # [NBATCH]
    c, s = np.cos(angle), np.sin(angle) # [NBATCH]
    cx, cy, cz = (1 - c) * x, (1 - c) * y, (1 - c) * z
    rots = np.zeros((nbatch, 3, 3), dtype=np.float32)
    rots[:, 0, 0] = cx * x + c
    rots[:, 1, 0] = cy * x - z * s
    rots[:, 2, 0] = cz * x + y * s
    rots[:, 0, 1] = cx * y + z * s
    rots[:, 1, 1] = cy * y + c
    rots[:, 2, 1] = cz * y - x * s
    rots[:, 0, 2] = cx * z - y * s
    rots[:, 1, 2] = cy * z + x * s
    rots[:, 2, 2] = cz * z + c
    return rots

def _frenet_frames(points, closed):
    """Calculates and returns the tangents, normals and binormals for
    the tube.
    """
    tangents = np.zeros((len(points), 3))
    normals = np.zeros((len(points), 3))
    normals2 = np.zeros((len(points), 3))

    epsilon = 0.0001

    # Compute tangent vectors for each segment
    tangents = np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)
    if not closed:
        tangents[0] = points[1] - points[0]
        tangents[-1] = points[-1] - points[-2]
    mags = np.sqrt(np.sum(tangents * tangents, axis=1))
    tangents /= mags[:, np.newaxis]

    # Get initial normal and binormal
    t = np.abs(tangents[0])

    smallest = np.argmin(t)
    normal = np.zeros(3)
    normal[smallest] = 1.

    vec = np.cross(tangents[0], normal)
    normals[0] = np.cross(tangents[0], vec)

    # Compute normal and binormal vectors along the path
    #print("compute normal binormal")
    #import time
    #print(time.perf_counter())
    vecs = np.cross(np.roll(tangents, 1, axis=0), tangents)
    thetas_rad = -np.arccos(np.clip(np.einsum('ij,ij->i', np.roll(tangents, 1, axis=0), tangents), -1, 1))
    vec_norms = norm(vecs, axis=-1)
    vecs = vecs / vec_norms[..., np.newaxis]
    rots = _batch_rotate(thetas_rad, vecs)
   
    for i in range(1, len(points)):
        if vec_norms[i] > epsilon:
            normals[i] = rots[i].dot(normals[i-1])
        else:
            normals[i] = normals[i-1]
    #print(time.perf_counter())

    if closed:
        theta = np.arccos(np.clip(normals[0].dot(normals[-1]), -1, 1))
        theta /= len(points) - 1

        if tangents[0].dot(np.cross(normals[0], normals[-1])) > 0:
            theta *= -1.

        for i in range(1, len(points)):
            normals[i] = rotate(-np.degrees(theta*i),
                                tangents[i])[:3, :3].dot(normals[i])

    binormals = np.cross(tangents, normals)

    return tangents, normals, binormals
