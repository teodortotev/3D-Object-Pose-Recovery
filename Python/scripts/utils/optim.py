import math
import numpy as np
from scipy import optimize

def compute_error(v, x2d, x3d):

    a = v[0]
    e = v[1]
    d = v[2]
    f = v[3]
    principal_point = np.array([v[4], v[5]])
    theta = v[6]

    # Camera center
    C = np.zeros(3)
    C[0] = d*math.cos(e)*math.sin(a)
    C[1] = -d*math.cos(e)*math.cos(a)
    C[2] = d*math.sin(e)

    a = -a
    e = -(math.pi/2-e)

    # Rotation matrix
    Rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])   # rotate by a
    Rx = np.array([[1, 0, 0], [0, math.cos(e), -math.sin(e)], [0, math.sin(e), math.cos(e)]])   # rotate by e
    R = Rx.dot(Rz)

    # Perspective project matrix
    M = 3000
    P = np.array([[M*f, 0, 0], [0, M*f, 0], [0, 0, -1]]).dot(np.c_[R, -R.dot(C)])

    # Project
    x = P.dot(np.r_[x3d, [np.ones(x3d.shape[1])]])
    x[0, :] = x[0, :]/x[2, :]
    x[1, :] = x[1, :]/x[2, :]
    x[2, :] = x[2, :]/x[2, :]
    x = x[0:2, :]

    # Rotation matrix 2D
    R2d = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    x = R2d.dot(x)

    # Transform to image coordinates
    x[:, 1] = -x[:, 1]
    x[0, :] = x[0, :] + principal_point[0]
    x[1, :] = x[1, :] + principal_point[1]

    # Compute error
    error = 0
    for i in range(x2d.shape[1]):
        error = error + np.linalg.norm(x[:, i] - x2d[:, i])

    error = error/x2d.shape[1]

    return error

def optim(x2d, c2d, x3d, c3d, size):

    # # Filter out the masks
    # idcs = np.argsort(-c2d)[:4]
    # x2d = x2d[:, idcs]
    # x3d = x3d[:, idcs]

    # Initialize viewpoint with best possible guesses
    vp = np.zeros(7)
    lb = np.zeros(7)
    ub = np.zeros(7)
    # Azimuth
    # azimuth = math.pi # in radians
    azimuth = 5.85
    vp[0] = azimuth
    azimuth_bound = np.array([0, 2*math.pi])
    lb[0] = azimuth_bound[0]
    ub[0] = azimuth_bound[1]
    # Elevation
    # elevation = 0 # in radians
    elevation = -0.03
    vp[1] = elevation
    elevation_bound = np.array([-math.pi/2, math.pi/2])
    lb[1] = elevation_bound[0]
    ub[1] = elevation_bound[1]
    # Distance
    # distance = 50
    distance = 5.0525
    vp[2] = distance
    distance_bound = np.array([0, 100])
    lb[2] = distance_bound[0]
    ub[2] = distance_bound[1]
    # Focal length
    focal_length = 1
    vp[3] = focal_length
    focal_length_bound = np.array([1, 1])
    lb[3] = focal_length_bound[0]
    ub[3] = focal_length_bound[1]
    # Px
    px = size[0]/2
    vp[4] = px
    px_bound = np.array([0, size[0]])
    lb[4] = px_bound[0]
    ub[4] = px_bound[1]
    # Py
    py = size[1]/2
    vp[5] = py
    py_bound = np.array([0, size[1]])
    lb[5] = py_bound[0]
    ub[5] = py_bound[1]
    # Inplane rot
    # inplane_rot = 0
    inplane_rot = 0.1807
    vp[6] = inplane_rot
    inplane_rot_bound = np.array([-math.pi, math.pi])
    lb[6] = inplane_rot_bound[0]
    ub[6] = inplane_rot_bound[1]

    bnds = ((lb[0], ub[0]), (lb[1], ub[1]), (lb[2], ub[2]), (lb[3], ub[3]), (lb[4], ub[4]), (lb[5], ub[5]), (lb[6], ub[6]))
    

    res = optimize.minimize(fun=compute_error, x0=vp, args= (x2d, x3d), bounds=bnds)

    print(res)
    exit()
    

