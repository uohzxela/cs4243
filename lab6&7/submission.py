import math
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

pts = np.zeros([11, 3])
pts[0, :] = [-1, -1, -1]
pts[1, :] = [1, -1, -1]
pts[2, :] = [1 , 1, -1]
pts[3, :] = [-1, 1, -1]
pts[4, :] = [-1, -1, 1]
pts[5, :] = [1, -1, 1]
pts[6, :] = [1, 1, 1]
pts[7, :] = [-1, 1, 1]
pts[8, :] = [-0.5, -0.5, -1]
pts[9, :] = [0.5, -0.5, -1]
pts[10, :] = [0, 0.5, -1]

# part 1.1

def quatmult(q1, q2):
    out = [0, 0, 0, 0]
    out[0] = q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3]
    out[1] = q1[0]*q2[1] + q1[1]*q2[0] + q1[2]*q2[3] - q1[3]*q2[2]
    out[2] = q1[0]*q2[2] - q1[1]*q2[3] + q1[2]*q2[0] + q1[3]*q2[1]
    out[3] = q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1] + q1[3]*q2[0]
    return out

def point_rotation_by_quaternion(point,q):
    r = [0]+point
    q_conj = [q[0],-q[1],-q[2],-q[3]]
    return quatmult(quatmult(q,r),q_conj)[1:]

def get_rotation_quaternion(deg, axis_of_rotation):
    rad = math.radians(deg)
    w_x, w_y, w_z = axis_of_rotation
    return [math.cos(rad/2), math.sin(rad/2)*w_x, math.sin(rad/2)*w_y, math.sin(rad/2)*w_z]

cam_loc_1 = [0., 0., -5.]
cam_loc_2 = point_rotation_by_quaternion(cam_loc_1, get_rotation_quaternion(-30, [0,1,0]))
cam_loc_3 = point_rotation_by_quaternion(cam_loc_1, get_rotation_quaternion(-60, [0,1,0]))
cam_loc_4 = point_rotation_by_quaternion(cam_loc_1, get_rotation_quaternion(-90, [0,1,0]))

# print "Camera position at frame 1:", cam_loc_1
# print "Camera position at frame 2:", cam_loc_2
# print "Camera position at frame 3:", cam_loc_3
# print "Camera position at frame 4:", cam_loc_4

# part 1.2

def quat2rot(q):
    out = [[0 for j in xrange(3)] for i in xrange(3)]

    out[0][0] = q[0]**2 + q[1]**2 - q[2]**2 - q[3]**2
    out[0][1] = 2 * (q[1]*q[2] - q[0]*q[3])
    out[0][2] = 2 * (q[1]*q[3] + q[0]*q[2])

    out[1][0] = 2 * (q[1]*q[2] + q[0]*q[3])
    out[1][1] = q[0]**2 + q[2]**2 - q[1]**2 - q[3]**2
    out[1][2] = 2 * (q[2]*q[3] - q[0]*q[1])

    out[2][0] = 2*(q[1]*q[3] - q[0]*q[2])
    out[2][1] = 2*(q[2]*q[3] + q[0]*q[1])
    out[2][2] = q[0]**2 + q[3]**2 - q[1]**2 - q[2]**2

    return out

quatmat_1 = [[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]
quatmat_2 = quat2rot(get_rotation_quaternion(30, [0,1,0]))
quatmat_3 = quat2rot(get_rotation_quaternion(60, [0,1,0]))
quatmat_4 = quat2rot(get_rotation_quaternion(90, [0,1,0]))

# print "Rotation matrix 1:", quatmat_1
# print "Rotation matrix 2:", quatmat_2
# print "Rotation matrix 3:", quatmat_3
# print "Rotation matrix 4:", quatmat_4

# part 2

def perspective_projection(s, t, r):
    cam_horizontal_axis, cam_vertical_axis, cam_optical_axis = r
    u = np.dot((s - t), cam_horizontal_axis) / np.dot((s - t), cam_optical_axis)
    v = np.dot((s - t), cam_vertical_axis) / np.dot((s - t), cam_optical_axis)
    return u, v

def orthographic_projection(s, t, r):
    cam_horizontal_axis, cam_vertical_axis, cam_optical_axis = r
    u = np.dot((s - t), cam_horizontal_axis)
    v = np.dot((s - t), cam_vertical_axis)
    return u, v

FRAME_1 = 0
FRAME_2 = 1
FRAME_3 = 2
FRAME_4 = 3

perspective_projections = [[], [], [], []]
orthographic_projections = [[], [], [], []]

for pt in pts:
    perspective_projections[FRAME_1].append(
        perspective_projection(pt, cam_loc_1, quatmat_1))
    orthographic_projections[FRAME_1].append(
        orthographic_projection(pt, cam_loc_1, quatmat_1))

    perspective_projections[FRAME_2].append(
        perspective_projection(pt, cam_loc_2, quatmat_2))
    orthographic_projections[FRAME_2].append(
        orthographic_projection(pt, cam_loc_2, quatmat_2))

    perspective_projections[FRAME_3].append(
        perspective_projection(pt, cam_loc_3, quatmat_3))
    orthographic_projections[FRAME_3].append(
        orthographic_projection(pt, cam_loc_3, quatmat_3))

    perspective_projections[FRAME_4].append(
        perspective_projection(pt, cam_loc_4, quatmat_4))
    orthographic_projections[FRAME_4].append(
        orthographic_projection(pt, cam_loc_4, quatmat_4))


def plot_image_points(ax, pts, title):
    x, y = zip(*pts)
    ax.scatter(x, y, color="red")
    ax.set_title("Frame " + str(title+1), fontsize=11)

fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
plot_image_points(ax1, perspective_projections[FRAME_1], title=FRAME_1)
plot_image_points(ax2, perspective_projections[FRAME_2], title=FRAME_2)
plot_image_points(ax3, perspective_projections[FRAME_3], title=FRAME_3)
plot_image_points(ax4, perspective_projections[FRAME_4], title=FRAME_4)
fig1.canvas.set_window_title('Perspective projection')
fig1.savefig('perspective')

fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
plot_image_points(ax1, orthographic_projections[FRAME_1], title=FRAME_1)
plot_image_points(ax2, orthographic_projections[FRAME_2], title=FRAME_2)
plot_image_points(ax3, orthographic_projections[FRAME_3], title=FRAME_3)
plot_image_points(ax4, orthographic_projections[FRAME_4], title=FRAME_4)
fig2.canvas.set_window_title('Orthographic projection')
fig2.savefig('orthographic')

# part 3

point_correspondences =[
(perspective_projections[FRAME_3][0], pts[0]), 
(perspective_projections[FRAME_3][1], pts[1]), 
(perspective_projections[FRAME_3][2], pts[2]), 
(perspective_projections[FRAME_3][3], pts[3]), 
(perspective_projections[FRAME_3][8], pts[8])
]

A = []

for img_pt, euclidean_pt in point_correspondences:
    u, v = img_pt
    x, y, z = euclidean_pt
    A.append([-x, -y, -z, 0, 0, 0, u*x, u*y, u*z])
    A.append([0, 0, 0, -x, -y, -z, v*x, v*y, v*z])

U, S, VT = la.svd(A)
h = VT[-1]

# normalizing H matrix
for i in xrange(len(h)):
    h[i] = h[i] / h[-1]

H = [
[h[0], h[1], h[2]],
[h[3], h[4], h[5]],
[h[6], h[7], h[8]]
]

H = np.matrix(H)
print "H:"
print H
