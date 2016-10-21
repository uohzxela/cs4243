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

# quaternion multiplication
def quatmult(q1, q2):
    out = [0, 0, 0, 0]
    out[0] = q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3]
    out[1] = q1[0]*q2[1] + q1[1]*q2[0] + q1[2]*q2[3] - q1[3]*q2[2]
    out[2] = q1[0]*q2[2] - q1[1]*q2[3] + q1[2]*q2[0] + q1[3]*q2[1]
    out[3] = q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1] + q1[3]*q2[0]
    return out


# rotate a point using quaternion multiplication
def point_rotation_by_quaternion(point,q):
    r = [0]+point
    q_conj = [q[0],-q[1],-q[2],-q[3]]
    return quatmult(quatmult(q,r),q_conj)[1:]


# represent a rotation as a quaternion
def get_rotation_quaternion(deg, axis_of_rotation):
    rad = math.radians(deg)
    w_x, w_y, w_z = axis_of_rotation
    return [math.cos(rad/2), 
            math.sin(rad/2)*w_x, 
            math.sin(rad/2)*w_y, 
            math.sin(rad/2)*w_z]


cam_loc_1 = [0., 0., -5.]
cam_loc_2 = point_rotation_by_quaternion(cam_loc_1, get_rotation_quaternion(-30, [0,1,0]))
cam_loc_3 = point_rotation_by_quaternion(cam_loc_1, get_rotation_quaternion(-60, [0,1,0]))
cam_loc_4 = point_rotation_by_quaternion(cam_loc_1, get_rotation_quaternion(-90, [0,1,0]))


# part 1.2

# return a 3x3 rotation matrix parameterized 
# by the elements of a given input quaternion.
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


def plot_image_points(ax, pts, title):
    x, y = zip(*pts)
    ax.scatter(x, y, color="red")
    ax.set_title("Frame " + str(title+1), fontsize=11)
    for i in xrange(len(pts)):
        ax.annotate(i, (x[i],y[i]))


def plot(projections):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
    plot_image_points(ax1, projections[FRAME_1], title=FRAME_1)
    plot_image_points(ax2, projections[FRAME_2], title=FRAME_2)
    plot_image_points(ax3, projections[FRAME_3], title=FRAME_3)
    plot_image_points(ax4, projections[FRAME_4], title=FRAME_4)
    return fig


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

fig1 = plot(perspective_projections)
fig1.savefig('perspective_projections')
print "Saved perspective projections to perspective_projections.png"

fig2 = plot(orthographic_projections)
fig2.savefig('orthographic_projections')
print "Saved orthographic projections to orthographic_projections.png"


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

# take the row vector of VT as homography values
# which corresponds to the index of the smallest sigma value
homography_values = VT[np.argmin(S)]

# normalize homography values
homography_values = [v / homography_values[-1] for v in homography_values]

# reshape the values into 3x3 homography matrix
H = np.reshape(homography_values, (-1, 3))

f = open('homography_matrix.txt', 'w')
print >> f, 'Homography matrix:'
print >> f, H
f.close()
print "Saved homography matrix to homography_matrix.txt"
