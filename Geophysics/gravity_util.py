import tensorflow as tf
import numpy as np
import copy
import matplotlib.pyplot as plt
from gempy.plot.visualization_2d_pro import *


def make_variables(k, initializer,dtype = tf.float32):
    '''
        create TensorFlow variable with shape k
        initializer: random variable initializer
    '''
    return tf.Variable(initializer(shape=[k], dtype=dtype))


def softmax_space(parameters):
    return tf.nn.softmax(parameters)


def constant32(x):
    return tf.constant(x, dtype=tf.float32)

def constant64(x):
    return tf.constant(x, dtype=tf.float64)

# def target_tz(radius, resolution):
#     if not isinstance(radius, list) or isinstance(radius, np.ndarray):
#         radius = np.repeat(radius, 3)
#     g_ = []
#     g_2 = []  # contains exp coord, left right xy, top and bottom

#     for xyz in [0, 1, 2]:

#         if xyz == 2:
#             # Make the grid only negative for the z axis

#             g_ = np.geomspace(0.01, 1, int(resolution[xyz]))
#             g_2.append((np.concatenate(([0], g_)) + 0.005) * -radius[xyz])
#         else:
#             g_ = np.geomspace(0.01, 1, int(resolution[xyz] / 2) + 1)
#             g_2.append(np.concatenate((-g_[::-1], g_)) * radius[xyz])
#     # my modification below, change the left/right boundary to grow exponentally instead of the center point

#     x_center = (g_2[0][:-1] + g_2[0][1:]) / 2
#     y_center = (g_2[1][:-1] + g_2[1][1:]) / 2
#     z_center = (g_2[-1][:-1] + g_2[-1][1:]) / 2
#     g = np.meshgrid(x_center, y_center, z_center)

#     d_left_x = np.abs(g_2[0][:-1] - x_center)
#     d_left_y = np.abs(g_2[1][:-1] - y_center)
#     d_right_x = np.abs(g_2[0][1:] - x_center)
#     d_right_y = np.abs(g_2[1][1:] - y_center)
#     d_z = z_center - g_2[-1][:-1]

#     d_left = np.meshgrid(d_left_x, d_left_y, d_z)
#     d_right = np.meshgrid(d_right_x, d_right_y, d_z)

#     kernel_g = np.vstack(tuple(map(np.ravel, g))).T.astype("float64")
#     kernel_d_left = np.vstack(tuple(map(np.ravel, d_left))).T.astype("float64")
#     kernel_d_right = np.vstack(tuple(map(np.ravel, d_right))).T.astype("float64")

#     grid_values = kernel_g

#     s_gr_x = grid_values[:, 0]
#     s_gr_y = grid_values[:, 1]
#     s_gr_z = grid_values[:, 2]

#     # getting the coordinates of the corners of the voxel...
#     x_cor = np.stack(
#         (s_gr_x - kernel_d_left[:, 0], s_gr_x + kernel_d_right[:, 0]), axis=1
#     )
#     y_cor = np.stack(
#         (s_gr_y - kernel_d_left[:, 1], s_gr_y + kernel_d_right[:, 1]), axis=1
#     )
#     z_cor = np.stack(
#         (s_gr_z - kernel_d_left[:, 2], s_gr_z + kernel_d_right[:, 2]), axis=1
#     )

#     # compute the target tz for the whole grid
#     UNI_grid_x = np.array([[np.min(x_cor), np.max(x_cor)]])
#     UNI_grid_y = np.array([[np.min(y_cor), np.max(y_cor)]])
#     UNI_grid_z = np.array([[np.max(z_cor), np.min(z_cor)]])

#     xUNI_matrix = np.repeat(UNI_grid_x, 4, axis=1)
#     yUNI_matrix = np.tile(np.repeat(UNI_grid_y, 2, axis=1), (1, 2))
#     zUNI_matrix = np.tile(UNI_grid_z, (1, 4))

#     sUNI_r = np.sqrt(xUNI_matrix ** 2 + yUNI_matrix ** 2 + zUNI_matrix ** 2)

#     mu = np.array([1, -1, -1, 1, -1, 1, 1, -1])
#     G = 6.674e-3

#     tz_UNI = G * np.sum(
#         -1
#         * mu
#         * (
#             xUNI_matrix * np.log(yUNI_matrix + sUNI_r)
#             + yUNI_matrix * np.log(xUNI_matrix + sUNI_r)
#             - zUNI_matrix
#             * np.arctan(xUNI_matrix * yUNI_matrix / (zUNI_matrix * sUNI_r))
#         ),
#         axis=1,
#     )
#     avg_tz = tz_UNI / kernel_g.shape[0]
#     # print(xUNI_matrix,yUNI_matrix,z_cor)
#     return avg_tz


# activate the customized grid
def activate_customized_grid(model,grid_kernel):
    model.geo_data.grid.custom_grid=grid_kernel
    model.geo_data.grid.deactivate_all_grids()
    # activate also rescaled the grid
    model.geo_data.grid.set_active('custom')

    model.geo_data.update_from_grid()
    model.geo_data.rescaling.set_rescaled_grid()
    model.grid = model.geo_data.grid
    model.from_gempy_interpolator()
    return model


def plot_centered_kernel(res, radius, centerkernel, direction="y", n_section=5):
    res_ = copy.deepcopy(res)
    if direction == "y":
        i = 0
    else:
        i = 1

    kernel_g, kernel_d_left, kernel_d_right = centerkernel.create_irregular_grid_kernel(
        resolution=res, radius=radius
    )
    print(kernel_g.shape)
    Slice = n_section
    # Trick just for plotting
    # the resolution is transposed in meshgrid
    if res_[0] % 2 == 0:
        res_[0] += 1
    if res_[1] % 2 == 0:
        res_[1] += 1
    # res_[2]+=1
    res_[0], res_[1] = res_[1], res_[0]
    # plotting the centers
    if i == 0:  # y direction
        plt.scatter(
            kernel_g[:, i].reshape(res_)[Slice, :, :],
            kernel_g[:, 2].reshape(res_)[Slice, :, :],
            c="b",
        )  # center
        plt.scatter(
            (kernel_g[:, i] - kernel_d_left[:, i]).reshape(res_)[Slice, :, :],
            kernel_g[:, 2].reshape(res_)[Slice, :, :],
            "r",
        )
    if i == 1:  # x direction
        plt.scatter(
            kernel_g[:, i].reshape(res_)[:, Slice, :],
            kernel_g[:, 2].reshape(res_)[:, Slice, :],
            s=19,
            c="b",
        )  # center
        left_x = (
            (kernel_g[:, i] - kernel_d_left[:, i]).reshape(res_)[:, Slice, :].flatten()
        )
        right_x = (
            (kernel_g[:, i] + kernel_d_right[:, i]).reshape(res_)[:, Slice, :].flatten()
        )
        top_z = (
            (kernel_g[:, 2] - kernel_d_left[:, 2]).reshape(res_)[:, Slice, :].flatten()
        )
        bot_z = (
            (kernel_g[:, 2] + kernel_d_left[:, 2]).reshape(res_)[:, Slice, :].flatten()
        )
        print(np.max(top_z))
        print(np.min(bot_z))
        plt.scatter(
            left_x,
            kernel_g[:, 2].reshape(res_)[:, Slice, :].flatten(),
            s=13,
            c="g",
            alpha=0.5,
        )  # left
        plt.scatter(
            right_x,
            kernel_g[:, 2].reshape(res_)[:, Slice, :].flatten(),
            s=13,
            c="r",
            alpha=0.5,
        )  # right
        plt.scatter(
            kernel_g[:, i].reshape(res_)[:, Slice, :], top_z, s=13, c="k", alpha=0.5
        )  # top
        plt.scatter(
            kernel_g[:, i].reshape(res_)[:, Slice, :], bot_z, s=13, c="y", alpha=0.5
        )  # bot

        plt.vlines(
            x=left_x[:: res[-1]],
            ymax=np.min(bot_z),
            ymin=np.max(top_z),
            linewidth=0.3,
            alpha=1,
            colors="black",
        )
        plt.vlines(
            x=right_x[-1],
            ymax=np.min(bot_z),
            ymin=np.max(top_z),
            linewidth=0.3,
            alpha=1,
            colors="black",
        )

    plt.hlines(
        y=bot_z[: res[-1]],
        xmax=np.min(left_x),
        xmin=np.max(right_x),
        linewidth=0.3,
        alpha=1,
        colors="black",
    )
    plt.hlines(
        y=top_z[0],
        xmax=np.min(left_x),
        xmin=np.max(right_x),
        linewidth=0.3,
        alpha=1,
        colors="black",
    )


def create_irregular_grid_kernel(resolution, radius, a, b, c):

    g_x = tf.cumsum(softmax_space(a))
    g_y = tf.cumsum(softmax_space(b))
    g_z = tf.cumsum(softmax_space(c))

    g2_x = tf.concat((-g_x[::-1], g_x), axis=0) * radius[0]
    g2_y = tf.concat((-g_y[::-1], g_y), axis=0) * radius[1]
    g2_z = (tf.concat(([0], g_z), axis=0) + 0.005) * -radius[2]

    x_center = (g2_x[:-1] + g2_x[1:]) / 2
    y_center = (g2_y[:-1] + g2_y[1:]) / 2
    z_center = (g2_z[:-1] + g2_z[1:]) / 2

    g = tf.meshgrid(x_center, y_center, z_center)

    d_left_x = tf.math.abs(g2_x[:-1] - x_center)
    d_left_y = tf.math.abs(g2_y[:-1] - y_center)
    d_right_x = tf.math.abs(g2_x[1:] - x_center)
    d_right_y = tf.math.abs(g2_y[1:] - y_center)
    d_z = z_center - g2_z[:-1]

    d_left = tf.meshgrid(d_left_x, d_left_y, d_z)
    d_right = tf.meshgrid(d_right_x, d_right_y, d_z)

    kernel_g = tf.concat(
        [
            tf.reshape(g[0], [-1, 1]),
            tf.reshape(g[1], [-1, 1]),
            tf.reshape(g[2], [-1, 1]),
        ],
        axis=1,
    )
    kernel_d_left = tf.concat(
        [
            tf.reshape(d_left[0], [-1, 1]),
            tf.reshape(d_left[1], [-1, 1]),
            tf.reshape(d_left[2], [-1, 1]),
        ],
        axis=1,
    )
    kernel_d_right = tf.concat(
        [
            tf.reshape(d_right[0], [-1, 1]),
            tf.reshape(d_right[1], [-1, 1]),
            tf.reshape(d_right[2], [-1, 1]),
        ],
        axis=1,
    )

    return kernel_g, kernel_d_left, kernel_d_right


def plot_grid(model, extent, radius, reg_res, center_res):
    model.activate_regular_grid()
    model.scalar_field(model.surface_points_coord)
    fig, ax = plt.subplots(3, 1, figsize=(30, 12))
    for ax_ in ax:
        ax_.set_xlim(extent[0], extent[1])
        # ax_.set_ylim(extent[-2], extent[-1] + 100)
        ax_.set_ylim(extent[-2], extent[-1] )

    ax[0].set_title("scalar field", fontsize=30)
    ax[1].set_title("regular grid", fontsize=30)
    ax[2].set_title("centered grid", fontsize=30)

    res = reg_res

    
    p = Plot2D(model)
    p.plot_scalar_field(ax[0], cell_number=reg_res[1] // 2)
    p.plot_contacts(ax[0], cell_number=reg_res[1] // 2)

    p.plot_lith(ax[1], cell_number=reg_res[1] // 2)
    p.plot_contacts(ax[1], cell_number=reg_res[1] // 2)
    
    p.plot_lith(ax[2], cell_number=reg_res[1] // 2)
    p.plot_contacts(ax[2], cell_number=reg_res[1] // 2)
    #########
    # regular grid

    # compute the upper coordinate of the grid
    dx, dy, dz = model.grid.regular_grid.get_dx_dy_dz()
    regular_center = model.grid.regular_grid.values[:,-1][:reg_res[-1]]
    h_lines = regular_center - dz/2
    
    v_lines = np.linspace(extent[0], extent[1], reg_res[0] + 1)

    for v in v_lines:
        ax[1].axvline(x=v, linewidth=0.5, c="w")
    
    for h in h_lines:
        ax[1].axhline(y=h, linewidth=0.5, c="w")

    ##########
    # centerred grid
    centered_grid = model.grid.centered_grid
    a, b, c = (
        centered_grid.kernel_centers,
        centered_grid.kernel_dxyz_left,
        centered_grid.kernel_dxyz_right,
    )

    center_res = copy.deepcopy(center_res)

    # dimension of x and y +1
    for index, integer in enumerate(center_res):
        if index < 2:
            center_res[index] += 1

    resolution = center_res

    res = np.array(resolution)

    slice = int(res[1] / 2)

    # plot the ground surface at 0
    ax[2].axhline(y=extent[-1], linewidth=4, color="k", alpha=0.7)
    ax[1].axhline(y=extent[-1], linewidth=4, color="k", alpha=0.7)

    # Centered grid left x coordinate 
    vlines = (
        a[:, 0].reshape(res)[slice, :, slice].ravel()
        - b[:, 0].reshape(res)[slice, :, slice].ravel()
    )

    center_receiver = extent[0]/2
    # Centered grid vertical lines
    for xc in vlines:
        ax[2].axvline(x=xc + extent[1]/2, ymin=0, ymax=1, linewidth=2, color="w", alpha=0.2)

    ## the right most vline
    ax[2].axvline(
        x=(a + b)[:, 0].reshape(res)[slice, -1, slice].ravel() + extent[1]/2,
        ymin=0,
        ymax=1,
        linewidth=2,
        color="w",
        alpha=0.2,
    )

    left_most = vlines[0]

    # compute the perportion of horizontal line from left coordinates
    left_most_vline = (extent[1]/2 + left_most) / extent[1]
    right_most_vline = (extent[1]/2 - left_most) / extent[1]

    for yc in (
        a[:, 2].reshape(res)[slice, slice, :].ravel()
        - b[:, 2].reshape(res)[slice, slice, :].ravel()
    ):
        ax[2].axhline(
            y=yc + extent[-1],
            xmin=left_most_vline,
            xmax=right_most_vline,
            linewidth=2,
            color="w",
            alpha=0.2,
        )


    ax[2].plot(extent[1]/2, extent[-1], marker=7, c="r", markersize=15, label="Receiver")


    # plt.savefig('/content/drive/MyDrive/YJ/threelayer_scalar_field.png',bbox_inches='tight',pad_inches = 0)
    plt.show()




