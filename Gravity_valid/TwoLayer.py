# %%
import tensorflow as tf

import os
import numpy as np
import sys
import timeit
import csv

sys.path.append("/Users/zhouji/Documents/github/YJ/GP_old")

sys.path.append("/Users/zhouji/Documents/github/YJ/")
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import json
import gempy as gp
from gempy.core.tensor.tensorflow_graph_test import TFGraph
import tensorflow as tf
import tensorflow_probability as tfp
import pandas as pd
from gempy.core.solution import Solution
from gempy import create_data, map_series_to_surfaces
from gempy.assets.geophysics import GravityPreprocessing
from gempy.core.grid_modules.grid_types import RegularGrid


tfd = tfp.distributions


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def label(ax, string):
    ax.annotate(
        string,
        (1, 1),
        xytext=(-8, -8),
        ha="right",
        va="top",
        size=14,
        xycoords="axes fraction",
        textcoords="offset points",
        fontsize=18,
        fontweight="bold",
    )


class ModelTwo(object):
    def __init__(
        self,
        master_path,
        surface_path,
        orientations_path,
        center_grid_resolution=[50, 50, 50],
        regular_grid_resolution=[100, 100, 30],
        name=None,
        receivers=None,
        dtype="float32",
        radius=[900, 900, 1000],
        extent = [0, 10000, 0, 10000, 0, 1000],
        top = [5000,5000,995],
        bot = [6000,5000,0]
    ):

        if dtype == "float32":
            self.tfdtype = tf.float32
        elif dtype == "float64":
            self.tfdtype = tf.float64

        self.dtype = dtype

        self.path = master_path

        if name is None:
            self.modelName = "model"
        else:
            self.modelName = str(name)

        self.regular_grid_resolution = regular_grid_resolution
        # the regular cell is 200*200*200
        self.extent = extent
        top_x = top[0]
        top_y = 5000
        top_z = top[-1]

        bot_x = bot[0]
        bot_y = 5000
        bot_z = bot[-1]


        # find the 2d othorganal vector
        def PerpendicularConterClockwise(x,z):
            n = np.linalg.norm(np.array([x,z]))
            x/= n
            z/= n
            return -z,x


        orien_x = (top_x+bot_x)/2
        orien_y = (top_y+bot_y)/2
        orien_z = (top_z+bot_z)/2

        self.geo_data = gp.create_model('Model1')
        self.geo_data = gp.init_data(self.geo_data, extent=[0, 10000, 0, 10000, 0, 1000], resolution=regular_grid_resolution)
        self.geo_data.set_default_surfaces()
        self.geo_data.add_surface_points(X=top_x, Y=top_y, Z=top_z, surface='surface1')
        self.geo_data.add_surface_points(X=bot_x, Y=bot_y, Z=bot_z, surface='surface1')
        vec = PerpendicularConterClockwise((bot_x-top_x),(bot_z - top_z))
        self.geo_data.add_orientations(X=orien_x, Y=orien_y, Z=orien_z, surface='surface1', pole_vector=(vec[0], 0, vec[1]))

        map_series_to_surfaces(
            self.geo_data,
            {"Strat_Series": ("surface1"), "Basement_Series": ("surface2")},
        )

        # define density
        ## the order is following the indexing not id in geo_data.surfaces
        self.geo_data.add_surface_values(
            [2.5, 3.5]
        )  # density 2*10^3 kg/m3 (sandstone,igeous rock,salt )

        # define indexes where we set dips position as surface position
        self.sf = self.geo_data.surface_points.df.loc[:, ["X", "Y", "Z"]]

        ## customize irregular receiver set-up, XY pairs are required
        # append a Z value 1000 to the array
        self.top = 1000
        self.Zs = np.expand_dims(self.top * np.ones([receivers.shape[0]]), axis=1)
        self.xy_ravel = np.concatenate([receivers, self.Zs], axis=1)

        self.center_grid_resolution = center_grid_resolution
        self.radius = radius
        self.geo_data.set_centered_grid(
            self.xy_ravel, resolution=self.center_grid_resolution, radius=self.radius
        )

        # self.activate_centered_grid()
        self.from_gempy_interpolator()

        mu_true = self.geo_data.surface_points.df.loc[:, "Z"].to_numpy()
        self.mu_true = tf.convert_to_tensor(mu_true, self.tfdtype)


        self.mu_prior = self.mu_true
        self.sigma = 500
        self.mu_prior_r = (self.mu_prior - self.centers[2]) / self.rf + 0.5001
        self.cov_prior = ((self.sigma)) ** 2 * tf.eye(
            self.Number_surface_points, dtype=self.tfdtype
        )
        self.cov_prior_r = ((self.sigma) / self.rf) ** 2 * tf.eye(
            self.Number_surface_points, dtype=self.tfdtype
        )  # $$# double check here

    def from_gempy_interpolator(self):
        self.interpolator = self.geo_data.interpolator

        # extract data from gempy interpolator
        (
            dips_position,
            dip_angles,
            azimuth,
            polarity,
            surface_points_coord,
            fault_drift,
            grid,
            values_properties,
        ) = self.interpolator.get_python_input_block()[0:-3]

        dip_angles = tf.cast(dip_angles, self.tfdtype)
        _grid = grid
        grid = tf.cast(grid, self.tfdtype)
        self.dips_position = tf.cast(dips_position, self.tfdtype)
        azimuth = tf.cast(azimuth, self.tfdtype)
        polarity = tf.cast(polarity, self.tfdtype)
        fault_drift = tf.cast(fault_drift, self.tfdtype)
        values_properties = tf.cast(values_properties, self.tfdtype)
        self.Number_surface_points = int(surface_points_coord.shape[0])

        self.surface_points = self.geo_data.surface_points
        self.surfaces = self.geo_data.surfaces
        self.orientations = self.geo_data.orientations

        self.centers = self.geo_data.rescaling.df.loc["values", "centers"].astype(
            self.dtype
        )

        g = GravityPreprocessing(self.geo_data.grid.centered_grid)

        # precomputed gravity impact from each grid
        tz = g.set_tz_kernel()

        self.tz = tf.cast(tz, self.tfdtype)

        len_rest_form = (
            self.interpolator.additional_data.structure_data.df.loc[
                "values", "len surfaces surface_points"
            ]
            - 1
        )
        Range = self.interpolator.additional_data.kriging_data.df.loc["values", "range"]
        C_o = self.interpolator.additional_data.kriging_data.df.loc["values", "$C_o$"]
        rescale_factor = self.interpolator.additional_data.rescaling_data.df.loc[
            "values", "rescaling factor"
        ]

        self.rf = rescale_factor
        nugget_effect_grad = np.cast[self.dtype](
            np.tile(self.interpolator.orientations.df["smooth"], 3)
        )
        nugget_effect_scalar = np.cast[self.dtype](
            self.interpolator.surface_points.df["smooth"]
        )

        # surface_points_coord = tf.Variable(surface_points_coord, dtype=self.tfdtype)

        self.dip_angles = tf.convert_to_tensor(dip_angles, dtype=self.tfdtype)
        self.azimuth = tf.convert_to_tensor(azimuth, dtype=self.tfdtype)
        self.polarity = tf.convert_to_tensor(polarity, dtype=self.tfdtype)

        self.fault_drift = tf.convert_to_tensor(fault_drift, dtype=self.tfdtype)
        self.grid_tensor = tf.convert_to_tensor(grid, dtype=self.tfdtype)
        self.values_properties = tf.convert_to_tensor(
            values_properties, dtype=self.tfdtype
        )
        self.len_rest_form = tf.convert_to_tensor(len_rest_form, dtype=self.tfdtype)
        self.Range = tf.convert_to_tensor(Range, self.tfdtype)
        self.C_o = tf.convert_to_tensor(C_o, dtype=self.tfdtype)
        self.nugget_effect_grad = tf.convert_to_tensor(
            nugget_effect_grad, dtype=self.tfdtype
        )
        self.nugget_effect_scalar = tf.convert_to_tensor(
            nugget_effect_scalar, dtype=self.tfdtype
        )
        self.rescale_factor = tf.convert_to_tensor(rescale_factor, self.tfdtype)

        self.surface_points_coord = tf.convert_to_tensor(
            surface_points_coord, self.tfdtype
        )

        self.solutions = Solution(_grid, self.geo_data.surfaces, self.geo_data.series)

        self.lg_0 = self.interpolator.grid.get_grid_args("centered")[0]
        self.lg_1 = self.interpolator.grid.get_grid_args("centered")[1]

    def activate_centered_grid(
        self,
    ):
        self.geo_data.grid.deactivate_all_grids()
        self.geo_data.grid.set_active(["centered"])
        self.geo_data.update_from_grid()
        self.grid = self.geo_data.grid
        self.from_gempy_interpolator()

    def activate_regular_grid(
        self,
    ):
        self.geo_data.grid.deactivate_all_grids()
        self.geo_data.grid.set_active(["regular"])
        self.geo_data.update_from_grid()
        self.grid = self.geo_data.grid
        self.from_gempy_interpolator()

    def plot_model(self, plot_gravity=False, save=False):

        # use gempy build-in method to plot surface points and orientation points
        geomode = gp.plot.plot_data(self.geo_data, direction="z")

        ax = plt.gca()
        fig = plt.gcf()

        rec = ax.scatter(
            self.xy_ravel[:, 0],
            self.xy_ravel[:, 1],
            s=150,
            zorder=1,
            label="Receivers",
            marker="X",
        )

        # if plot_gravity == True:
        # xx, yy = np.meshgrid(self.X, self.Y)
        # triang = tri.Triangulation(self., y)
        # interpolator = tri.LinearTriInterpolator(triang, self.grav)
        # gravity = tf.reshape(self.grav, [self.grav_res_y, self.grav_res_x])
        # img = ax.contourf(xx, yy, gravity,levels=50, zorder=-1)

        # img = ax.contourf(xx, yy, gravity, zorder=-1)
        # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        ax.legend([rec], ["Receivers"], loc="center left", bbox_to_anchor=(1.3, 0.5))

        # clb = plt.colorbar(img)
        # clb.set_label('mgal',labelpad=-65, y=1.06, rotation=0)

        index = self.geo_data.surface_points.df.index[0 : self.Number_surface_points]
        xs = self.geo_data.surface_points.df["X"].to_numpy()
        ys = self.geo_data.surface_points.df["Y"].to_numpy()
        # adding number to surface points
        for x, y, i in zip(xs, ys, index):
            plt.text(x + 50, y + 50, i, color="red", fontsize=12)

        fig.set_size_inches((7, 7))
        if save == True:
            plt.savefig(
                str("/content/drive/My Drive/RWTH/Figs/" + self.modelName + ".png")
            )

        return fig, ax

    def plot_gravity(self, receivers, values=None):
        if values is None:
            values = self.grav.numpy()

        from scipy.interpolate import griddata

        x_min = np.min(receivers[:, 0])
        x_max = np.max(receivers[:, 0])
        y_min = np.min(receivers[:, 1])
        y_max = np.max(receivers[:, 1])
        grid_x, grid_y = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
        grav = griddata(receivers, values, (grid_x, grid_y), method="cubic")
        plt.imshow(
            grav.T,
            extent=(self.W, self.E, self.S, self.N),
            origin="lower",
            cmap="viridis",
        )
        plt.plot(receivers[:, 0], receivers[:, 1], "k.")
        plt.xlabel("x")
        plt.ylabel("y")

    @tf.function
    def calculate_grav(self, surface_coord, values_properties):

        ## set the dips position the same as surface point position

        self.TFG = TFGraph(
            self.dip_angles,
            self.azimuth,
            self.polarity,
            self.fault_drift,
            self.grid_tensor,
            self.values_properties,
            self.len_rest_form,
            self.Range,
            self.C_o,
            self.nugget_effect_scalar,
            self.nugget_effect_grad,
            self.rescale_factor,
            dtype=self.tfdtype,
        )
        Z_x = self.TFG.scalar_field(surface_coord, self.dips_position)

        scalar_field_at_surface_points = self.TFG.get_scalar_field_at_surface_points(
            Z_x
        )
        # formations_block = self.TFG.export_formation_block(
        #     Z_x, scalar_field_at_surface_points, self.values_properties)
        formations_block = self.TFG.export_formation_block(
            Z_x, scalar_field_at_surface_points, values_properties
        )

        # densities = formations_block[1][self.lg_0:self.lg_1]
        densities = formations_block[1][self.lg_0 : self.lg_1]

        gravity = self.TFG.compute_forward_gravity(
            self.tz, self.lg_0, self.lg_1, densities
        )

        return gravity


    ## compatibility with gempy plotting function
    def scalar_field(self, surface_coord):
        """
        This is only for regular grid
        """
        self.TFG = TFGraph(
            self.dip_angles,
            self.azimuth,
            self.polarity,
            self.fault_drift,
            self.grid_tensor,
            self.values_properties,
            self.len_rest_form,
            self.Range,
            self.C_o,
            self.nugget_effect_scalar,
            self.nugget_effect_grad,
            self.rescale_factor,
            dtype=self.tfdtype,
        )

        self.grid = self.geo_data.grid
        self._grid = self.grid  ## this is due to new gempy naming
        self._surfaces = self.geo_data.surfaces
        self._surface_points = self.geo_data.surface_points
        self._orientations = self.geo_data.orientations

        self.Z_x = self.TFG.scalar_field(surface_coord, self.dips_position)

        self.scalar_field_at_surface_points = (
            self.TFG.get_scalar_field_at_surface_points(self.Z_x)
        )

        # get lithologies at surface
        self.formations_block = self.TFG.export_formation_block(
            self.Z_x, self.scalar_field_at_surface_points, self.values_properties
        )

        regular = self.grid.get_grid_args("regular")
        # regular_grid = self._grid[regular[0]:regular[1]]
        # regular_scalar = self.Z_x[regular[0]:regular[1]]
        # meshgrid = np.reshape(regular_scalar,self._grid.regular_grid.resolution)
        self.solutions.scalar_field_matrix = tf.expand_dims(
            self.Z_x[regular[0] : regular[1]], axis=0
        ).numpy()
        self.solutions.lith_block = np.round(
            self.formations_block[0][regular[0] : regular[1]]
        )
        self.solutions.scalar_field_at_surface_points = np.expand_dims(
            self.scalar_field_at_surface_points.numpy(), axis=0
        )

        # this is pseudo code for my plotting, should only work for 1 series
        self.solutions.mask_matrix = np.expand_dims(
            np.ones(self.solutions.lith_block.shape, dtype="bool"), axis=0
        )
        self.solutions.mask_matrix_pad = [
            np.ones(self.grid.regular_grid.resolution, dtype="bool")
        ]

        self.solutions.grid = self.grid
        self.solutions.compute_all_surfaces()

        #####
        sfai_order = self.solutions.scalar_field_at_surface_points.sum(axis=0)
        self.sfai_order_0 = sfai_order
        sel = self._surfaces.df["isActive"] & ~self._surfaces.df["isBasement"]
        self._surfaces.df.loc[sel, "sfai"] = sfai_order
        self._surfaces.df.sort_values(
            by=["series", "sfai"], inplace=True, ascending=False
        )

        self._surface_points.df["id"] = (
            self._surface_points.df["surface"]
            .map(self._surfaces.df.set_index("surface")["id"])
            .astype(int)
        )
        self._orientations.df["id"] = (
            self._orientations.df["surface"]
            .map(self._surfaces.df.set_index("surface")["id"])
            .astype(int)
        )
        self._surface_points.sort_table()
        self._orientations.sort_table()

        self._surfaces.reset_order_surfaces()
        self._surfaces.sort_surfaces()
        self._surfaces.set_basement()
        #####

        return self.Z_x

    def calculate_grav_RegAll(self):
        self.activate_regular_grid()
        self.scalar_field(self.surface_points_coord)
        g = GravityPreprocessingRegAll(self.model.grid.regular_grid)

    ## compatibility with gempy plotting function


class GravityPreprocessingRegAll(RegularGrid):
    def __init__(self, model, regular_grid: RegularGrid = None):

        if regular_grid is None:
            super().__init__()
        elif isinstance(regular_grid, RegularGrid):
            self.model = model
            # self.kernel_centers = np.repeat(regular_grid.values[:,:,np.newaxis],2,axis=2) - model.xy_ravel.T
            self.num_receivers = self.model.xy_ravel.shape[0]
            # self.kernel_dxyz_right = regular_grid.kernel_dxyz_right
            # self.kernel_dxyz_left = regular_grid.kernel_dxyz_left
        self.tz = np.empty(0)

    def set_tz_kernel(self, scale=True, **kwargs):
        # grid_values = self.kernel_centers
        # dimension of x y z
        dx, dy, dz = self.model.grid.regular_grid.get_dx_dy_dz()

        # we need to find the closest center for each receiver to keep numerical stability
        # here we find the smallest center which is greater than the receiver coordinates for x and y
        re_x = self.model.xy_ravel.T[0]
        re_x = re_x + (dx / 2 - re_x % dx)

        re_y = self.model.xy_ravel.T[1]
        re_y = re_y + (dx / 2 - re_y % dx)

        new_xy_ravel = np.stack(
            [
                re_x,
                re_y,
            ],
            axis=0,
        )
        # concat with z value
        self.new_xy_ravel = np.concatenate(
            [new_xy_ravel, self.model.xy_ravel.T[2, None]]
        )

        
        kernel_centers = (
            np.repeat(
                self.model.grid.regular_grid.values[:, :, np.newaxis],
                self.num_receivers,
                axis=2,
            )
            - self.new_xy_ravel
        )
        kernel_centers = kernel_centers[:, :, :]

        x_cor = np.stack(
            (kernel_centers[:, 0] - dx / 2, kernel_centers[:, 0] + dx / 2), axis=1
        )
        y_cor = np.stack(
            (kernel_centers[:, 1] - dy / 2, kernel_centers[:, 1] + dy / 2), axis=1
        )
        z_cor = np.stack(
            (kernel_centers[:, 2] + dz / 2, kernel_centers[:, 2] - dz / 2), axis=1
        )

        # ...and prepare them for a vectorial op
        x_matrix = np.repeat(x_cor, 4, axis=1)
        y_matrix = np.tile(np.repeat(y_cor, 2, axis=1), (1, 2, 1))
        z_matrix = np.tile(z_cor, (1, 4, 1)) - 0.001 * self.model.extent[-1]

        s_r = np.sqrt(x_matrix ** 2 + y_matrix ** 2 + z_matrix ** 2)

        # This is the vector that determines the sign of the corner of the voxel
        mu = np.array([1, -1, -1, 1, -1, 1, 1, -1])
        mu = np.tile(mu, (self.num_receivers, 1)).T

        G = 6.674e-3  # ugal     cm3⋅g−1⋅s−26.67408e-2 -- 1 m/s^2 to microgal =

        self.tz = G * np.sum(
            -1
            * mu
            * (
                x_matrix * np.log(y_matrix + s_r)
                + y_matrix * np.log(x_matrix + s_r)
                - z_matrix * np.arctan(x_matrix * y_matrix / (z_matrix * s_r))
            ),
            axis=1,
        )
        return self.tz

