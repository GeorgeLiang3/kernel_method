import numpy as np



    
    
# %%
import tensorflow as tf

import os
import numpy as np
import sys
# import timeit
# import csv

sys.path.append("/Users/zhouji/Documents/github/YJ/GP_old")

sys.path.append("/Users/zhouji/Documents/github/YJ/")
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import json
import gempy as gp
from gempy.core.tensor.tensorflow_graph_test import TFGraph
import tensorflow as tf
import tensorflow_probability as tfp
# import pandas as pd
# from gempy.core.solution import Solution
# from gempy import create_data, map_series_to_surfaces
# from gempy.assets.geophysics import GravityPreprocessing
# from gempy.core.grid_modules.grid_types import RegularGrid


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


class Sphere_scalerfield(object):
    def __init__(
        self,
        sphere_radius,
        sphere_position,
        center_grid_resolution=[50, 50, 50],
        regular_grid_resolution=[100, 100, 30],
        name=None,
        receivers=None,
        dtype="float32",
        radius=[900, 900, 1000],
        extent = [0, 10000, 0, 10000, 0, 1000],
        # top = [5000,5000,995],
        # bot = [6000,5000,0]
    ):
        #######
        self.sphere_radius = sphere_radius
        self.sphere_position = sphere_position

        #######
        
        if dtype == "float32":
            self.tfdtype = tf.float32
        elif dtype == "float64":
            self.tfdtype = tf.float64

        self.dtype = dtype

        if name is None:
            self.modelName = "model"
        else:
            self.modelName = str(name)

        self.regular_grid_resolution = regular_grid_resolution
        # the regular cell is 200*200*200
        self.extent = extent
        # top_x = top[0]
        # top_y = 5000
        # top_z = top[-1]

        # bot_x = bot[0]
        # bot_y = 5000
        # bot_z = bot[-1]


        # find the 2d othorganal vector
        def PerpendicularConterClockwise(x,z):
            n = np.linalg.norm(np.array([x,z]))
            x/= n
            z/= n
            return -z,x


        # orien_x = (top_x+bot_x)/2
        # orien_y = (top_y+bot_y)/2
        # orien_z = (top_z+bot_z)/2

        self.geo_data = gp.create_model('Model1')
        self.geo_data = gp.init_data(self.geo_data, extent=[0, 10000, 0, 10000, 0, 1000], resolution=regular_grid_resolution)
        #############
        # self.geo_data.set_default_surfaces()
        # self.geo_data.add_surface_points(X=top_x, Y=top_y, Z=top_z, surface='surface1')
        # self.geo_data.add_surface_points(X=bot_x, Y=bot_y, Z=bot_z, surface='surface1')
        # vec = PerpendicularConterClockwise((bot_x-top_x),(bot_z - top_z))
        # self.geo_data.add_orientations(X=orien_x, Y=orien_y, Z=orien_z, surface='surface1', pole_vector=(vec[0], 0, vec[1]))

        # map_series_to_surfaces(
        #     self.geo_data,
        #     {"Strat_Series": ("surface1"), "Basement_Series": ("surface2")},
        # )

        # # define density
        # ## the order is following the indexing not id in geo_data.surfaces
        # self.geo_data.add_surface_values(
        #     [2.5, 3.5]
        # )  # density 2*10^3 kg/m3 (sandstone,igeous rock,salt )

        # # define indexes where we set dips position as surface position
        # self.sf = self.geo_data.surface_points.df.loc[:, ["X", "Y", "Z"]]
        #############
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
        
        self.grid = self.geo_data.grid

        # self.activate_centered_grid()
        # self.from_gempy_interpolator()

    def update_sphere(self,sphere_radius,sphere_position):
        self.sphere_radius = sphere_radius
        self.sphere_position = sphere_position
        
    def scalar_field(self,grid,sigmoid = True,slope = 50):
        dist = tf.sqrt(tf.reduce_sum(tf.square(grid - self.sphere_position),axis=1))
        if sigmoid:
            Z_x = tf.math.sigmoid(slope * (self.sphere_radius-dist))
        else:
            Z_x = tf.where(dist>self.sphere_radius,tf.constant(0,dtype = self.tfdtype),tf.constant(1,dtype = self.tfdtype))
            
        return Z_x
    
    