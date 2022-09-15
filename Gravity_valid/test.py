# %%
import tensorflow as tf

import os
import numpy as np
import sys
import timeit
import csv
sys.path.append('/Users/zhouji/Documents/github/YJ/GP_old')

sys.path.append('/Users/zhouji/Documents/github/YJ/')
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
    ax.annotate(string, (1, 1), xytext=(-8, -8), ha='right', va='top',
                size=14, xycoords='axes fraction', textcoords='offset points',fontsize = 18,fontweight='bold')

class ModelThree(object):
    def __init__(self,master_path, surface_path, orientations_path, name = None, receivers=None,dtype='float32'):
    
        if dtype == 'float32':
            self.tfdtype = tf.float32
        elif dtype == 'float64':
            self.tfdtype = tf.float64
        
        self.dtype = dtype 
        
        self.path = master_path
        
        if name is None:
            self.modelName = 'model'
        else:
            self.modelName = str(name)
            
        regular_grid_resolution = [20, 20, 30]
        # the regular cell is 200*200*200
        self.geo_data = create_data([0, 1000, 0, 1000, -700, 300], resolution=regular_grid_resolution,
                        path_o=self.path+ orientations_path,
                        path_i=self.path + surface_path)
        
        map_series_to_surfaces(self.geo_data, {"Strat_Series": (
            'rock2', 'rock1'), "Basement_Series": ('basement')})
        
        # self.geo_data = create_data([self.W, self.E, self.S, self.N, -1000, self.top], resolution=[1, 1, 1],
        #                 path_o=self.path+ orientations_path,
        #                 path_i=self.path + surface_path)
        # map_series_to_surfaces(self.geo_data, {"Strat_Series": ('OVB', 'UPX'), "Basement_Series": ('basement')})
        
        
        # define density
        ## the order is following the indexing not id in geo_data.surfaces
        self.geo_data.add_surface_values([2.5, 3.5, 2.0]) # density 2*10^3 kg/m3 (sandstone,igeous rock,salt )

        # define indexes where we set dips position as surface position
        self.sf = self.geo_data.surface_points.df.loc[:,['X','Y','Z']]

        # self.indexes = self.sf[((self.sf['Y']==500))&((self.sf['X']==400)| (self.sf['X']==600))].index.values

        

        ## customize irregular receiver set-up, XY pairs are required
        # append a Z value 1000 to the array
        self.top = 300
        self.Zs = np.expand_dims(self.top*np.ones([receivers.shape[0]]),axis =1)
        self.xy_ravel = np.concatenate([receivers,self.Zs],axis=1)
            

        self.geo_data.set_centered_grid(self.xy_ravel, resolution=[40, 40, 50], radius=1000)
        
        # self.activate_centered_grid()
        self.from_gempy_interpolator()
        
        mu_true = self.geo_data.surface_points.df.loc[:,'Z'].to_numpy()
        self.mu_true = tf.convert_to_tensor(mu_true,self.tfdtype)
        
        # surface_coord =tf.concat([self.surface_points_coord[:,:2],Z_coord],axis = 1)

        
        # self.grav = self.calculate_grav(surface_coord)

        self.mu_prior = self.mu_true
        self.sigma = 50
        self.mu_prior_r = (self.mu_prior-self.centers[2])/self.rf+0.5001
        self.cov_prior = ((self.sigma))**2 * tf.eye(self.Number_surface_points, dtype=self.tfdtype) 
        self.cov_prior_r = ((self.sigma)/self.rf)**2 * tf.eye(self.Number_surface_points, dtype=self.tfdtype) #$$# double check here

    def from_gempy_interpolator(self):
        self.interpolator = self.geo_data.interpolator

        # extract data from gempy interpolator
        dips_position, dip_angles, azimuth, polarity, surface_points_coord, fault_drift, grid, values_properties = self.interpolator.get_python_input_block()[0:-3]


        dip_angles = tf.cast(dip_angles,self.tfdtype)
        _grid = grid
        grid = tf.cast(grid,self.tfdtype)
        self.dips_position = tf.cast(dips_position,self.tfdtype)
        azimuth = tf.cast(azimuth,self.tfdtype)
        polarity = tf.cast(polarity,self.tfdtype)
        fault_drift = tf.cast(fault_drift,self.tfdtype)
        values_properties = tf.cast(values_properties,self.tfdtype)
        self.Number_surface_points = int(surface_points_coord.shape[0])


        self.surface_points = self.geo_data.surface_points
        self.surfaces = self.geo_data.surfaces
        self.orientations = self.geo_data.orientations
        
        self.centers = self.geo_data.rescaling.df.loc['values', 'centers'].astype(self.dtype)

        g = GravityPreprocessing(self.geo_data.grid.centered_grid)

        # precomputed gravity impact from each grid
        tz = g.set_tz_kernel()

        self.tz = tf.cast(tz,self.tfdtype)

        len_rest_form = self.interpolator.additional_data.structure_data.df.loc[
            'values', 'len surfaces surface_points'] - 1
        Range = self.interpolator.additional_data.kriging_data.df.loc['values', 'range']
        C_o = self.interpolator.additional_data.kriging_data.df.loc['values', '$C_o$']
        rescale_factor = self.interpolator.additional_data.rescaling_data.df.loc[
            'values', 'rescaling factor']

        self.rf = rescale_factor
        nugget_effect_grad = np.cast[self.dtype](
            np.tile(self.interpolator.orientations.df['smooth'], 3))
        nugget_effect_scalar = np.cast[self.dtype](
            self.interpolator.surface_points.df['smooth'])

        # surface_points_coord = tf.Variable(surface_points_coord, dtype=self.tfdtype)

        self.dip_angles = tf.convert_to_tensor(dip_angles,dtype=self.tfdtype)
        self.azimuth = tf.convert_to_tensor(azimuth,dtype=self.tfdtype)
        self.polarity = tf.convert_to_tensor(polarity,dtype=self.tfdtype)

        self.fault_drift = tf.convert_to_tensor(fault_drift,dtype=self.tfdtype)
        self.grid_tensor = tf.convert_to_tensor(grid,dtype=self.tfdtype)
        self.values_properties = tf.convert_to_tensor(values_properties,dtype=self.tfdtype)
        self.len_rest_form = tf.convert_to_tensor(len_rest_form,dtype=self.tfdtype)
        self.Range = tf.convert_to_tensor(Range, self.tfdtype)
        self.C_o = tf.convert_to_tensor(C_o,dtype=self.tfdtype)
        self.nugget_effect_grad = tf.convert_to_tensor(nugget_effect_grad,dtype=self.tfdtype)
        self.nugget_effect_scalar = tf.convert_to_tensor(nugget_effect_scalar,dtype=self.tfdtype)
        self.rescale_factor = tf.convert_to_tensor(rescale_factor, self.tfdtype)
        
        self.surface_points_coord = tf.convert_to_tensor(surface_points_coord,self.tfdtype)
        
        self.solutions = Solution(_grid,self.geo_data.surfaces,self.geo_data.series)
        
        self.lg_0 = self.interpolator.grid.get_grid_args('centered')[0]
        self.lg_1 = self.interpolator.grid.get_grid_args('centered')[1]
        
    def activate_centered_grid(self,):
        self.geo_data.grid.deactivate_all_grids()
        self.geo_data.grid.set_active(['centered'])
        self.geo_data.update_from_grid()
        self.from_gempy_interpolator()
    
    def activate_regular_grid(self,):
        self.geo_data.grid.deactivate_all_grids()
        self.geo_data.grid.set_active(['regular'])
        self.geo_data.update_from_grid()
        self.from_gempy_interpolator()

    def plot_model(self, plot_gravity = False, save = False):
        
        # use gempy build-in method to plot surface points and orientation points
        geomode = gp.plot.plot_data(self.geo_data, direction='z')

        ax = plt.gca()
        fig = plt.gcf()

        rec = ax.scatter(self.xy_ravel[:, 0], self.xy_ravel[:, 1], s=150, zorder=1,label = 'Receivers',marker = 'X')

        # if plot_gravity == True:
            # xx, yy = np.meshgrid(self.X, self.Y)
            # triang = tri.Triangulation(self., y)
            # interpolator = tri.LinearTriInterpolator(triang, self.grav)
            # gravity = tf.reshape(self.grav, [self.grav_res_y, self.grav_res_x])
            # img = ax.contourf(xx, yy, gravity,levels=50, zorder=-1)

        # img = ax.contourf(xx, yy, gravity, zorder=-1)
        # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        ax.legend([rec],['Receivers'],loc='center left', bbox_to_anchor=(1.3, 0.5))

        # clb = plt.colorbar(img)
        # clb.set_label('mgal',labelpad=-65, y=1.06, rotation=0)

        index = self.geo_data.surface_points.df.index[0:self.Number_surface_points]
        xs = self.geo_data.surface_points.df['X'].to_numpy()
        ys = self.geo_data.surface_points.df['Y'].to_numpy()
        # adding number to surface points
        for x, y, i in zip(xs, ys, index):
            plt.text(x+50, y+50, i, color="red", fontsize=12)

        fig.set_size_inches((7,7))
        if save == True:
            plt.savefig(str('/content/drive/My Drive/RWTH/Figs/'+self.modelName+'.png'))
            
        return fig,ax
    
    def plot_gravity(self,receivers,values = None):
        if values is None: values = self.grav.numpy()
        
        from scipy.interpolate import griddata
        x_min = np.min(receivers[:,0])
        x_max = np.max(receivers[:,0])
        y_min = np.min(receivers[:,1])
        y_max = np.max(receivers[:,1])
        grid_x, grid_y = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
        grav = griddata(receivers, values, (grid_x, grid_y), method='cubic')
        plt.imshow(grav.T, extent=(self.W, self.E, self.S, self.N), origin='lower',cmap = 'viridis')
        plt.plot(receivers[:,0],receivers[:,1],'k.')
        plt.xlabel('x')
        plt.ylabel('y')
        

    @tf.function    
    def calculate_grav(self,surface_coord, values_properties):

        ## set the dips position the same as surface point position

        self.TFG = TFGraph(self.dip_angles, self.azimuth,
                self.polarity, self.fault_drift,
                self.grid_tensor, self.values_properties, self.len_rest_form, self.Range,
                self.C_o, self.nugget_effect_scalar, self.nugget_effect_grad,
                self.rescale_factor,dtype = self.tfdtype)
        Z_x = self.TFG.scalar_field(surface_coord,self.dips_position)

        scalar_field_at_surface_points = self.TFG.get_scalar_field_at_surface_points(Z_x)
        # formations_block = self.TFG.export_formation_block(
        #     Z_x, scalar_field_at_surface_points, self.values_properties)
        formations_block = self.TFG.export_formation_block(
            Z_x, scalar_field_at_surface_points, values_properties)

        # densities = formations_block[1][self.lg_0:self.lg_1]
        densities = formations_block[1][self.lg_0:self.lg_1]

        gravity = self.TFG.compute_forward_gravity(self.tz, self.lg_0, self.lg_1, densities)
    
        return gravity,densities
            
            
    # def Create_NoisyData(self, number_data = 10,std = 3,save = False, path = None):
    #     grav_list = []
    #     start = timeit.default_timer()
    #     number_forward = number_data
    #     tf.random.set_seed(16)
    #     for i in range(number_forward):
    #         mu = self.mu_true+tf.random.normal(self.mu_true.shape,mean = 0,stddev=std,dtype=self.tfdtype)
    #         Z_coord = tf.expand_dims(tf.concat([((mu-self.thickness-self.centers[2])/self.rf+0.5001),((mu-self.centers[2])/self.rf+0.5001)],axis=0),axis=1)
    #         surface_coord =tf.concat([self.surface_points_coord[:,:2],Z_coord],axis = 1)
    #         grav = self.calculate_grav(surface_coord)
    #         grav_list.append(grav)
    #     end = timeit.default_timer()
    #     print('time for {} samples: {:.3}'.format(number_forward,(end - start)))
    #     self.G = np.array(grav_list)
        
    #     if save == True:
    #     ## Save data
    #         json_dump = json.dumps({'G': self.G}, cls=NumpyEncoder)

    #         with open(path, 'w') as outfile:
    #             json.dump(json_dump, outfile)

    # def Load_GravityData(self,path):
    #     with open(path) as f:
    #         data = json.load(f)
    #     data = json.loads(data)
    #     self.G = np.asarray(data['G'])
    
    # def G_stat(self):
    #     self.cov_matrix = tf.linalg.diag(tfp.stats.variance(self.G))
    #     self.Data = tf.convert_to_tensor(np.mean(self.G,axis=0),dtype=self.tfdtype)
        
        
    # # @tf.function
    # def negative_log_posterior(self, mu):
    #     return tf.negative(self.joint_log( mu))
    
    # def loss(self,mu):
    #     lost =  tf.negative(self.joint_log(mu*1000))
    #     return lost

    # def loss_minimize(self):
    #     lost =  tf.negative(self.joint_log(self.mu*1000))
    #     return lost

    
    # def findMAP(self,mu_init,method = 'Nadam',learning_rate = 0.002, iterations = 500):

    #     # mu_init = mu_true
    #     if method == 'Nadam':
    #         optimizer = tf.keras.optimizers.Nadam(
    #             learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08
    #         )
    #     if method == 'Adam':
    #         optimizer = tf.keras.optimizers.Adam(
    #             learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08
    #         )
    #     cost_A = []
    #     mu_list = []
    #     self.mu = tf.Variable(mu_init/1000)
    #     start = timeit.default_timer()
        
    #     tolerance  = 3e-3

    #     for step in range(iterations):

    #         optimizer.minimize(self.loss_minimize, var_list=[self.mu])
    #         loss = self.loss(self.mu).numpy()
            
    #         # stop criteria: if cost stops decreasing
    #         if cost_A:
    #             if (cost_A[-1]-loss)>0 and (cost_A[-1]-loss)<tolerance: 
    #                 break # check if cost list is empty
            
    #         cost_A.append(loss)

    #         print ('step:',step,'loss:',loss)

    #         mu_list.append(self.mu.numpy())
    #     end = timeit.default_timer()
    #     print('Adam: %.3f' % (end - start))
    #     self.MAP = tf.convert_to_tensor(mu_list[-1],self.tfdtype)*1000
        
    #     return mu_list,cost_A
    
    # @tf.function  
    # def joint_log(self,mu):
  
    #     Z_coord = tf.expand_dims(tf.concat([((mu-self.thickness-self.centers[2])/self.rf+0.5001),((mu-self.centers[2])/self.rf+0.5001)],axis=0),axis=1)
    #     surface_coord =tf.concat([self.surface_points_coord[:,:2],Z_coord],axis = 1)
        
    #     mvn_prior = tfd.MultivariateNormalTriL(
    #         loc=self.mu_prior_r,
    #         scale_tril = tf.linalg.cholesky(self.cov_prior_r))
        
    #     Gm_ = self.calculate_grav(surface_coord)

    #     mvn_likelihood = tfd.MultivariateNormalTriL(
    #         loc=Gm_,
    #         scale_tril=tf.cast(tf.linalg.cholesky(self.cov_matrix),self.tfdtype))
        
    #     likelihood_log_prob = tf.reduce_sum(mvn_likelihood.log_prob(self.G))

    #     prior_log_prob = mvn_prior.log_prob(tf.squeeze(Z_coord[self.Number_para:]))

    #     joint_log = prior_log_prob + likelihood_log_prob
    #     tf.print('prior:',prior_log_prob)
    #     tf.print('likelihood:',likelihood_log_prob)

    #     return joint_log
  
    # @tf.function
    # def hvp(self,mu,tangents):
    #     with tf.autodiff.ForwardAccumulator(mu, tangents) as acc:
    #         with tf.GradientTape(watch_accessed_variables=False) as t:
    #             t.watch(mu)
    #             joint_log = tf.negative(self.joint_log(mu))
    #         loss = t.gradient(joint_log,mu)
    #     hess = acc.jvp(loss)
    #     return(hess)
    
    # def Compile_graph(self):
    #     tangents = np.zeros(self.MAP.shape)
    #     tangents[0]=1
    #     tangents = tf.convert_to_tensor(tangents,dtype=self.tfdtype)
    #     self.hvp(self.mu_true,tangents)
    #     return
    
    # def calculate_Hessian(self,MAP):
    #     Hess = []
    #     start = timeit.default_timer()
    #     for i in range(self.Number_para):
    #         tangents = np.zeros(MAP.shape)
    #         tangents[i]=1
    #         tangents = tf.convert_to_tensor(tangents,dtype=self.tfdtype)

    #         Hess.append(self.hvp(MAP,tangents).numpy())
    #         self.Hess = np.array(Hess)
    #     end = timeit.default_timer()
    #     print('time for Hessian calculation: %.3f' % (end - start))
    #     return self.Hess
            
        
    # def test_Hess(self):
    #     try:
    #         tf.linalg.cholesky(self.Hess)
    #         print('Hessian is positive definite')
    #     except:
    #         print('check eigen value')
        
    # def save_Hessian(self,path):
    #     json_dump = json.dumps({'Hess': self.Hess.numpy()}, cls=NumpyEncoder)
    #     with open(path, 'w') as outfile:
    #         json.dump(json_dump, outfile)
            
    # def load_Hessian(self,path):
    #     with open(path) as f:
    #         data = json.load(f)
    #         data = json.loads(data)
    #         self.Hess = np.asarray(data['Hess'])
            
    # def Hess_eigen(self,plot = True):
    #     eigval,eigvec = np.linalg.eig(self.Hess)
    #     if plot is True:
    #         plt.plot(np.sort(eigval)[::-1],'-o')
    #         plt.title('eig value of  Hessian')
    #     return eigval,eigvec
    
    # def load_data(self,path):
    #     with open(path) as f:
    #         data = json.load(f)
    #         data = json.loads(data)
            
    #     self.cov_matrix = tf.convert_to_tensor(np.asarray(data['cov_matrix']),dtype = self.tfdtype)
    #     self.Data = tf.convert_to_tensor(np.asarray(data['Data']),dtype = self.tfdtype)
    #     MAP = tf.convert_to_tensor(np.asarray(data['MAP']),dtype = self.tfdtype)
    #     self.MAP = MAP
    #     self.len_rest_form = tf.convert_to_tensor(np.asarray(data['len_rest_form']),dtype = self.tfdtype)
    #     self.nugget_effect_scalar = tf.convert_to_tensor(np.asarray(data['nugget_effect_scalar']),dtype = self.tfdtype)
    #     self.nugget_effect_grad = tf.convert_to_tensor(np.asarray(data['nugget_effect_grad']),dtype = self.tfdtype)
    #     rescale_factor =  tf.convert_to_tensor(np.asarray(data['rescale_factor'])[0],dtype = self.tfdtype)
    #     self.surface_points_coord =  tf.convert_to_tensor(np.asarray(data['surface_points_coord']),dtype = self.tfdtype)
    #     self.mu_prior_r =  tf.convert_to_tensor(np.asarray(data['mu_prior_r']),dtype = self.tfdtype)
    #     self.cov_prior_r =  tf.convert_to_tensor(np.asarray(data['cov_prior_r']),dtype = self.tfdtype)
    #     self.tz = tf.convert_to_tensor(np.asarray(data['tz']),dtype = self.tfdtype)
    #     self.rf = rescale_factor

    #     self.centers = data['centers']
    #     # indexes = data['indexes']

    #     self.Range = tf.convert_to_tensor(np.asarray(data['Range'])[0],dtype = self.tfdtype)
    #     self.C_o = tf.convert_to_tensor(np.asarray(data['C_o'])[0],dtype = self.tfdtype)
    #     self.thickness = tf.convert_to_tensor(np.asarray(data['thickness'])[0],dtype = self.tfdtype)
        
    #     self.lg_0 = data['lg_0'][0]
    #     self.lg_1 = data['lg_1'][0]
    #     self.Number_para = data['Number_para'][0]
        
    # def export_data(self,path):

    #     json_dump = json.dumps({'cov_matrix': self.cov_matrix.numpy(),
    #                             'Data':self.Data.numpy(), 
    #                             'MAP': self.MAP.numpy(),
    #                             'dip_angles':self.dip_angles.numpy(),
    #                             'azimuth':self.azimuth.numpy(),
    #                             'polarity':self.polarity.numpy(),
    #                             'fault_drift':self.fault_drift.numpy(),
    #                             'grid':self.grid_tensor.numpy(),
    #                             'values_properties':self.values_properties.numpy(),
    #                             'len_rest_form':self.len_rest_form.numpy(),
    #                             'Range':np.array([self.Range.numpy()]),
    #                             'C_o':np.array([self.C_o.numpy()]),
    #                             'nugget_effect_scalar':self.nugget_effect_scalar.numpy(),
    #                             'nugget_effect_grad':self.nugget_effect_grad.numpy(),
    #                             'rescale_factor':np.array([self.rescale_factor.numpy()]),
    #                             'thickness':np.array([self.thickness.numpy()]),
    #                             'lg_0':np.array([self.lg_0]),
    #                             'lg_1':np.array([self.lg_1]),
    #                             'Number_para':np.array([self.Number_para]),
    #                             'centers':self.centers,
    #                             'surface_points_coord':self.surface_points_coord.numpy(),
    #                             # 'self.indexes':indexes,
    #                             'mu_prior_r':self.mu_prior_r.numpy(),
    #                             'cov_prior_r':self.cov_prior_r.numpy(),
    #                             'tz':self.tz.numpy(),
    #                             'mu_prior':self.mu_prior.numpy(),


    #                             }, cls=NumpyEncoder)
    #     with open(path, 'w') as outfile:
    #         json.dump(json_dump, outfile)
            
        
    ## compatibility with gempy plotting function
    def scalar_field(self,surface_coord):
        """
            This is only for regular grid
        """
        self.TFG = TFGraph(self.dip_angles, self.azimuth,
                self.polarity, self.fault_drift,
                self.grid_tensor, self.values_properties, self.len_rest_form, self.Range,
                self.C_o, self.nugget_effect_scalar, self.nugget_effect_grad,
                self.rescale_factor,dtype = self.tfdtype)
        
        self.grid = self.geo_data.grid
        self._grid = self.grid ## this is due to new gempy naming
        self._surfaces = self.geo_data.surfaces 
        self._surface_points = self.geo_data.surface_points
        self._orientations = self.geo_data.orientations
        
        self.Z_x = self.TFG.scalar_field(surface_coord,self.dips_position)
        

        self.scalar_field_at_surface_points = self.TFG.get_scalar_field_at_surface_points(self.Z_x)
        
        # get lithologies at surface
        self.formations_block = self.TFG.export_formation_block(
        self.Z_x, self.scalar_field_at_surface_points, self.values_properties)
        
        regular = self.grid.get_grid_args('regular')
        # regular_grid = self._grid[regular[0]:regular[1]]
        # regular_scalar = self.Z_x[regular[0]:regular[1]]
        # meshgrid = np.reshape(regular_scalar,self._grid.regular_grid.resolution)
        self.solutions.scalar_field_matrix = tf.expand_dims(self.Z_x[regular[0]:regular[1]],axis=0).numpy()
        self.solutions.lith_block = np.round(self.formations_block[0][regular[0]:regular[1]])
        self.solutions.scalar_field_at_surface_points = np.expand_dims(self.scalar_field_at_surface_points.numpy(),axis=0)
        
        # this is pseudo code for my plotting, should only work for 1 series 
        self.solutions.mask_matrix = np.expand_dims(np.ones(self.solutions.lith_block.shape, dtype='bool'),axis=0)
        self.solutions.mask_matrix_pad = [np.ones(self.grid.regular_grid.resolution, dtype='bool')]
        
        self.solutions.grid = self.grid
        self.solutions.compute_all_surfaces()
        
        #####
        sfai_order = self.solutions.scalar_field_at_surface_points.sum(axis=0)
        self.sfai_order_0 = sfai_order
        sel = self._surfaces.df['isActive'] & ~self._surfaces.df['isBasement']
        self._surfaces.df.loc[sel, 'sfai'] = sfai_order
        self._surfaces.df.sort_values(by=['series', 'sfai'], inplace=True, ascending=False)

        self._surface_points.df['id'] = self._surface_points.df['surface'].map(
            self._surfaces.df.set_index('surface')['id']).astype(int)
        self._orientations.df['id'] = self._orientations.df['surface'].map(
            self._surfaces.df.set_index('surface')['id']).astype(int)
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
        dx,dy,dz = self.model.grid.regular_grid.get_dx_dy_dz()
        kernel_centers = np.repeat(self.model.grid.regular_grid.values[:,:,np.newaxis],self.num_receivers,axis=2)-self.model.xy_ravel.T
        kernel_centers = kernel_centers[:,:,:]

        x_cor = np.stack((kernel_centers[:,0] - dx/2,kernel_centers[:,0] + dx/2), axis=1)
        y_cor = np.stack((kernel_centers[:,1] - dy/2,kernel_centers[:,1] + dy/2), axis=1)
        z_cor = np.stack((kernel_centers[:,2] + dz/2,kernel_centers[:,2] - dz/2), axis=1)

                
        # ...and prepare them for a vectorial op
        x_matrix = np.repeat(x_cor, 4, axis=1)
        y_matrix = np.tile(np.repeat(y_cor, 2, axis=1), (1, 2, 1))
        z_matrix = np.tile(z_cor, (1, 4, 1))-0.001

        s_r = np.sqrt(x_matrix ** 2 + y_matrix ** 2 + z_matrix ** 2)+0.001

        # This is the vector that determines the sign of the corner of the voxel
        mu = np.array([1, -1, -1, 1, -1, 1, 1, -1])
        mu = np.tile(mu,(self.num_receivers,1)).T
        
        
        G = 6.674e-3 # ugal     cm3⋅g−1⋅s−26.67408e-2 -- 1 m/s^2 to milligal = 

        self.tz = (
            G *
            np.sum(- 1 *
                    mu * (
                            x_matrix * np.log(y_matrix + s_r) +
                            y_matrix * np.log(x_matrix + s_r) -
                            z_matrix * np.arctan(x_matrix * y_matrix / (z_matrix * s_r))),
                    axis=1))
        return self.tz
        


    # def set_tz_kernel(self, scale=True, **kwargs):
    #     if self.kernel_centers.size == 0:
    #         self.set_centered_kernel(**kwargs)

    #     grid_values = self.kernel_centers

    #     s_gr_x = grid_values[:, 0]
    #     s_gr_y = grid_values[:, 1]
    #     s_gr_z = grid_values[:, 2]

    #     # getting the coordinates of the corners of the voxel...
    #     x_cor = np.stack((s_gr_x - self.kernel_dxyz_left[:, 0], s_gr_x + self.kernel_dxyz_right[:, 0]), axis=1)
    #     y_cor = np.stack((s_gr_y - self.kernel_dxyz_left[:, 1], s_gr_y + self.kernel_dxyz_right[:, 1]), axis=1)
    #     z_cor = np.stack((s_gr_z - self.kernel_dxyz_left[:, 2], s_gr_z + self.kernel_dxyz_right[:, 2]), axis=1)

    #     # ...and prepare them for a vectorial op
    #     x_matrix = np.repeat(x_cor, 4, axis=1)
    #     y_matrix = np.tile(np.repeat(y_cor, 2, axis=1), (1, 2))
    #     z_matrix = np.tile(z_cor, (1, 4))

    #     s_r = np.sqrt(x_matrix ** 2 + y_matrix ** 2 + z_matrix ** 2)

    #     # This is the vector that determines the sign of the corner of the voxel
    #     mu = np.array([1, -1, -1, 1, -1, 1, 1, -1])

    #     if scale is True:
    #         #
    #         G = 6.674e-3 # ugal     cm3⋅g−1⋅s−26.67408e-2 -- 1 m/s^2 to milligal = 100000 milligal
    #     else:
    #         from scipy.constants import G

    #     self.tz = (
    #         G *
    #         np.sum(- 1 *
    #                mu * (
    #                        x_matrix * np.log(y_matrix + s_r) +
    #                        y_matrix * np.log(x_matrix + s_r) -
    #                        z_matrix * np.arctan(x_matrix * y_matrix / (z_matrix * s_r))),
    #                axis=1))

    #     return self.tz