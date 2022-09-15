import pyvista as pv
import numpy as np

pv.global_theme.background = 'white'
pv.global_theme.font.label_size = 50
# sphere objects

#### xcoordinate_ycoordinate_zcoordinate
name = '3000_5000_100'
sphere = pv.Sphere(radius=100,center=(3000,5000,100))
# name = '5000_5000_870'
# sphere = pv.Sphere(radius=100,center=(5000,5000,870))


# boxex
roi_model = pv.Cube(center=(5000, 5000, 500),
              x_length=6000, y_length=6000, z_length=1000)

roi_padding = pv.Cube(center=(5000, 5000, 500),
              x_length=10000, y_length=10000, z_length=1000)

# receivers
Y_r = [5000]
number_receivers = 11
X_r = np.linspace(2300,7700,number_receivers)


r = []
for x in X_r:
  for y in Y_r:
    r.append(np.array([x,y]))
receivers = np.array(r)
Z_r = 1200
n_devices = receivers.shape[0]
xyz = np.meshgrid(X_r, Y_r, Z_r)
xy_ravel = np.vstack(list(map(np.ravel, xyz))).T
poly = pv.PolyData(xy_ravel)
geom = pv.Cone(direction=[0.0, 0.0, -1.0])
glyphs = poly.glyph(factor=200.0,geom=geom)

p = pv.Plotter(shape=(1, 1))
p.add_mesh(sphere, color="red", show_edges=False)
p.add_mesh(roi_model, opacity=0.15, color="#a4d294")
p.add_mesh(roi_padding, opacity=0.15, color="#3679ad")
p.add_mesh(glyphs, opacity=0.15, color="black")
p.add_bounding_box(color='gray')
p.show_bounds(color='gray',bounds = [0,10000, 0,10000, 0, 1000],location='outer')
p.set_scale(1, 1, 1.05)

# hsize = 1000

cpos = p.show(cpos=[(-4978.033300391534, -5668.684211203955, 5207.980692559044),
 (4893.089563999404, 4839.699534670127, -484.05937145097766),
 (0.27435251033635133, 0.2676071484839352, 0.9236433912243615)],
              window_size = [1500, 1000],
              screenshot='/Users/zhouji/Documents/Geophysics/GridPaper/Fig/sherePyvista_1st_'+str(name)+'.png'
              )


print(cpos)