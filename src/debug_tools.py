from datetime import datetime
import time
def TicTocGenerator():
    ti, tf = 0, time.time()
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti
TicToc = TicTocGenerator()
def toc(tempBool=True):
    tempTimeInterval = next(TicToc)
    if tempBool: print( "Elapsed time: %f seconds.\n" %tempTimeInterval )
def tic():
    toc(False)

def plot_xy_raster(data, lon=[-180, 180], lat=[-90, 90], proj='cyl', cmap='viridis', sz=6):
    """Plots a raster on a world map assuming the data covers the whole world.

    Args:
        data (numpy.ndarray): A 2D numpy array containing the raster data.
        cmap (str, optional): The name of the matplotlib colormap to use (default: 'jet').
    """
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.basemap import Basemap

    if np.shape(data)[1] / np.shape(data)[0] > 2.0:
        new_data = data
        lines_to_fill = int((np.shape(data)[1] / 2) - np.shape(data)[0])
        data = np.append(new_data, np.zeros((lines_to_fill, np.shape(data)[1])), axis=0)
        del new_data
        del lines_to_fill

    plt.figure(figsize=(sz, int((float(sz) / float(np.shape(data)[0])) * float(np.shape(data)[1]))))
    m = Basemap(projection=proj, lon_0=0, resolution='c')
    lons, lats = np.meshgrid(np.linspace(np.min(lon), np.max(lon), data.shape[1]+1),
                             np.linspace(np.min(lat), np.max(lat), data.shape[0]+1))
    im = m.pcolormesh(lons, lats, np.rot90(np.swapaxes(data, 0, 1)), cmap=cmap, latlon=True)
    cb = m.colorbar(im, 'bottom', size='5%', pad='10%')
    plt.title('World Raster')
    m.drawcoastlines()
    m.drawcountries()
    m.drawmeridians(np.arange(-180, 180, 60), labels=[0,0,0,1])
    m.drawparallels(np.arange(-90, 90, 30), labels=[1,0,0,0])
    plt.savefig('test_map.png', dpi=300) 


def plot_xy_raster_domain(data, domain, proj='cyl', cmap='viridis', sz=6, name='', limits=[]):
    """Plots a raster on a world map assuming the data covers the whole world.

    Args:
        data (numpy.ndarray): A 2D numpy array containing the raster data.
        cmap (str, optional): The name of the matplotlib colormap to use (default: 'jet').
    """
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.basemap import Basemap

    lon = [domain[0], domain[2]]
    lat = [domain[3], domain[1]]

    plt.figure(figsize=(int((float(sz) / float(np.shape(data)[0])) * float(np.shape(data)[1])), sz))
    m = Basemap(projection=proj, llcrnrlat=domain[3], urcrnrlat=domain[1], llcrnrlon=domain[0], urcrnrlon=domain[2])
    m.drawcountries()
    lons, lats = np.meshgrid(np.linspace(np.min(lon), np.max(lon), data.shape[1]+1),
                             np.linspace(np.min(lat), np.max(lat), data.shape[0]+1))
    if len(limits) > 1:
        im = m.pcolormesh(lons, lats, np.rot90(np.swapaxes(data, 0, 1)), cmap=cmap, latlon=True, vmin=np.nanmin(limits), vmax=np.nanmax(limits))
    else:
        im = m.pcolormesh(lons, lats, np.rot90(np.swapaxes(data, 0, 1)), cmap=cmap, latlon=True)
    cb = m.colorbar(im, 'bottom', size='5%', pad='10%')
    plt.title('World Raster')
    
    if name == '':
        plt.savefig('test_map.png', dpi=200) 
    else:
        plt.savefig(name, dpi=200)

    plt.close()
    # print('complete!')


def plot_xy_raster_imshow(data, nodata = .0, lon=[-180, 180], lat=[-90, 90], cmap='viridis', sz=6, limits=[], interpolation='bilinear'):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    plt.figure(figsize=(sz, int((float(sz) / float(np.shape(data)[0])) * float(np.shape(data)[1]))))
    if len(limits) > 0:
        im = ax.imshow(data, cmap=cmap, aspect='equal', vmin=np.min(limits), vmax=np.max(limits), interpolation=interpolation)
    else:
        im = ax.imshow(data, cmap=cmap, aspect='equal', interpolation=interpolation)
    ax.set_aspect('equal')
    fig.colorbar(im)
    plt.close()
    fig.savefig('test_imshow.png', dpi=300)


def plot_xy_raster_domain_suitability(data, domain, proj='cyl', cmap='viridis', sz=6, name='', limits=[], nodata=0.):
    """Plots a raster on a world map assuming the data covers the whole world.

    Args:
        data (numpy.ndarray): A 2D numpy array containing the raster data.
        cmap (str, optional): The name of the matplotlib colormap to use (default: 'jet').
    """
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.basemap import Basemap

    lon = [domain[0], domain[2]]
    lat = [domain[3], domain[1]]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ['white', 'dimgray', 'darkred', 'yellow', 'limegreen', 'darkgreen'])
    plt.figure(figsize=(int((float(sz) / float(np.shape(data)[0])) * float(np.shape(data)[1])), sz))
    m = Basemap(projection=proj, llcrnrlat=domain[3], urcrnrlat=domain[1], llcrnrlon=domain[0], urcrnrlon=domain[2])
    m.drawcountries()
    m.drawcoastlines()
    m.fillcontinents(color='white',lake_color='lightblue')
    lons, lats = np.meshgrid(np.linspace(np.min(lon), np.max(lon), data.shape[1]+1),
                             np.linspace(np.min(lat), np.max(lat), data.shape[0]+1))
    data = data.astype(float)
    data[data==nodata] = np.nan
    if len(limits) > 1:
        im = m.pcolormesh(lons, lats, np.rot90(np.swapaxes(data*100, 0, 1)), cmap=cmap, latlon=True, vmin=0, vmax=100)
    else:
        im = m.pcolormesh(lons, lats, np.rot90(np.swapaxes(data*100, 0, 1)), cmap=cmap, latlon=True, vmin=0, vmax=100)
    
    cb = m.colorbar(im, 'bottom', size='5%', pad='10%')
    plt.title('Suitability')
    if name == '':
        plt.savefig('test_map.png', dpi=200) 
    else:
        plt.savefig(name, dpi=200)

    plt.close()
    # print('complete!')


def plot_xy_suitability(data, nodata = .0, lon=[-180, 180], lat=[-90, 90], sz=6, limits=[], name='suitability_test'):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.colors
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ['white', 'dimgray', 'darkred', 'yellow', 'limegreen', 'darkgreen'])
    fig, ax = plt.subplots()
    plt.figure(figsize=(sz, int((float(sz) / float(np.shape(data)[0])) * float(np.shape(data)[1]))))
    if len(limits) > 0:
        im = ax.imshow(data*100, cmap=cmap, aspect='equal', vmin=0, vmax=100)
    else:
        im = ax.imshow(data, cmap=cmap, aspect='equal')
    ax.set_aspect('equal')
    plt.title('Suitability')
    fig.colorbar(im)
    plt.close()
    fig.savefig(name+'.png', dpi=300)


def plot_xy_limfact(data, lon=[-180, 180], lat=[-90, 90], sz=6, limits=[], name='limitingfactor_test'):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.colors

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ['red', 'blue'])

    fig, ax = plt.subplots()
    plt.figure(figsize=(sz, int((float(sz) / float(np.shape(data)[0])) * float(np.shape(data)[1]))))
    if len(limits) > 0:
        im = ax.imshow(data, cmap=cmap, aspect='equal', vmin=0, vmax=1)
    else:
        im = ax.imshow(data, cmap=cmap, aspect='equal')
    ax.set_aspect('equal')
    plt.title('Limiting Factor')
    fig.colorbar(im)
    plt.close()
    fig.savefig(name+'.png', dpi=300)


def plot_xy_lengrowperio(data, lon=[-180, 180], lat=[-90, 90], sz=6, limits=[], name='lengthgrowingperiod_test'):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.colors

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ['white', 'green'])

    fig, ax = plt.subplots()
    plt.figure(figsize=(sz, int((float(sz) / float(np.shape(data)[0])) * float(np.shape(data)[1]))))
    if len(limits) > 0:
        im = ax.imshow(data, cmap=cmap, aspect='equal', vmin=0, vmax=1)
    else:
        im = ax.imshow(data, cmap=cmap, aspect='equal')
    ax.set_aspect('equal')
    plt.title('Length of Growing Period')
    fig.colorbar(im)
    plt.close()
    fig.savefig(name+'.png', dpi=300)


def set_start_time():
    return datetime.now()


def get_time_diff(start_time):
    print(f"Time difference: {(datetime.now() - start_time).total_seconds():.5f} seconds")


def plot_1d_data(data, sz=6):
    import numpy as np
    import matplotlib
    matplotlib.use('qtagg')
    import matplotlib.pyplot as plt
    plt.plot(data)
    plt.savefig('p1d.png', dpi=300)


def reshape_as_sqarray(data, dimension):
    import numpy as np
    if len(data)/dimension**2 == 1.0:
        return np.transpose(np.reshape(data, (dimension, dimension)))
    else:
        return None


def plot_plant_params(form_arr, crop, sz=10, no_stops = 25):
    import numpy as np
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(sz, sz*2/3))
    plt.title(crop)
    plt.subplot(4, 4, 1)
    plt.plot(np.linspace(form_arr[0][1], form_arr[0][2], no_stops), form_arr[0][0](np.linspace(form_arr[0][1], form_arr[0][2], no_stops)))
    plt.ylabel('Suitability')
    plt.xlabel('Temperature [°C]')
    plt.subplot(4, 4, 2)
    plt.plot(np.linspace(form_arr[1][1], form_arr[1][2], no_stops), form_arr[1][0](np.linspace(form_arr[1][1], form_arr[1][2], no_stops)))
    plt.ylabel('Suitability')
    plt.xlabel('Precipitation [mm/day]')
    plt.subplot(4, 4, 5)
    plt.plot(np.linspace(form_arr[2][1], form_arr[2][2], no_stops), form_arr[2][0](np.linspace(form_arr[2][1], form_arr[2][2], no_stops)))
    plt.ylabel('Suitability')
    plt.xlabel('Slope [°]')
    plt.subplot(4, 4, 6)
    plt.plot(np.linspace(form_arr[3][1], form_arr[3][2], no_stops), form_arr[3][0](np.linspace(form_arr[3][1], form_arr[3][2], no_stops)))
    plt.ylabel('Suitability')
    plt.xlabel('Soil Depth')
    plt.subplot(4, 4, 9)
    plt.plot(np.linspace(form_arr[4][1], form_arr[4][2], no_stops), form_arr[4][0](np.linspace(form_arr[4][1], form_arr[4][2], no_stops)))
    plt.ylabel('Suitability')
    plt.xlabel('Texture')
    plt.subplot(4, 4, 10)
    plt.plot(np.linspace(form_arr[5][1], form_arr[5][2], no_stops), form_arr[5][0](np.linspace(form_arr[5][1], form_arr[5][2], no_stops)))
    plt.ylabel('Suitability')
    plt.xlabel('Coarse Fragments')
    plt.subplot(4, 4, 11)
    plt.plot(np.linspace(form_arr[6][1], form_arr[6][2], no_stops), form_arr[6][0](np.linspace(form_arr[6][1], form_arr[6][2], no_stops)))
    plt.ylabel('Suitability')
    plt.xlabel('Gypsum')
    plt.subplot(4, 4, 12)
    plt.plot(np.linspace(form_arr[7][1], form_arr[7][2], no_stops), form_arr[7][0](np.linspace(form_arr[7][1], form_arr[7][2], no_stops)))
    plt.ylabel('Suitability')
    plt.xlabel('Base Saturation')
    plt.subplot(4, 4, 13)
    plt.plot(np.linspace(form_arr[8][1], form_arr[8][2], no_stops), form_arr[8][0](np.linspace(form_arr[8][1], form_arr[8][2], no_stops)))
    plt.ylabel('Suitability')
    plt.xlabel('pH')
    plt.subplot(4, 4, 14)
    plt.plot(np.linspace(form_arr[9][1], form_arr[9][2], no_stops), form_arr[9][0](np.linspace(form_arr[9][1], form_arr[9][2], no_stops)))
    plt.ylabel('Suitability')
    plt.xlabel('Organic Carbon')
    plt.subplot(4, 4, 15)
    plt.plot(np.linspace(form_arr[10][1], form_arr[10][2], no_stops), form_arr[10][0](np.linspace(form_arr[10][1], form_arr[10][2], no_stops)))
    plt.ylabel('Suitability')
    plt.xlabel('ELCO')
    plt.subplot(4, 4, 16)
    plt.plot(np.linspace(form_arr[11][1], form_arr[11][2], no_stops), form_arr[11][0](np.linspace(form_arr[11][1], form_arr[11][2], no_stops)))
    plt.ylabel('Suitability')
    plt.xlabel('ESP')
    fig.tight_layout()
    plt.show()