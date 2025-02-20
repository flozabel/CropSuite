#! usr/bin/env python
import sys
import os

try:
    sys.path.append(os.path.dirname(__file__))
    sys.path.append(os.getcwd())
    sys.path.append(os.path.join(os.getcwd(), 'src'))
    sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
except Exception as e:
    print(f"Failed to modify system path: {e}")

from tkinter import * #type:ignore
from tkinter import filedialog
from tkinter import ttk
import tkinter as tk
from CropSuite import run
import os
import platform
import shutil
global plant_window
from src import check_prereqs as cp
from src import read_climate_ini as rci
from src import data_tools as dt
from src import nc_tools as nc
from src import preproc_gui as ppgui
from src import param_gui as pargui
from src import limfact_analyzer as limfa
import numpy as np
from PIL import Image, ImageTk
import subprocess
import re
import matplotlib.pyplot as plt
import rasterio
from rasterio.windows import from_bounds
import matplotlib.colors as clr
import matplotlib.cm as cm
from matplotlib.patches import Patch
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.figure import Figure
from scipy.interpolate import interp2d
from matplotlib.patches import Rectangle
from datetime import datetime
import warnings
import xarray as xr
from src import plant_param_gui
warnings.filterwarnings('ignore')

version = '1.0'
date = '2024-12-12'
current_cfg = ''

plant_param_dir = ''

loading_gif = os.path.join('src', 'loading.gif')

def get_screen_resolution():
    results = str(subprocess.Popen(['system_profiler SPDisplaysDataType'],stdout=subprocess.PIPE, shell=True).communicate()[0])
    res = re.search(r'Resolution: \d* x \d*', results).group(0).split(' ') #type:ignore
    return int(res[1]), int(res[3])
    
class CropSuiteGui:
    global config_ini_val
    def __init__(self, config_ini=''):
        if config_ini == '':
            config_ini = './config.ini'
        self.config_ini_val = config_ini
        self.window = Tk()
        self.window.geometry(f'{540}x{850}')
        self.window.title("CropSuite")
        self.window.resizable(0, 1) #type:ignore

        # Define colors
        self.gray = '#BFBFBF'
        self.red = '#E40808'
        self.yellow = '#E4C308'
        self.green = '#08E426'
        self.invisible = '#F0F0F0'
        self.exit = False

        self.font10 = f'Helvetica 10'
        self.font12 = f'Helvetica 12'
        self.font14 = f'Helvetica 14'
        self.font16 = f'Helvetica 16'
        self.font18 = f'Helvetica 18'
        self.font20 = f'Helvetica 20'

        self.image_unibas = PhotoImage(file=os.path.abspath(os.path.join(os.path.dirname(__file__), 'src', 'unibas.png')), master=self.window)
        #self.image_cgiar = PhotoImage(file=os.path.abspath(os.path.join(os.path.dirname(__file__), 'src', 'cgiar.png')), master=self.window)

        self.setup_ui(config_ini)


    def setup_ui(self, config_path):
        # UI setup
        
        Label(self.window, text='CropSuite\n', font=self.font18 + ' bold').pack()
        Label(self.window, text=f'Version {version}').pack()
        Label(self.window, text=date+'\n').pack()
        Label(self.window, text='Florian Zabel & Matthias Knüttel\n').pack()
        Label(self.window, image=self.image_unibas).pack()
        #Label(self.window, text='\nIn cooperation with:').pack()
        #Label(self.window, image=self.image_cgiar).pack()
        Label(self.window, text='\n© All rights reserved\n').pack()

        frame = Frame(self.window)
        frame.pack()
        cfg = Label(frame, text='  Path to config.ini:', highlightbackground='white')
        cfg.pack(side='left')
        self.config_ini = Label(frame, text = config_path, width=40) # type:ignore
        self.config_ini.pack(side='right')

        empty = Label(self.window, text='', fg=self.yellow, font=self.font12 + ' bold', anchor='w')
        empty.pack()

        #self.libraries = Label(self.window, text='  ☐  Checking required Libraries', fg=self.gray, font=self.font12 + ' bold', anchor='w')
        self.libraries = Label(self.window, text='  ', fg=self.gray, font=self.font12 + ' bold', anchor='w')
        self.libraries.pack(anchor='w')

        #self.check_cfg_ini = Label(self.window, text='  ☐  Checking config.ini', fg=self.gray, font=self.font12 + ' bold', anchor='w')
        self.check_cfg_ini = Label(self.window, text='  ', fg=self.gray, font=self.font12 + ' bold', anchor='w')
        self.check_cfg_ini.pack(anchor='w')

        #self.check_inputs = Label(self.window, text='  ☐  Checking all input files', fg=self.gray, font=self.font12 + ' bold', anchor='w')
        self.check_inputs = Label(self.window, text='  ', fg=self.gray, font=self.font12 + ' bold', anchor='w')
        self.check_inputs.pack(anchor='w')

        #self.all_checked = Label(self.window, text='\n  All requirements successfully checked!', fg=self.invisible, font=self.font12 + ' bold', anchor='w')
        self.all_checked = Label(self.window, text=' ', fg=self.invisible, font=self.font12 + ' bold', anchor='w')
        self.all_checked.pack(anchor='w')

        #self.downscaling = Label(self.window, text='\n  ☑  Downscaling Completed!', fg=self.invisible, font=self.font12 + ' bold', anchor='w')
        self.downscaling = Label(self.window, text=' ', fg=self.invisible, font=self.font12 + ' bold', anchor='w')
        self.downscaling.pack(anchor='w')

        #self.extent = Label(self.window, text='\n  Current Extent:', fg=self.invisible, font=self.font12 + ' bold', anchor='w')
        self.extent = Label(self.window, text=' ', fg=self.invisible, font=self.font12 + ' bold', anchor='w')
        self.extent.pack(anchor='w')

        #self.clim_suit = Label(self.window, text='  ☑  Climate Suitability Calculation Completed!', fg=self.invisible, font=self.font12 + ' bold', anchor='w')
        self.clim_suit = Label(self.window, text='  ', fg=self.invisible, font=self.font12 + ' bold', anchor='w')
        self.clim_suit.pack(anchor='w')

        #self.crop_suit = Label(self.window, text='  ☑  Crop Suitability Calculation Completed!', fg=self.invisible, font=self.font12 + ' bold', anchor='w')
        self.crop_suit = Label(self.window, text='  ', fg=self.invisible, font=self.font12 + ' bold', anchor='w')
        self.crop_suit.pack(anchor='w')

        #self.merge = Label(self.window, text='  ☑  Merging files!', fg=self.invisible, font=self.font12 + ' bold', anchor='w')
        self.merge = Label(self.window, text='  ', fg=self.invisible, font=self.font12 + ' bold', anchor='w')
        self.merge.pack(anchor='w')

        #self.finish = Label(self.window, text='  Completed!', fg=self.invisible, font=self.font14 + ' bold', anchor='w')
        self.finish = Label(self.window, text='  ', fg=self.invisible, font=self.font14 + ' bold', anchor='w')
        self.finish.pack(anchor='w')

        # self.setup_buttons()
        self.run_proc()
        self.window.mainloop()
        
    def select_cfg_file(self):
        config_file = ''
        while not str(config_file).endswith('.ini'):
            print('Select config.ini')
            config_file = filedialog.askopenfilename(initialdir=os.path.join(os.getcwd()), title='Select config.ini', defaultextension='.ini')
        self.set_config_ini(config_file)

    def run_proc(self):
        self.libraries.config(text='  ☐  Checking required Libraries', fg=self.yellow, font=self.font12 + ' bold', anchor='w')
        self.window.update()
        run(silent_mode=True, gui=self)

    def restart(self):
        self.window.destroy()
        main_gui()

    def viewer(self):
        self.window.destroy()
        cfg = self.config_ini_val
        viewer_gui(cfg)

    def setup_buttons(self):
        self.but_frame = Frame(self.window)
        self.but_frame.pack(anchor='w', fill='x', expand=True)

        self.restart = Button(self.window, text='Restart', command=self.restart, font=self.font16 + ' bold', highlightbackground='white') #type:ignore
        self.restart.pack(side='left', fill=X, expand=True)

        self.viewer = Button(self.window, text='Data Viewer', command=self.viewer, font=self.font16 + ' bold', highlightbackground='white') #type:ignore
        self.viewer.pack(side='left', fill=X, expand=True)

        self.but_exit = Button(self.window, text='EXIT', command=self.exit_all, font=self.font16 + ' bold', highlightbackground='white')
        self.but_exit.pack(side='left', fill=X, expand=True)

    def exit_all(self):
        self.exit = True
        sys.exit()

    def set_config_ini(self, ini_path):
        self.config_ini.config(text=ini_path)
        self.window.update()

    def get_config_path(self):
        return self.config_ini.cget('text')

    def set_libraries_true(self):
        self.libraries.config(text='  ☑  Required Libraries checked', fg=self.green, font=self.font12 + ' bold', anchor='w')
        self.check_cfg_ini.config(text='  ☐  Checking config.ini', fg=self.yellow, font=self.font12 + ' bold', anchor='w')
        self.window.update()

    def set_libraries_false(self):
        self.libraries.config(text='  ✗  Missing Libraries', fg=self.red, font=self.font12 + ' bold', anchor='w')
        self.window.update()

    def check_cfg_ini_true(self):
        self.check_cfg_ini.config(text='  ☑  config.ini checked', fg=self.green, font=self.font12 + ' bold', anchor='w')
        self.check_inputs.config(text='  ☐  Checking all input files', fg=self.yellow, font=self.font12 + ' bold', anchor='w')
        self.window.update()

    def check_cfg_ini_false(self):
        self.check_cfg_ini.config(text='  ✗  config.ini faulty', fg=self.red, font=self.font12 + ' bold', anchor='w')
        self.window.update()

    def check_inpts_true(self):
        self.check_inputs.config(text='  ☑  All input files checked', fg=self.green, font=self.font12 + ' bold', anchor='w')
        self.all_checked.config(text='\n  All requirements successfully checked!\n', fg=self.green, font=self.font12 + ' bold', anchor='w')
        self.downscaling.config(text='  ☐  Downscaling', fg=self.gray, font=self.font12 + ' bold', anchor='w')
        self.window.update()

    def check_inpts_false(self):
        self.check_inputs.config(text='  ✗  One or more input files faulty', fg=self.red, font=self.font12 + ' bold', anchor='w')
        self.window.update()

    def set_finish(self):
        self.clim_suit.config(text = '  ☑  Climate Suitability Calculation Completed!', fg=self.green)
        self.crop_suit.config(text='  ☑  Crop Suitability Calculation Completed!', fg=self.green)
        self.downscaling.config(text='  ☑  Downscaling Completed!', fg=self.green)
        self.extent.config(text='  ☑  All extents processed!', fg=self.green)
        self.finish.config(text='\n  Completed!', fg=self.green, font='Helvetica 12 bold', anchor='w')
        self.setup_buttons()
        self.window.update()

    def set_extent(self, completed, extent=[], no=0, out_of=0):
        if not completed:
            top_str = str(extent[0])+'°N' if extent[0] >= 0 else str(abs(extent[0]))+'°S'
            bot_str = str(extent[2])+'°N' if extent[2] >= 0 else str(abs(extent[2]))+'°S'
            left_str = str(extent[1])+'°E' if extent[1] >= 0 else str(abs(extent[1]))+'°W'
            right_str = str(extent[3])+'°E' if extent[3] >= 0 else str(abs(extent[3]))+'°W'
            extent_str = f'{top_str} {left_str} ☐ {bot_str} {right_str}'
        else:
            extent_str = ''

        if completed:
            self.extent.config(text='  All extents processed!', fg=self.green)
        if extent != [] and out_of > 0:
            self.extent.config(text=f'  Current Extent: {extent_str} - {no} out of {out_of}', fg=self.yellow, font=self.font12 + ' bold', anchor='w')
        if extent != [] and out_of == 0:
            self.extent.config(text=f'  Current Extent: {extent_str}', fg=self.yellow, font=self.font12 + ' bold', anchor='w')
        if extent == [] and out_of > 0:
            self.extent.config(text=f'  Current Extent: {no} out of {out_of}', fg=self.yellow, font=self.font12 + ' bold', anchor='w')
        if extent == [] and out_of == 0:
            self.extent.config(text=f'  Current Extent', fg=self.yellow, font=self.font12 + ' bold', anchor='w')
        self.window.update()

    def set_downscaling(self, completed=False, extent=[], no=0, out_of=0):
        extent_str = ' - '+f'{extent}'
        oof = f'- {no} out of {out_of}'
        if completed:
            text = '  ☑  Downscaling Completed!'
        else:
            if extent != []:
                if out_of == 0:
                    text = f'  ☐  Downscaling in Progress {extent_str}'
                else:
                    text = f'  ☐  Downscaling in Progress {extent_str} {oof}'
            else:
                text = f'  ☐  Downscaling in Progress {oof}'
        fg = self.green if completed else self.yellow
        self.downscaling.config(text=text, fg=fg)
        self.window.update()

    def set_climsuit(self, completed=False, extent=[], no=0, out_of=0, started=False):
        extent_str = ' - '+f'{extent}'
        oof = f'- {no} out of {out_of}'
        prog_text = '' if not started else ' in Progress'
        if completed:
            text = '  ☑  Climate Suitability Calculation Completed!'
        else:
            if extent != []:
                if out_of == 0:
                    text = f'  ☐  Climate Suitability Calculation{prog_text} {extent_str}'
                else:
                    text = f'  ☐  Climate Suitability Calculation{prog_text} {extent_str} {oof}'
            else:
                text = f'  ☐  Climate Suitability Calculation{prog_text}'
        fg = self.green if completed else self.yellow
        fg = self.gray if not started else fg
        self.clim_suit.config(text=text, fg=fg)
        self.window.update()


    def set_cropsuit(self, completed=False, no=0, out_of=0, started=False):
        oof = f'- {no} out of {out_of}'
        prog_text = '' if not started else ' in Progress'
        if completed:
            text = '  ☑  Crop Suitability Calculation Completed!'
        else:
            text = f'  ☐  Crop Suitability Calculation{prog_text}'
        fg = self.green if completed else self.yellow
        fg = self.gray if not started else fg
        self.crop_suit.config(text=text, fg=fg)
        self.window.update()

    def set_merge(self, completed=False, merge_required=True, started=False):
        if completed and merge_required:
            text = '  ☑  Merging and Cleaning Up finished!'
        if completed and not merge_required:
            text = f'  ☑  Cleaning Up finished!'
        if not completed and merge_required:
            text = '  ☐  Merging and Cleaning Up in Progress'
        if not completed and not merge_required:
            text = '  ☐  Cleaning Up in Progress'

        if not started:
            fg = self.gray
        else:
            if completed:
                fg = self.green
            else:
                fg = self.yellow
        self.merge.config(text=text, fg=fg)
        self.window.update()

    def update(self):
        if self.exit:
            return False
        self.window.update()
        return True

def loading_gui():
    loading_window = Tk()
    loading_window.focus_force()
    loading_window.title("CropSuite")
    loading_window.resizable(0, 0) #type:ignore
    loading_window.overrideredirect(1) #type:ignore
    
    x, y = 500, 500
    loading_window.geometry(f'{x}x{y}+{(loading_window.winfo_screenwidth() - x) // 2}+{(loading_window.winfo_screenheight() - y) // 2}')

    splash_image = Image.open(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src', 'splashscreen.png')))
    splash_image = splash_image.resize((500, 500))
    splash_image = ImageTk.PhotoImage(splash_image)
    Label(loading_window, image=splash_image).pack() #type:ignore

    os.makedirs(os.path.abspath(os.path.join(os.path.dirname(__file__), 'plant_params')), exist_ok=True)
    os.makedirs(os.path.abspath(os.path.join(os.path.dirname(__file__), 'results')), exist_ok=True)
    os.makedirs(os.path.abspath(os.path.join(os.path.dirname(__file__), 'temp')), exist_ok=True)
    os.makedirs(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')), exist_ok=True)

    loading_window.after(5000, lambda: loading_window.destroy())
    loading_window.mainloop()
    main_gui()

def get_tifs(dir, compare_val):
    if compare_val == 0:
        files_to_remove = ['all_climlim_factors.tif', 'all_suitability_vals.tif', 'limiting_factor.tif']
    else:
        files_to_remove = ['all_climlim_factors.tif', 'all_suitability_vals.tif', 'limiting_factor.tif', 'crop_limiting_factor.tif']
    tifs = sorted([f.lower() for f in os.listdir(dir) if str(f).endswith('.tif') or str(f).endswith('.nc')])
    return [file for file in tifs if file not in files_to_remove]

def get_min_area(x_min1, x_max1, y_min1, y_max1, x_min2, x_max2, y_min2, y_max2):
    x_min_overlap, x_max_overlap = max(x_min1, x_min2), min(x_max1, x_max2)
    y_min_overlap, y_max_overlap = max(y_min1, y_min2), min(y_max1, y_max2)
    
    if x_min_overlap >= x_max_overlap or y_min_overlap >= y_max_overlap:
        raise ValueError("The bounding boxes do not overlap.")
    
    return x_min_overlap, x_max_overlap, y_min_overlap, y_max_overlap

def read_geotiff(geotiff, extent=[0, 0, 0, 0]):
    with rasterio.open(geotiff) as src:
        if extent != [0, 0, 0, 0]:
            x_min, x_max, y_min, y_max = extent
            window = from_bounds(x_min, y_min, x_max, y_max, src.transform)
            data = src.read(1, window=window)
        else:
            data = src.read(1)
            transform = src.transform
            x_min, y_max = transform * (0, 0)
            x_max, y_min = transform * (src.width, src.height)
        nodata = src.nodata
    data = data.astype(float)
    return data, x_min, x_max, y_min, y_max, nodata

def select_layer(data, label):
    layer_win = Tk()
    layer_win.title('Select Data Layer')
    layer_win.resizable(0, 0) #type:ignore
    layer_win.focus_force()
    x, y = 200, 100
    layer_win.geometry(f'{x}x{y}+{(layer_win.winfo_screenwidth() - x) // 2}+{(layer_win.winfo_screenheight() - y) // 2}')

    layers = np.linspace(1, data.shape[2]+1, data.shape[2]).astype(int)
    frm2 = Frame(layer_win)
    frm2.pack(anchor='w', padx=5, pady=5, fill='x')
    layer_val = IntVar(layer_win, layers[0])

    layer_lab = Label(frm2, text=f'{label}: ')
    layer_sel = ttk.Combobox(frm2, textvariable=layer_val, width=20)
    layer_sel['values'] = layers

    layer_lab.pack(side='left')
    layer_sel.pack(side='right')

    def ret_val():
        global value
        value = layer_val.get()
        layer_win.destroy()
        return value
    
    Button(layer_win, text='Ok', command=ret_val).pack(pady=5,fill='x')
    layer_win.wait_window(layer_win)
    return value

def read_netcdf(netcdf_file, layer=[0, 0], extent=[0, 0, 0, 0]):
    ds = xr.open_dataset(netcdf_file)
    nodata = int(ds.attrs['_FillValue'])
    data_variable = None
    for var_name in ds.data_vars:
        data_variable = var_name
        break
    if data_variable is None:
        raise ValueError("No data variables found in the NetCDF file.")
    
    if layer[1] > 0:
        nodata_mask = ds[data_variable].isel(day=1).values == nodata
        if 'downscaled_temperature' in os.path.basename(netcdf_file):
            data = ds[data_variable].sel(day=slice(layer[0], layer[1]))
            data = np.nanmean(data, axis=2)
            data[nodata_mask] = nodata
        elif 'downscaled_precipitation' in os.path.basename(netcdf_file):
            data = ds[data_variable].sel(day=slice(layer[0], layer[1]))
            data = np.nansum(data, axis=2)
            data[nodata_mask] = nodata
    else:
        if extent != [0, 0, 0, 0]:
            try:
                data = ds[data_variable].sel(lat=slice(extent[3], extent[2])).sel(lon=slice(extent[0], extent[1])).values
            except:
                data = ds[data_variable].sel(latitude=slice(extent[2], extent[3])).sel(longitude=slice(extent[0], extent[1])).values
        else:
            data = ds[data_variable].values

    if len(data.shape) == 3 and layer[1] == 0:
        data = data[..., select_layer(data, 'DoY')]

    try:
        lon, lat = ds.coords['lon'].values, ds.coords['lat'].values
    except:
        try:
            lon, lat = ds.coords['longitude'].values, ds.coords['latitude'].values
        except:
            lon, lat = np.asarray([-180, 180]), np.asarray([-90, 90])
    x_min, x_max = lon.min(), lon.max()
    y_min, y_max = lat.min(), lat.max()
    
    return data, x_min, x_max, y_min, y_max, nodata

def read_data_file(data_file, layer, comp_file='', extent=[0, 0, 0, 0]):

    if os.path.basename(data_file) == 'downscaled_temperature':
        data_file = [os.path.join(os.path.dirname(data_file), f'ds_temp_{i}.nc') for i in range(layer[0], layer[1])]
        extension = '.nc'
    elif os.path.basename(data_file) == 'downscaled_precipitation':
        data_file = [os.path.join(os.path.dirname(data_file), f'ds_prec_{i}.nc') for i in range(layer[0], layer[1])]
        extension = '.nc'
    else:
        _, extension = os.path.splitext(data_file)

    if extension in ['.tif', '.tiff']:
        if comp_file == '':
            return read_geotiff(geotiff = data_file, extent=extent)
        else:
            data, x_min, x_max, y_min, y_max, nodata = read_geotiff(geotiff=data_file, extent=extent)
            _, comp_x_min, comp_x_max, comp_y_min, comp_y_max, _ = read_geotiff(geotiff=comp_file, extent=extent)
            x_min, x_max, y_min, y_max = get_min_area(x_min, x_max, y_min, y_max, comp_x_min, comp_x_max, comp_y_min, comp_y_max)
            comp_data, _ = dt.load_specified_lines(comp_file, [y_min, x_min, y_max, x_max], False)
            comp_data = comp_data.astype(float)
            if extent != [0, 0, 0, 0]:
                return data, comp_data, extent[0], extent[1], extent[2], extent[3], nodata
            else:
                return data, comp_data, x_min, x_max, y_min, y_max, nodata
    elif extension in ['.nc', '.nc4', '.cdf']:
        if comp_file == '':
            if isinstance(data_file, list):
                dat, x_min, x_max, y_min, y_max, nodata = read_netcdf(netcdf_file = data_file[0], extent=extent)
                arr = np.empty((dat.shape[0], dat.shape[1], layer[1]-layer[0]), dtype=dat.dtype)
                for idx, fn in enumerate(data_file):
                    arr[..., idx] = read_netcdf(netcdf_file = fn, extent=extent)[0]
                arr = np.mean(arr, axis=2) if os.path.basename(data_file[0]).startswith('ds_temp_') else np.sum(arr, axis=2)
                arr = arr.astype(float)
                if extent != [0, 0, 0, 0]:
                    return arr, extent[0], extent[1], extent[2], extent[3], nodata
                else:
                    return arr, x_min, x_max, y_min, y_max, nodata
            else:
                arr, x_min, x_max, y_min, y_max, nodata = read_netcdf(netcdf_file = data_file, layer=layer, extent=extent)
                if extent != [0, 0, 0, 0]:
                    return arr, extent[0], extent[1], extent[2], extent[3], nodata
                else:
                    return arr, x_min, x_max, y_min, y_max, nodata
        else:
            data, x_min, x_max, y_min, y_max, nodata = read_netcdf(netcdf_file = data_file, layer=layer, extent=extent)
            _, comp_x_min, comp_x_max, comp_y_min, comp_y_max, comp_nodata = read_netcdf(netcdf_file = comp_file, layer=layer, extent=extent)
            x_min, x_max, y_min, y_max = get_min_area(x_min, x_max, y_min, y_max, comp_x_min, comp_x_max, comp_y_min, comp_y_max)
            comp_data, _ = nc.read_area_from_netcdf(comp_file, [y_max, x_min, y_min, x_max], variable='', day_range=layer)

            nodata_mask = comp_data[..., 0] == comp_nodata
            if 'downscaled_temperature' in os.path.basename(comp_file):
                comp_data = np.nanmean(comp_data, axis=2)
                comp_data[nodata_mask] = nodata
            elif 'downscaled_precipitation' in os.path.basename(comp_file):
                comp_data = np.nansum(comp_data, axis=2)
                comp_data[nodata_mask] = nodata

            comp_data = comp_data.astype(float)
            return data, comp_data, x_min, x_max, y_min, y_max, nodata

def resize_array_interp(array, new_shape, limit=(-9999, -9999), dtype=np.float16):
    h, w = array.shape
    interp_func = interp2d(np.arange(w), np.arange(h), array, kind='linear')
    ret = interp_func(np.linspace(0, w - 1, int(new_shape[1])), np.linspace(0, h - 1, int(new_shape[0])))
    if limit[0] != -9999:
        ret[ret<=-limit[0]] = 0
    if limit[1] != -9999:
        ret[ret>=limit[1]] = 0
    return ret.astype(dtype)

def resize_array(array):
    non_nan_indices = np.where(~np.isnan(array))
    non_nan_values = array[non_nan_indices]
    non_nan_row_indices, non_nan_col_indices = non_nan_indices
    new_shape = (array.shape[0] // 4, array.shape[1] // 4)
    smaller_array = np.full(new_shape, np.nan)
    smaller_array[non_nan_row_indices // 4, non_nan_col_indices // 4] = non_nan_values
    return smaller_array

def classify_2d_array(data, num_classes=4, min_val=0, max_val=100):
    nan_mask = np.isnan(data)
    boundaries = np.linspace(min_val, max_val, num_classes)
    classified_data = np.zeros_like(data, dtype=float)
    for i, boundary in enumerate(boundaries):
        classified_data[data > boundary] = i + 1
    classified_data[nan_mask] = np.nan
    return classified_data

def populate_treeview(tree, parent, path):
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            item_id = tree.insert(parent, 'end', text=item, open=False)
            populate_treeview(tree, item_id, item_path)

def show_image_centered(canvas, master, img_path):
    image = Image.open(img_path)
    canvas_width = canvas.winfo_width()
    canvas_height = canvas.winfo_height()
    image_ratio = image.width / image.height
    canvas_ratio = canvas_width / canvas_height
    if image_ratio > canvas_ratio:
        new_width = canvas_width
        new_height = int(new_width / image_ratio)
    else:
        new_height = canvas_height
        new_width = int(new_height * image_ratio)
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    image_tk = ImageTk.PhotoImage(resized_image, master=master) #type:ignore
    x_center = canvas_width // 2
    y_center = canvas_height // 2
    canvas.create_image(x_center, y_center, image=image_tk, anchor='center')
    canvas.image = image_tk

def plot_data(data, filename, x_min, x_max, y_min, y_max, projection=ccrs.PlateCarree, classified=False, colormap='viridis'):
    fn, _ = os.path.splitext(os.path.basename(filename))

    if isinstance(colormap, list):
        cmap = clr.LinearSegmentedColormap.from_list('', colormap)
        color_list = colormap
    else:
        cmap = cm.get_cmap(colormap)
        color_list = cmap(np.linspace(0, 1, 254))

    if fn == 'downscaled_temperature':
        data[data <= -500] = np.nan
        data /= 10
        minimum, maximum = 0, int(np.nanpercentile(data, 98))
        if classified:
            num_classes = 7
            data = classify_2d_array(data, num_classes, min_val=minimum, max_val=maximum)
            step = (maximum - minimum) / num_classes
            #labels = ['-25 - -15 °C', '-15 - -5 °C', '-5 - 5 °C', '5 - 15 °C', '15 - 25 °C', '25 - 35 °C', '35 - 45 °C']
            labels = [f"{minimum + i * step:.1f} - {minimum + (i + 1) * step:.1f} °C" for i in range(num_classes)]
            color_list = cmap(np.linspace(0, 1, len(labels)))
            cmap = clr.LinearSegmentedColormap.from_list('', color_list)
            #color_list = ['rebeccapurple', 'mediumblue', 'turquoise', 'lightgreen', 'orange', 'firebrick', 'saddlebrown']
            #cmap = clr.LinearSegmentedColormap.from_list('', color_list)
            color_list = cmap(np.linspace(0, 1, len(labels)))
            VariableLimits = np.arange(len(labels))
            norm = clr.BoundaryNorm(np.append(VariableLimits, 1000), ncolors=256)
            #patches = [Patch(color=color, label=label) for label, color in zip(labels, color_list)] 
            patches = [Patch(color=color, label=label) for label, color in zip(labels, color_list)] 
        else:
            #color_list = ['rebeccapurple', 'mediumblue', 'turquoise', 'lightgreen', 'orange', 'firebrick', 'saddlebrown']
            #cmap = clr.LinearSegmentedColormap.from_list('', color_list)
            min_val = int(np.nanpercentile(data, 1))
            max_val = int(np.nanpercentile(data, 99))
            label = 'Temperature [°C]'

    elif fn == 'downscaled_precipitation':
        data = data / 10
        data[data <= 0] = np.nan
        minimum, maximum = 0, int(np.nanpercentile(data, 98))
        if classified:
            data = classify_2d_array(data, 5, min_val=0, max_val=maximum)
            step = maximum / 5
            labels = [f'< {step:.0f} mm'] + [f'{int(step * i):.0f} mm' for i in range(2, 5)] + [f'> {int(step * 5):.0f} mm']
            #labels = ['< 500 mm', '1000 mm', '1500 mm', '2000 mm', '> 2500 mm']  if maximum >= 500 else ['< 100 mm', '200 mm', '300 mm', '400 mm', '> 500 mm'] #type:ignore
            color_list = cmap(np.linspace(0, 1, len(labels)))
            cmap = clr.LinearSegmentedColormap.from_list('', color_list)
            #
            # color_list = ['aquamarine', 'darkturquoise', 'cornflowerblue', 'mediumblue', 'darkviolet']
            #cmap = clr.LinearSegmentedColormap.from_list('', color_list)
            VariableLimits = np.arange(len(labels))
            norm = clr.BoundaryNorm(np.append(VariableLimits, 1000), ncolors=256)
            patches = [Patch(color=color, label=label) for label, color in zip(labels, color_list)] 
        else:
            #color_list = ['aquamarine', 'darkturquoise', 'cornflowerblue', 'mediumblue', 'darkviolet']
            #cmap = clr.LinearSegmentedColormap.from_list('', color_list)
            min_val = minimum
            max_val = maximum
            label = 'Precipitation [mm]'

    elif fn == 'base_saturation_combined':
        data[np.isnan(data)] = np.nan 
        if classified:
            labels = ['0 - 10 %', '10 - 20 %', '20 - 30 %', '30 - 40 %', '40 - 50 %', '50 - 60 %', '60 - 70 %', '70 - 80 %', '80 - 90 %', '90 - 100 %']
            color_list = cmap(np.linspace(0, 1, len(labels)))
            cmap = clr.LinearSegmentedColormap.from_list('', color_list)
            #color_list = ['#563209', '#50380b', '#494b16', '#3c6025', '#266f30', '#0a713b', '#00684a', '#005856', '#00475e', '#043863']
            data = classify_2d_array(data, len(labels), min_val=0, max_val=100)
            #cmap = clr.LinearSegmentedColormap.from_list('', color_list)
            VariableLimits = np.arange(len(labels))
            norm = clr.BoundaryNorm(np.append(VariableLimits, 1000), ncolors=256)
            patches = [Patch(color=color, label=label) for label, color in zip(labels, color_list)] 
        else:
            #color_list = ['#563209', '#187334', '#043863']
            #cmap = clr.LinearSegmentedColormap.from_list('', color_list)
            min_val = 0
            max_val = 100
            label = 'Base Saturation [%]'
    
    elif fn == 'coarse_fragments_combined':
        data[np.isnan(data)] = np.nan 
        if classified:
            labels = ['0 - 10 %', '10 - 20 %', '20 - 30 %', '30 - 40 %', '40 - 50 %']
            color_list = cmap(np.linspace(0, 1, len(labels)))
            cmap = clr.LinearSegmentedColormap.from_list('', color_list)
            #color_list = ['darkgreen', 'limegreen', 'yellow', 'red', 'darkred']
            data = classify_2d_array(data, len(labels), min_val=0, max_val=50)
            #cmap = clr.LinearSegmentedColormap.from_list('', color_list)
            VariableLimits = np.arange(len(labels))
            norm = clr.BoundaryNorm(np.append(VariableLimits, 1000), ncolors=256)
            patches = [Patch(color=color, label=label) for label, color in zip(labels, color_list)] 
        else:
            #color_list = ['darkgreen', 'limegreen', 'yellow', 'red', 'darkred']
            #cmap = clr.LinearSegmentedColormap.from_list('', color_list)
            min_val = 0
            max_val = 50
            label = 'Coarse Fragments [%]'

    elif fn == 'gypsum_combined':
        labels = ['0 %', '1 %', '2 %', '3 %', '>4 %']
        color_list = cmap(np.linspace(0, 1, len(labels)))
        cmap = clr.LinearSegmentedColormap.from_list('', color_list)
        #color_list = ['darkgreen', 'limegreen', 'yellow', 'red', 'darkred']
        data = classify_2d_array(data, len(labels), min_val=0, max_val=5)
        #cmap = clr.LinearSegmentedColormap.from_list('', color_list)
        VariableLimits = np.arange(len(labels))
        norm = clr.BoundaryNorm(np.append(VariableLimits, 1000), ncolors=256)
        patches = [Patch(color=color, label=label) for label, color in zip(labels, color_list)] 

    elif fn == 'ph_combined':
        data[np.isnan(data)] = np.nan   
        if classified:
            labels = ['4', '5', '6', '7', '8', '9']
            color_list = cmap(np.linspace(0, 1, len(labels)))
            cmap = clr.LinearSegmentedColormap.from_list('', color_list)
            #color_list = ['darkcyan', 'aquamarine', 'darkgreen', 'limegreen', 'coral', 'darkred']
            data = classify_2d_array(data, len(labels), min_val=4, max_val=9)
            #cmap = clr.LinearSegmentedColormap.from_list('', color_list)
            VariableLimits = np.arange(len(labels))
            norm = clr.BoundaryNorm(np.append(VariableLimits, 1000), ncolors=256)
            patches = [Patch(color=color, label=label) for label, color in zip(labels, color_list)] 
        else:
            #color_list = ['darkcyan', 'aquamarine', 'darkgreen', 'limegreen', 'coral', 'darkred']
            #cmap = clr.LinearSegmentedColormap.from_list('', color_list)
            min_val = 4
            max_val = 9
            label = 'pH'

    elif fn == 'salinity_combined':
        data[np.isnan(data)] = np.nan   
        data *= 4
        if classified:
            labels = ['0 ds/m', '4 ds/m', '8 ds/m', '12 ds/m', '>16 ds/m']
            color_list = cmap(np.linspace(0, 1, len(labels)))
            cmap = clr.LinearSegmentedColormap.from_list('', color_list)
            #color_list = ['darkgreen', 'limegreen', 'yellow', 'orange', 'darkred']
            data = classify_2d_array(data, len(labels), min_val=0, max_val=16)
            #cmap = clr.LinearSegmentedColormap.from_list('', color_list)
            VariableLimits = np.arange(len(labels))
            norm = clr.BoundaryNorm(np.append(VariableLimits, 1000), ncolors=256)
            patches = [Patch(color=color, label=label) for label, color in zip(labels, color_list)] 
        else:
            #color_list = ['darkgreen', 'limegreen', 'yellow', 'orange', 'darkred']
            #cmap = clr.LinearSegmentedColormap.from_list('', color_list)
            min_val = 0
            max_val = 16
            label = 'Salinity [ds/m]'

    elif fn == 'slope_combined':
        data[data <= 0] = np.nan   
        if classified:
            labels = ['0°', '5°', '10°', '>15°']
            color_list = cmap(np.linspace(0, 1, len(labels)))
            cmap = clr.LinearSegmentedColormap.from_list('', color_list)
            #color_list = ['darkgreen', 'limegreen', 'orange', 'darkred']
            data = classify_2d_array(data, len(labels), min_val=0, max_val=15)
            #cmap = clr.LinearSegmentedColormap.from_list('', color_list)
            VariableLimits = np.arange(len(labels))
            norm = clr.BoundaryNorm(np.append(VariableLimits, 1000), ncolors=256)
            patches = [Patch(color=color, label=label) for label, color in zip(labels, color_list)] 
        else:
            #color_list = ['darkgreen', 'limegreen', 'yellow', 'orange', 'darkred']
            #cmap = clr.LinearSegmentedColormap.from_list('', color_list)
            min_val = 0
            max_val = 15
            label = 'Slope [°]'
    
    elif fn == 'sodicity_combined':
        data[np.isnan(data)] = np.nan   
        if classified:
            labels = ['0 %', '5 %', '>10 %']
            color_list = cmap(np.linspace(0, 1, len(labels)))
            cmap = clr.LinearSegmentedColormap.from_list('', color_list)
            #color_list = ['darkgreen', 'limegreen', 'darkred']
            data = classify_2d_array(data, len(labels), min_val=0, max_val=10)
            #cmap = clr.LinearSegmentedColormap.from_list('', color_list)
            VariableLimits = np.arange(len(labels))
            norm = clr.BoundaryNorm(np.append(VariableLimits, 1000), ncolors=256)
            patches = [Patch(color=color, label=label) for label, color in zip(labels, color_list)] 
        else:
            #color_list = ['darkgreen', 'limegreen', 'darkred']
            #cmap = clr.LinearSegmentedColormap.from_list('', color_list)
            min_val = 0
            max_val = 15
            label = 'Sodicity'
    
    elif fn == 'soil_organic_carbon_combined':
        data[np.isnan(data)] = np.nan   
        if classified:
            labels = ['0 %', '2.5 %', '5 %', '7.5 %', '> 10 %']
            color_list = cmap(np.linspace(0, 1, len(labels)))
            cmap = clr.LinearSegmentedColormap.from_list('', color_list)
            #color_list = ['darkred', 'yellow', 'limegreen', 'darkgreen', 'darkslategrey']
            data = classify_2d_array(data, len(labels), min_val=0, max_val=10)
            #cmap = clr.LinearSegmentedColormap.from_list('', color_list)
            VariableLimits = np.arange(len(labels))
            norm = clr.BoundaryNorm(np.append(VariableLimits, 1000), ncolors=256)
            patches = [Patch(color=color, label=label) for label, color in zip(labels, color_list)] 
        else:
            #color_list = ['darkred', 'yellow', 'limegreen', 'darkgreen', 'darkslategrey']
            #cmap = clr.LinearSegmentedColormap.from_list('', color_list)
            min_val = 0
            max_val = 10
            label = 'Soil Organic Carbon [%]'
    
    elif fn == 'soildepth_combined':
        data[np.isnan(data)] = np.nan   
        if classified:
            labels = ['0.5 m', '1 m', '1.5 m', '2 m %', '>2.5 m']
            color_list = cmap(np.linspace(0, 1, len(labels)))
            cmap = clr.LinearSegmentedColormap.from_list('', color_list)
            #color_list = ['darkred', 'yellow', 'limegreen', 'darkgreen', 'darkslategrey']
            data = classify_2d_array(data, len(labels), min_val=0, max_val=3)
            #cmap = clr.LinearSegmentedColormap.from_list('', color_list)
            VariableLimits = np.arange(len(labels))
            norm = clr.BoundaryNorm(np.append(VariableLimits, 1000), ncolors=256)
            patches = [Patch(color=color, label=label) for label, color in zip(labels, color_list)] 
        else:
            #color_list = ['darkred', 'yellow', 'limegreen', 'darkgreen', 'darkslategrey']
            #cmap = clr.LinearSegmentedColormap.from_list('', color_list)
            min_val = 0
            max_val = 3
            label = 'Soildepth [m]'

    elif fn == 'texture_combined':
        data[data == 0] = np.nan   
        labels = ['Sand', 'Loamy Sand', 'Sandy Loam', 'Sandy Clay Loam', 'Loam', 'Sandy Clay', 'Silty Loam', 'Silt', 'Clay Loam', 'Silty Clay Loam', 'Clay', 'Silty Clay', 'Heavy Clay']
        color_list = cmap(np.linspace(0, 1, len(labels)))
        cmap = clr.LinearSegmentedColormap.from_list('', color_list)
        #color_list = ['#f3f3d9', '#d5bb76', '#a78834', '#a17a10', '#76531b', '#987a4d', '#736142', '#8a652b', '#8a4f2b', '#a66b4a', '#a6494a', '#8f4a80', '#64303b']
        data = classify_2d_array(data, len(labels), min_val=1, max_val=13)
        #cmap = clr.LinearSegmentedColormap.from_list('', color_list)
        VariableLimits = np.arange(len(labels))
        norm = clr.BoundaryNorm(np.append(VariableLimits, 1000), ncolors=256)
        patches = [Patch(color=color, label=label) for label, color in zip(labels, color_list)] 

    elif fn == 'climate_suitability' or fn == 'crop_suitability' or fn == 'soil_suitability':
        if classified:
            labels = ['Unsuitable', 'Marginally Suitable', 'Moderately Suitable', 'Highly Suitable', 'Very high Suitable']
            color_list = cmap(np.linspace(0, 1, len(labels)))
            cmap = clr.LinearSegmentedColormap.from_list('', color_list)
            data = classify_2d_array(data, len(labels))
            #color_list = ['white', '#a54200', '#ffd700', '#72e332', '#006400']
            #cmap = clr.LinearSegmentedColormap.from_list('', color_list)
            VariableLimits = np.arange(len(labels))
            norm = clr.BoundaryNorm(np.append(VariableLimits, 1000), ncolors=256)
            patches = [Patch(color=color, label=label) for label, color in zip(labels, color_list)] 
        else:
            #color_list = ['white', 'darkgray', 'darkred', 'yellow', 'greenyellow', 'limegreen', 'darkgreen', 'darkslategray']
            #cmap = clr.LinearSegmentedColormap.from_list('', color_list)
            min_val = 0
            max_val = 100
            label = 'Suitability'

    elif fn == 'multiple_cropping_sum' or fn == 'climate_suitability_mc':
        if classified:
            data = classify_2d_array(data, 5, min_val=0, max_val=300)
            labels = ['Unsuitable', 'Marginally Suitable', 'Moderately Suitable', 'Highly Suitable', 'Very high Suitable']
            color_list = cmap(np.linspace(0, 1, len(labels)))
            cmap = clr.LinearSegmentedColormap.from_list('', color_list)
            #color_list = ['white', '#a54200', '#ffd700', '#72e332', '#006400']
            #cmap = clr.LinearSegmentedColormap.from_list('', color_list)
            VariableLimits = np.arange(len(labels))
            norm = clr.BoundaryNorm(np.append(VariableLimits, 1000), ncolors=256)
            patches = [Patch(color=color, label=label) for label, color in zip(labels, color_list)] 
        else:
            #color_list = ['white', 'darkgray', 'darkred', 'yellow', 'greenyellow', 'limegreen', 'darkgreen', 'darkslategray']
            #cmap = clr.LinearSegmentedColormap.from_list('', color_list)
            min_val = 0
            max_val = 300
            label = 'Suitability'
    
    elif fn in ['optimal_sowing_date', 'optimal_sowing_date_mc_first', 'optimal_sowing_date_mc_second', 'optimal_sowing_date_mc_third', 'optimal_sowing_date_vernalization', 'optimal_sowing_date_with_vernalization', 'start_growing_cycle_after_vernalization']:
        data[data == 0] = np.nan
        if classified:
            data = classify_2d_array(data, 12, min_val=0, max_val=365)
            labels = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
            color_list = cmap(np.linspace(0, 1, len(labels)))
            cmap = clr.LinearSegmentedColormap.from_list('', color_list)
            #color_list = ['#0047f1', '#00a4c0', '#00e767', '#4cff01', '#a3ff05', '#e4ff05', '#ffe600', '#ff9800', '#ff3400', '#de1e47', '#8944a0', '#2e2ce8']
            #cmap = clr.LinearSegmentedColormap.from_list('', color_list)
            VariableLimits = np.arange(len(labels))
            norm = clr.BoundaryNorm(np.append(VariableLimits, 1000), ncolors=256)
            patches = [Patch(color=color, label=label) for label, color in zip(labels, color_list)] 
        else:
            #color_list = ['blue', 'green', 'yellow', 'red', 'blue']
            #cmap = clr.LinearSegmentedColormap.from_list('', color_list)
            min_val = 1
            max_val = 365
            label = 'Day of Year (DOY)'    

    elif fn == 'suitable_sowing_days':
        data[data == 0] = np.nan
        #color_list = ['palegoldenrod', 'forestgreen', 'darkgreen', 'teal']
        #cmap = clr.LinearSegmentedColormap.from_list('', color_list)
        min_val = 0
        max_val = 365
        label = 'Number of Days'

    elif fn == 'crop_limiting_factor':
        labels = ['Temperature', 'Precipitation', 'Climate Variability', 'Photoperiod', 'Base Saturation', 'Coarse Fragments', 'Gypsum Content', 'Soil pH', 'Salinity (EC)',\
                'Texture Class', 'Soil Organic Carbon Content', 'Sodicity (ESP)', 'Soil Depth', 'Slope', 'No Data']
        color_list = ['#FF7F7F', '#0060cc', '#ffaa00', '#ccca16', '#00e6a9', '#cd8966','#e1e1e1', '#0084a8', '#ffbee8', '#d7c29e', '#267300', '#f57ab6', '#a87000', '#686868', 'white']
        cmap = clr.LinearSegmentedColormap.from_list('', color_list)
        color_list = cmap(np.linspace(0, 1, len(labels)))
        cmap = clr.LinearSegmentedColormap.from_list('', color_list)
        VariableLimits = np.arange(len(labels))
        norm = clr.BoundaryNorm(np.append(VariableLimits, 1000), ncolors=256)
        if '_novar' in os.path.dirname(filename):
            patch_id_to_remove = 2
            patches = [Patch(color=color, label=label) for i, (label, color) in enumerate(zip(labels, color_list)) if i != patch_id_to_remove]
        else:
            patches = [Patch(color=color, label=label) for label, color in zip(labels, color_list)] 
    
    elif fn == 'multiple_cropping':
        labels = ['Unsuitable', 'One Harvest', 'Two Harvests', 'Three Harvests']
        color_list = cmap(np.linspace(0, 1, len(labels)))
        cmap = clr.LinearSegmentedColormap.from_list('', color_list)
        #color_list = ['white', '#d0d392', '#a7c55c', '#287200']
        #cmap = clr.LinearSegmentedColormap.from_list('', color_list)
        VariableLimits = np.arange(len(labels))
        norm = clr.BoundaryNorm(np.append(VariableLimits, 1000), ncolors=256)
        patches = [Patch(color=color, label=label) for label, color in zip(labels, color_list)] 

    else:
        data[np.isnan(data)] = np.nan   
        #cmap = 'viridis'
        min_val = np.nanpercentile(data, 5)
        max_val = np.nanpercentile(data, 95)
        label = fn

    plt.ioff()
    fig = Figure(figsize=(5,7), facecolor='#f0f0f0', dpi=600, frameon=False)
    fig_legend = plt.figure(figsize=(3, 1), frameon=False)
    ax_legend = fig_legend.add_subplot(111)
    fig.set_facecolor('#f0f0f0')
    ax = fig.add_subplot(111, projection=projection)

    unknown = fn not in ['downscaled_precipitation', 'downscaled_temperature', 'climate_suitability', 'crop_suitability', 'multiple_cropping_sum', 'climate_suitability_mc',
                'optimal_sowing_date', 'optimal_sowing_date_mc_first', 'optimal_sowing_date_mc_second', 'optimal_sowing_date_mc_third', 'optimal_sowing_date_vernalization',
                'suitable_sowing_days', 'soil_suitability', 'optimal_sowing_date_vernalization', 'base_saturation_combined', 'coarse_fragments_combined',
                'ph_combined', 'salinity_combined', 'slope_combined', 'sodicity_combined', 'soil_organic_carbon_combined', 'soildepth_combined', 'multiple_cropping',
                'crop_limiting_factor', 'texture_combined', 'optimal_sowing_date_with_vernalization', 'start_growing_cycle_after_vernalization']
    
    if (unknown) or (fn in ['downscaled_precipitation', 'downscaled_temperature', 'climate_suitability', 'crop_suitability', 'multiple_cropping_sum', 'climate_suitability_mc',
                'optimal_sowing_date', 'optimal_sowing_date_mc_first', 'optimal_sowing_date_mc_second', 'optimal_sowing_date_mc_third', 'optimal_sowing_date_vernalization',
                'suitable_sowing_days', 'soil_suitability', 'optimal_sowing_date_vernalization', 'base_saturation_combined', 'coarse_fragments_combined',
                'ph_combined', 'salinity_combined', 'slope_combined', 'sodicity_combined', 'soil_organic_carbon_combined', 'soildepth_combined',
                'optimal_sowing_date_with_vernalization', 'start_growing_cycle_after_vernalization'] and not classified):
        im = ax.imshow(data, extent=(x_min, x_max, y_min, y_max), origin='upper', cmap=cmap, transform=ccrs.PlateCarree(), vmin=min_val, vmax=max_val, interpolation='nearest') #type:ignore
        if (abs(x_min) + abs(x_max) >= 300) or (abs(y_min) + abs(y_max) >= 120):
            ax.set_extent((-180, 180, -90, 90), crs=ccrs.PlateCarree())
        else:
            ax.set_extent((x_min, x_max, y_min, y_max), crs=ccrs.PlateCarree()) #type:ignore
        try:
            ax.add_feature(cfeature.COASTLINE, linewidth=.6) #type:ignore
            ax.add_feature(cfeature.BORDERS, linewidth=.3, linestyle='-') #type:ignore
            ax.add_feature(cfeature.OCEAN,facecolor='lightsteelblue') #type:ignore
            ax.add_feature(cfeature.LAKES, linewidth=.6) #type:ignore
        except:
            pass
        gl = ax.gridlines(draw_labels=True, linewidth=.5, color='gray', alpha=.4, linestyle='--', x_inline=False, y_inline=False) #type:ignore
        gl.xlocator = plt.FixedLocator(range(-180, 181, 10)) #type:ignore
        gl.ylocator = plt.FixedLocator(range(-90, 91, 10)) #type:ignore
        gl.xlabel_style = {'size': 7}
        gl.ylabel_style = {'size': 7}
        
        ax_legend.axis('off')
        dummy_im = ax_legend.imshow([[min_val, max_val]], cmap=cmap, visible=False)
        cbar = fig_legend.colorbar(dummy_im, ax=ax_legend, orientation='horizontal', fraction=0.15, pad=0.2)
        cbar.set_label(label, fontsize=7)
        cbar.ax.tick_params(labelsize=7)

    else:
        mesh = ax.pcolormesh(np.linspace(x_min, x_max, data.shape[1]), np.linspace(y_min, y_max, data.shape[0]),
                                np.rot90(np.transpose(data)), cmap=cmap, transform=ccrs.PlateCarree(), norm=norm)
        if (abs(x_min) + abs(x_max) >= 300) or (abs(y_min) + abs(y_max) >= 120):
            ax.set_extent((-180, 180, -90, 90), crs=ccrs.PlateCarree())
        else:
            ax.set_extent((x_min, x_max, y_min, y_max), crs=ccrs.PlateCarree()) #type:ignore
        try:
            ax.add_feature(cfeature.COASTLINE, linewidth=.6) #type:ignore
            ax.add_feature(cfeature.BORDERS, linewidth=.3, linestyle='-') #type:ignore
            ax.add_feature(cfeature.OCEAN, facecolor='lightsteelblue') #type:ignore
            ax.add_feature(cfeature.LAKES, linewidth=.6) #type:ignore
        except:
            pass
        gl = ax.gridlines(draw_labels=True, linewidth=.5, color='gray', alpha=.4, linestyle='--', x_inline=False, y_inline=False) #type:ignore
        gl.xlocator = plt.FixedLocator(range(-180, 181, 10)) #type:ignore
        gl.ylocator = plt.FixedLocator(range(-90, 91, 10)) #type:ignore
        gl.xlabel_style = {'size': 7}
        gl.ylabel_style = {'size': 7}
        if fn == 'crop_limiting_factor':
            for i, patch in enumerate(patches):
                if i == len(patches) - 1:  # Check if it's the last patch
                    patch.set_edgecolor('black')
                else:
                    patch.set_edgecolor('none')
        ax_legend.axis('off')
        leg = ax_legend.legend(handles=patches, loc='lower center', bbox_to_anchor=(0.5, -.74), ncol=3, edgecolor='none', facecolor='none', fontsize=6, fancybox=True)

    title_text = f'{datetime.now().year}-{datetime.now().month}-{datetime.now().day}'+'\nFile: '+str(filename)+'\nCreated by CropSuite v1.0'+'\n© Florian Zabel, Matthias Knüttel 2025'
    ax.set_title(title_text, loc='left', fontsize=4, color='black')
    save_file = os.path.join(os.path.dirname(filename), f'{fn}_class.png' if classified else f'{fn}.png')
    legend_file = os.path.join(os.path.dirname(filename), f'{fn}_legend.png')
    fig.savefig(save_file, bbox_inches='tight', pad_inches=.4, dpi=600)
    fig_legend.savefig(legend_file, bbox_inches='tight', dpi=600, transparent=True)
    fig_legend.clear()
    ax_legend.clear()
    del fig_legend, ax_legend
    return save_file, legend_file

def plot_diff_data(data, filename, x_min, x_max, y_min, y_max, compare_file, projection=ccrs.PlateCarree, classified=False, colormap='spectral'):
    fn, _ = os.path.splitext(os.path.basename(filename))

    if isinstance(colormap, list):
        cmap = clr.LinearSegmentedColormap.from_list('', colormap)
        color_list = colormap
    else:
        cmap = cm.get_cmap(colormap)
        color_list = cmap(np.linspace(0, 1, 254))

    if fn == 'downscaled_temperature':
        if classified:
            data = classify_2d_array(data, 5, min_val=-2, max_val=2)
            labels = ['Strong Decrease', 'Slight Decrease', 'No Changes', 'Slight Increase', 'Strong Increase']
            #color_list = ['darkblue', 'cornflowerblue', 'white', 'coral', 'darkred']
            #cmap = clr.LinearSegmentedColormap.from_list('', color_list)
            color_list = cmap(np.linspace(0, 1, len(labels)))
            cmap = clr.LinearSegmentedColormap.from_list('', color_list)
            VariableLimits = np.arange(len(labels))
            norm = clr.BoundaryNorm(np.append(VariableLimits, 1000), ncolors=256)
            patches = [Patch(color=color, label=label) for label, color in zip(labels, color_list)] 
        else:
            #color_list = ['darkblue', 'lightskyblue', 'white', 'coral', 'darkred']
            #cmap = clr.LinearSegmentedColormap.from_list('', color_list)
            min_val = -2
            max_val = 2
            label = 'Change in Temperature [K]'

    if fn == 'downscaled_precipitation':
        if classified:
            data = classify_2d_array(data, 5, min_val=-100, max_val=100)
            labels = ['Strong Decrease', 'Slight Decrease', 'No Changes', 'Slight Increase', 'Strong Increase']
            #color_list = ['darkred', 'coral', 'white', 'cornflowerblue', 'darkblue']
            #cmap = clr.LinearSegmentedColormap.from_list('', color_list)
            color_list = cmap(np.linspace(0, 1, len(labels)))
            cmap = clr.LinearSegmentedColormap.from_list('', color_list)
            VariableLimits = np.arange(len(labels))
            norm = clr.BoundaryNorm(np.append(VariableLimits, 1000), ncolors=256)
            patches = [Patch(color=color, label=label) for label, color in zip(labels, color_list)] 
        else:
            #color_list = ['darkred', 'coral', 'white', 'cornflowerblue', 'darkblue']
            #cmap = clr.LinearSegmentedColormap.from_list('', color_list)
            min_val = -100
            max_val = 100
            label = 'Change in Precipitation [mm]'

    if fn in ['climate_suitability', 'crop_suitability', 'multiple_cropping_sum', 'climate_suitability_mc', 'soil_suitability']:
        if classified:
            data = classify_2d_array(data, 5, min_val=-50, max_val=50)
            labels = ['Strong Decrease', 'Slight Decrease', 'No Changes', 'Slight Increase', 'Strong Increase']
            #color_list = ['darkred', 'coral', 'white', 'cornflowerblue', 'darkblue']
            #cmap = clr.LinearSegmentedColormap.from_list('', color_list)
            color_list = cmap(np.linspace(0, 1, len(labels)))
            cmap = clr.LinearSegmentedColormap.from_list('', color_list)
            VariableLimits = np.arange(len(labels))
            norm = clr.BoundaryNorm(np.append(VariableLimits, 1000), ncolors=256)
            patches = [Patch(color=color, label=label) for label, color in zip(labels, color_list)] 
        else:
            #color_list = ['darkred', 'coral', 'white', 'lightskyblue', 'darkblue']
            #cmap = clr.LinearSegmentedColormap.from_list('', color_list)
            min_val = -100
            max_val = 100
            label = 'Change in Suitability'

    elif fn == 'suitable_sowing_days':
        if classified:
            data = classify_2d_array(data, 5, min_val=-100, max_val=100)
            labels = ['Strong Decrease', 'Slight Decrease', 'No Changes', 'Slight Increase', 'Strong Increase']
            #color_list = ['darkred', 'coral', 'white', 'cornflowerblue', 'darkblue']
            #cmap = clr.LinearSegmentedColormap.from_list('', color_list)
            color_list = cmap(np.linspace(0, 1, len(labels)))
            cmap = clr.LinearSegmentedColormap.from_list('', color_list)
            VariableLimits = np.arange(len(labels))
            norm = clr.BoundaryNorm(np.append(VariableLimits, 1000), ncolors=256)
            patches = [Patch(color=color, label=label) for label, color in zip(labels, color_list)] 
        else:
            #color_list = ['darkred', 'coral', 'white', 'lightskyblue', 'darkblue']
            #cmap = clr.LinearSegmentedColormap.from_list('', color_list)
            min_val = -365
            max_val = 365
            label = 'Change in Suitable Sowing Days'

    elif fn == 'multiple_cropping':
        #color_list = ['darkred', 'red', 'coral', 'white', 'lightskyblue', 'blue', 'darkblue']
        #cmap = clr.LinearSegmentedColormap.from_list('', color_list)
        min_val = -3
        max_val = 3
        label = 'Change in Harvests'
    
    elif fn in ['optimal_sowing_date', 'optimal_sowing_date_mc_first', 'optimal_sowing_date_mc_second', 'optimal_sowing_date_mc_third', 'optimal_sowing_date_vernalization', 'optimal_sowing_date_with_vernalization', 'start_growing_cycle_after_vernalization']:
        if classified:
            data = classify_2d_array(data, 7, min_val=-105, max_val=105)
            labels = ['> 3 Months earlier', '2 Months earlier', '1 Month earlier', 'No significant change', '1 Month later', '2 Months later', '> 3 Months later']
            #color_list = ['darkred', 'coral', 'palegoldenrod', 'white', 'paleturquoise', 'cornflowerblue', 'darkblue']
            #cmap = clr.LinearSegmentedColormap.from_list('', color_list)
            color_list = cmap(np.linspace(0, 1, len(labels)))
            cmap = clr.LinearSegmentedColormap.from_list('', color_list)
            VariableLimits = np.arange(len(labels))
            norm = clr.BoundaryNorm(np.append(VariableLimits, 1000), ncolors=256)
            patches = [Patch(color=color, label=label) for label, color in zip(labels, color_list)] 
        else:
            #color_list = ['black', 'blue', 'white', 'red', 'black']
            #cmap = clr.LinearSegmentedColormap.from_list('', color_list)
            min_val = -180
            max_val = 180
            label = 'Shift in Day of Year (DOY)'  
            data[(data > 183) | (data < -183)] = np.nan

    
    plt.ioff()
    fig = Figure(figsize=(5,7), facecolor='#f0f0f0', dpi=600)
    fig_legend = plt.figure(figsize=(3, 1))
    ax_legend = fig_legend.add_subplot(111)
    fig.set_facecolor('#f0f0f0')
    ax = fig.add_subplot(111, projection=projection)

    if fn in ['climate_suitability', 'crop_suitability', 'multiple_cropping_sum', 'climate_suitability_mc',\
                                    'optimal_sowing_date', 'optimal_sowing_date_mc_first', 'optimal_sowing_date_mc_second',\
                                        'optimal_sowing_date_mc_third', 'optimal_sowing_date_vernalization', 'suitable_sowing_days', 'multiple_cropping', 'soil_suitability', 
                                        'downscaled_temperature', 'downscaled_precipitation', 'optimal_sowing_date_with_vernalization',
                                        'start_growing_cycle_after_vernalization'] and not classified:
        im = ax.imshow(data, extent=(x_min, x_max, y_min, y_max), origin='upper', cmap=cmap, transform=ccrs.PlateCarree(), vmin=min_val, vmax=max_val, interpolation='nearest')
        if (abs(x_min) + abs(x_max) >= 300) or (abs(y_min) + abs(y_max) >= 120):
            ax.set_extent((-180, 180, -90, 90), crs=ccrs.PlateCarree())
        else:
            ax.set_extent((x_min, x_max, y_min, y_max), crs=ccrs.PlateCarree()) #type:ignore
        try:
            ax.add_feature(cfeature.COASTLINE, linewidth=.6) #type:ignore
            ax.add_feature(cfeature.BORDERS, linewidth=.3, linestyle='-') #type:ignore
            ax.add_feature(cfeature.OCEAN, facecolor='lightsteelblue') #type:ignore
            ax.add_feature(cfeature.LAKES, linewidth=.6) #type:ignore
        except:
            pass
        gl = ax.gridlines(draw_labels=True, linewidth=.5, color='gray', alpha=.4, linestyle='--', x_inline=False, y_inline=False) #type:ignore
        gl.xlocator = plt.FixedLocator(range(-180, 181, 10)) #type:ignore
        gl.ylocator = plt.FixedLocator(range(-90, 91, 10)) #type:ignore
        gl.xlabel_style = {'size': 7}
        gl.ylabel_style = {'size': 7}

        ax_legend.axis('off')
        dummy_im = ax_legend.imshow([[min_val, max_val]], cmap=cmap, visible=False)
        cbar = fig_legend.colorbar(dummy_im, ax=ax_legend, orientation='horizontal', fraction=0.15, pad=0.2)
        cbar.set_label(label, fontsize=7)
        cbar.ax.tick_params(labelsize=7)

    else:
        mesh = ax.pcolormesh(np.linspace(x_min, x_max, data.shape[1]), np.linspace(y_min, y_max, data.shape[0]),
                                np.rot90(np.transpose(data)), cmap=cmap, transform=ccrs.PlateCarree(), norm=norm)
        if (abs(x_min) + abs(x_max) >= 300) or (abs(y_min) + abs(y_max) >= 120):
            ax.set_extent((-180, 180, -90, 90), crs=ccrs.PlateCarree())
        else:
            ax.set_extent((x_min, x_max, y_min, y_max), crs=ccrs.PlateCarree()) #type:ignore
        try:
            ax.add_feature(cfeature.COASTLINE, linewidth=.6) #type:ignore
            ax.add_feature(cfeature.BORDERS, linewidth=.3, linestyle='-') #type:ignore
            ax.add_feature(cfeature.OCEAN, facecolor='lightsteelblue') #type:ignore
            ax.add_feature(cfeature.LAKES, linewidth=.6) #type:ignore
        except:
            pass
        gl = ax.gridlines(draw_labels=True, linewidth=.5, color='gray', alpha=.4, linestyle='--', x_inline=False, y_inline=False) #type:ignore
        gl.xlocator = plt.FixedLocator(range(-180, 181, 10)) #type:ignore
        gl.ylocator = plt.FixedLocator(range(-90, 91, 10)) #type:ignore
        gl.xlabel_style = {'size': 7}
        gl.ylabel_style = {'size': 7}
        if fn == 'crop_limiting_factor':
            for i, patch in enumerate(patches):
                if i == len(patches) - 1:  # Check if it's the last patch
                    patch.set_edgecolor('black')
                else:
                    patch.set_edgecolor('none')
        ax_legend.axis('off')
        leg = ax_legend.legend(handles=patches, loc='lower center', bbox_to_anchor=(0.5, -.74), ncol=3, edgecolor='none', facecolor='none', fontsize=6, fancybox=True)

    title_text = f'{datetime.now().year}-{datetime.now().month}-{datetime.now().day}'+'\nFile: '+str(filename)+'\nCompared to: '+str(compare_file)+'\nCreated by CropSuite v1.0'+'\n© Florian Zabel, Matthias Knüttel 2025'
    ax.set_title(title_text, loc='left', fontsize=4, color='black')

    save_file = os.path.join(os.path.dirname(filename), f'{fn}_class.png' if classified else f'{fn}.png')
    legend_file = os.path.join(os.path.dirname(filename), f'{fn}_legend.png')
    fig.savefig(save_file, bbox_inches='tight', pad_inches=.4, dpi=600)
    fig_legend.savefig(legend_file, bbox_inches='tight', dpi=600, transparent=True)
    fig_legend.clear()
    ax_legend.clear()
    del fig_legend, ax_legend
    return save_file, legend_file

def merge_plot_legend(figure, legend):
    fig_img, leg_image = Image.open(figure), Image.open(legend)
    second_width, second_height = fig_img.size
    legend_width, legend_height = leg_image.size
    new_width, new_height = second_width, second_height + legend_height
    new_image = Image.new('RGBA', (new_width, new_height), (240, 240, 240, 255)) #type:ignore
    new_image.paste(fig_img, (0, 0))
    legend_x, legend_y = (new_width - legend_width) // 2, second_height
    new_image.paste(leg_image, (legend_x, legend_y), leg_image)
    new_image.save(figure, 'PNG')
    os.remove(legend)
    return figure

def resample_to_match(data, comp_data):
    if data.shape == comp_data.shape:
        return data, comp_data
    else:
        if np.prod(data.shape) > np.prod(comp_data.shape):
            larger, smaller = data, comp_data
        else:
            larger, smaller = comp_data, data
        slices = tuple(slice(0, min(larger_dim, smaller_dim)) for larger_dim, smaller_dim in zip(larger.shape, smaller.shape))
        resampled_larger = larger[slices]
        if np.prod(data.shape) > np.prod(comp_data.shape):
            return resampled_larger, comp_data
        else:
            return data, resampled_larger

def viewer_gui(config):
    config_path = config
    config = rci.read_ini_file(config)
    global results
    results = os.path.dirname(config['files']['output_dir'])

    viewer_gui = Tk()
    
    viewer_gui.title('CropSuite - Data Viewer')
    x, y = 1200, 800
    viewer_gui.geometry(f'{x}x{y}+{(viewer_gui.winfo_screenwidth() - x) // 2}+{(viewer_gui.winfo_screenheight() - y) // 2}')
    viewer_gui.resizable(1, 1) #type:ignore
    viewer_gui.focus_force()

    def get_item_path(item, treeview):
        try:
            path = []
            while item:
                path.insert(0, treeview.item(item, 'text'))
                item = treeview.parent(item)
            new = os.path.join(*path[1:])
        except:
            new = os.path.join(results)
        return new

    def on_treeview_select(event):
        selected_item = treeview_select.selection()[0]
        item_path = os.path.join(results, get_item_path(selected_item, treeview_select))
        files = get_tifs(item_path, compare_val.get())
        ds_ex = 'ds_temp_0.nc' in files
        files = [fn for fn in files if not str(fn).startswith('ds_prec_') and not str(fn).startswith('ds_temp_')] + (['downscaled_temperature', 'downscaled_precipitation'] if ds_ex else [])
        sel_crop['values'] = files
        sel_crop_val.set('crop_suitability.tif' if files else '')
        day_selector_var.set('Year')
        day_selector_box.config(state='disabled')
        if 'all_suitability_vals.tif' in os.listdir(item_path):
            limfabut.config(state='normal')
        else:
            limfabut.config(state='disabled')
        if files:
            try:
                ext = dt.get_geotiff_extent(os.path.join(item_path, 'crop_suitability.tif'))
                if ext != None:
                    cus_ymax.set(ext.top) #type:ignore
                    cus_ymin.set(ext.bottom) #type:ignore
                    cus_xmax.set(ext.right) #type:ignore
                    cus_xmin.set(ext.left) #type:ignore
            except:
                pass

    def plot_command():
        tif_path = os.path.join(results, get_item_path(treeview_select.selection()[0], treeview_select), sel_crop_val.get())
        start_day, end_day = months_days_of_year.get(day_selector_var.get()) #type:ignore
        start_day -= 1
        end_day -= 1
        # Startday und endday zum Slicing der Kliamdaten übergeben, dann Mittelwert

        colormap = colormap_list.get(cmap_sel_var.get(), 'viridis' if compare_val.get() == 0 else 'Spectral')

        if compare_val.get() == 1:
            compare_path = os.path.join(comp_dir, get_item_path(treeview_compare.selection()[0], treeview_compare), sel_crop_val.get())
            plot_canvas_compare(tif_path, compare_path, classified=classified_val.get() == 1, layer=[start_day, end_day], colormap=colormap)
        else:
            plot_canvas(tif_path, classified=classified_val.get() == 1, layer=[start_day, end_day], colormap=colormap)

    def get_directory(initdir):
        dir = filedialog.askdirectory(initialdir=initdir)
        if not dir == '':
            global results
            results = dir
            treeview_select.delete(*treeview_select.get_children())
            root_id = treeview_select.insert("", 'end', text=os.path.basename(dir), open=True)
            populate_treeview(treeview_select, root_id, dir)
    
    main_frame = tk.Frame(viewer_gui, borderwidth=1, relief='ridge')
    main_frame.pack(side='left', fill='y')

    sel_but = Button(main_frame, text=' Select Directory', compound='left', command=lambda: get_directory(results))
    sel_but.pack(side='top', fill='x', expand=True, padx=5, pady=5)

    frm_treeview = tk.Frame(main_frame)
    frm_treeview.pack(side='top', fill='both', expand=True, padx=5, pady=5)

    treeview_select = ttk.Treeview(frm_treeview)
    treeview_select.pack(side='left', fill='both', expand=True, padx=5, pady=5)

    scrollbar = ttk.Scrollbar(frm_treeview, orient="vertical", command=treeview_select.yview)
    scrollbar.pack(side='right', fill='y')
    treeview_select.configure(yscrollcommand=scrollbar.set)
    treeview_select.bind("<<TreeviewSelect>>", on_treeview_select) #type:ignore

    def sel_crop_changed(*args):
        fn = sel_crop_val.get()
        if fn in ['downscaled_temperature', 'downscaled_precipitation']:
            day_selector_box.config(state='normal')
        else:
            day_selector_var.set('Year')
            day_selector_box.config(state='disabled')
        
        cmap_sel_var.set(file_colors_dict.get(os.path.splitext(fn)[0].lower() + ('' if compare_val.get() == 0 else '_diff'), 'viridis'))
        if fn == 'crop_limiting_factor.tif':
            cmap_sel_box.config(state='disabled')
        else:
            cmap_sel_box.config(state='normal')

    sel_crop_val = tk.StringVar(viewer_gui, '')
    sel_crop = ttk.Combobox(main_frame, textvariable=sel_crop_val, width=20)
    sel_crop['values'] = ['']
    sel_crop.pack(side='top', padx=5, pady=5, fill='x')
    sel_crop.bind('<<ComboboxSelected>>', sel_crop_changed)

    months_days_of_year = {
        "Year": [1, 365],
        "January": [1, 31],
        "February": [32, 59],
        "March": [60, 90],
        "April": [91, 120],
        "May": [121, 151],
        "June": [152, 181],
        "July": [182, 212],
        "August": [213, 243],
        "September": [244, 273],
        "October": [274, 304],
        "November": [305, 334],
        "December": [335, 365]
    }

    day_selector_var = tk.StringVar(viewer_gui, 'Year')
    day_selector_box = ttk.Combobox(main_frame, textvariable=day_selector_var, width=20, state='disabled')
    day_selector_box['values'] = list(months_days_of_year.keys())
    day_selector_box.pack(side='top', padx=5, pady=5, fill='x')

    classified_val = IntVar(main_frame, 0)
    classified_cb = Checkbutton(main_frame, variable=classified_val, text='Classify')
    classified_cb.pack(side='top', padx=5, pady=5, fill='x')

    plot_but = Button(main_frame, text=' Plot', compound='left', command=plot_command)
    plot_but.pack(side='top', padx=5, pady=5, fill='x')

    file_colors_dict = {'base_saturation_combined': 'base_saturation', 'coarse_fragments_combined': 'coarse_fragments', 'gypsum_combined': 'gypsum',
                        'ph_combined': 'pH', 'salinity_combined': 'salinity', 'slope_combined': 'slope', 'sodicity_combined': 'sodicity',
                        'soil_organic_carbon_combined': 'organic_carbon', 'soildepth_combined': 'soildepth', 'texture_combined': 'texture',
                        'climate_suitability': 'suitability', 'climate_suitability_mc': 'suitability', 'crop_limiting_factor': 'crop_limiting',
                        'crop_suitability': 'suitability', 'multiple_cropping': 'multiple_cropping', 'optimal_sowing_date': 'sowing_date',
                        'optimal_sowing_date_mc_first': 'sowing_date', 'optimal_sowing_date_mc_second': 'sowing_date',
                        'optimal_sowing_date_mc_third': 'sowing_date', 'soil_suitability': 'suitability', 'suitable_sowing_days': 'suitable_sowing_days',
                        'downscaled_temperature': 'temperature', 'downscaled_precipitation': 'precipitation',
                        'climate_suitability_diff': 'diff', 'climate_suitability_mc_diff': 'diff',
                        'crop_suitability_diff': 'diff', 'multiple_cropping_diff': 'diff', 'optimal_sowing_date_diff': 'sowdate_diff',
                        'optimal_sowing_date_mc_first_diff': 'sowdate_diff', 'optimal_sowing_date_mc_second_diff': 'sowdate_diff',
                        'optimal_sowing_date_mc_third_diff': 'sowdate_diff', 'soil_suitability_diff': 'diff', 'suitable_sowing_days_diff': 'diff',
                        'optimal_sowing_date_with_vernalization_diff': 'diff', 'start_growing_cycle_after_vernalization_diff': 'diff',
                        'downscaled_temperature_diff': 'diff_r', 'downscaled_precipitation_diff': 'diff',
                        'optimal_sowing_date_with_vernalization': 'sowing_date', 'start_growing_cycle_after_vernalization': 'sowing_date'}

    colormap_list = {'suitability': ['white', 'darkgray', 'darkred', 'yellow', 'greenyellow', 'limegreen', 'darkgreen', 'darkslategray'],
                     'suitability_colorblind': ['#f5f5f5', '#a6611a', '#dfc27d', '#80cdc1', '#018571'],
                     'multiple_cropping': ['white', '#eceeba', '#a7c55c', '#0b3d00'],
                     'suitable_sowing_days': ['palegoldenrod', 'forestgreen', 'darkgreen', 'teal'],
                     'suitable_sowing_days_colorblind': ['#ffffcc', '#a1dab4', '#41b6c4', '#2c7fb8', '#253494'],
                     'sowing_date': ['blue', 'green', 'yellow', 'red', 'blue'],
                     'sowing_date_colorblind': ['gray', '#648fff', '#dc267f', '#ffb000', 'gray'],
                     'white-green-black': ['white', '#d0d392', '#a7c55c', '#287200', '#025012'],
                     'temperature': ['rebeccapurple', 'mediumblue', 'turquoise', 'lightgreen', 'orange', 'firebrick', 'saddlebrown'],
                     'precipitation': ['aquamarine', 'darkturquoise', 'cornflowerblue', 'mediumblue', 'darkviolet'],
                     'base_saturation': ['#563209', '#50380b', '#494b16', '#3c6025', '#266f30', '#0a713b', '#00684a', '#005856', '#00475e', '#043863'],
                     'coarse_fragments': ['darkgreen', 'limegreen', 'yellow', 'red', 'darkred'],
                     'gypsum': ['darkgreen', 'limegreen', 'yellow', 'red', 'darkred'],
                     'pH': ['darkcyan', 'aquamarine', 'darkgreen', 'limegreen', 'coral', 'darkred'],
                     'salinity': ['darkgreen', 'limegreen', 'yellow', 'orange', 'darkred'],
                     'slope': ['darkgreen', 'limegreen', 'orange', 'darkred'],
                     'sodicity': ['darkgreen', 'limegreen', 'darkred'],
                     'organic_carbon': ['darkred', 'yellow', 'limegreen', 'darkgreen', 'darkslategrey'],
                     'soildepth': ['darkred', 'yellow', 'limegreen', 'darkgreen', 'darkslategrey'],
                     'texture': ['#f3f3d9', '#d5bb76', '#a78834', '#a17a10', '#76531b', '#987a4d', '#736142', '#8a652b', '#8a4f2b', '#a66b4a', '#a6494a', '#8f4a80', '#64303b'],
                     'crop_limiting': ['#FF7F7F', '#0060cc', '#ffaa00', '#ccca16', '#00e6a9', '#cd8966','#e1e1e1', '#0084a8', '#ffbee8', '#d7c29e', '#267300', '#f57ab6', '#a87000', '#686868', 'white'],
                     'diff': ['darkred', 'coral', 'white', 'cornflowerblue', 'darkblue'], 'diff_r': ['darkblue', 'cornflowerblue', 'white', 'coral', 'darkred'],
                     'sowdate_diff': ['darkred', 'coral', 'palegoldenrod', 'white', 'paleturquoise', 'cornflowerblue', 'darkblue'],
                     'magma': 'magma', 'inferno': 'inferno', 'plasma': 'plasma', 'viridis': 'viridis', 'cividis': 'cividis', 'twilight': 'twilight', 'twilight_shifted': 'twilight_shifted', 'turbo': 'turbo',
                     'Blues': 'Blues', 'BrBG': 'BrBG', 'BuGn': 'BuGn', 'BuPu': 'BuPu', 'CMRmap': 'CMRmap', 'GnBu': 'GnBu', 'Greens': 'Greens', 'Greys': 'Greys', 'OrRd': 'OrRd',
                     'Oranges': 'Oranges', 'PRGn': 'PRGn', 'PiYG': 'PiYG', 'PuBu': 'PuBu', 'PuBuGn': 'PuBuGn', 'PuOr': 'PuOr', 'PuRd': 'PuRd', 'Purples': 'Purples', 'RdBu': 'RdBu',
                     'RdGy': 'RdGy', 'RdPu': 'RdPu', 'RdYlBu': 'RdYlBu', 'RdYlGn': 'RdYlGn', 'Reds': 'Reds', 'Spectral': 'Spectral', 'Wistia': 'Wistia', 'YlGn': 'YlGn', 'YlGnBu': 'YlGnBu',
                     'YlOrBr': 'YlOrBr', 'YlOrRd': 'YlOrRd', 'afmhot': 'afmhot', 'autumn': 'autumn', 'binary': 'binary', 'bone': 'bone', 'brg': 'brg', 'bwr': 'bwr', 'cool': 'cool',
                     'coolwarm': 'coolwarm', 'copper': 'copper', 'cubehelix': 'cubehelix', 'flag': 'flag', 'gist_earth': 'gist_earth', 'gist_gray': 'gist_gray', 'gist_heat': 'gist_heat',
                     'gist_ncar': 'gist_ncar', 'gist_rainbow': 'gist_rainbow', 'gist_stern': 'gist_stern', 'gist_yarg': 'gist_yarg', 'gnuplot': 'gnuplot', 'gnuplot2': 'gnuplot2',
                     'gray': 'gray', 'hot': 'hot', 'hsv': 'hsv', 'jet': 'jet', 'nipy_spectral': 'nipy_spectral', 'ocean': 'ocean', 'pink': 'pink', 'prism': 'prism', 'rainbow': 'rainbow',
                     'seismic': 'seismic', 'spring': 'spring', 'summer': 'summer', 'terrain': 'terrain', 'winter': 'winter', 'Accent': 'Accent', 'Dark2': 'Dark2', 'Paired': 'Paired',
                     'Pastel1': 'Pastel1', 'Pastel2': 'Pastel2', 'Set1': 'Set1', 'Set2': 'Set2', 'Set3': 'Set3', 'tab10': 'tab10', 'tab20': 'tab20', 'tab20b': 'tab20b', 'tab20c': 'tab20c',
                     'magma_r': 'magma_r', 'inferno_r': 'inferno_r', 'plasma_r': 'plasma_r', 'viridis_r': 'viridis_r', 'cividis_r': 'cividis_r', 'twilight_r': 'twilight_r',
                     'twilight_shifted_r': 'twilight_shifted_r', 'turbo_r': 'turbo_r', 'Blues_r': 'Blues_r', 'BrBG_r': 'BrBG_r', 'BuGn_r': 'BuGn_r', 'BuPu_r': 'BuPu_r',
                     'CMRmap_r': 'CMRmap_r', 'GnBu_r': 'GnBu_r', 'Greens_r': 'Greens_r', 'Greys_r': 'Greys_r', 'OrRd_r': 'OrRd_r', 'Oranges_r': 'Oranges_r', 'PRGn_r': 'PRGn_r',
                     'PiYG_r': 'PiYG_r', 'PuBu_r': 'PuBu_r', 'PuBuGn_r': 'PuBuGn_r', 'PuOr_r': 'PuOr_r', 'PuRd_r': 'PuRd_r', 'Purples_r': 'Purples_r', 'RdBu_r': 'RdBu_r',
                     'RdGy_r': 'RdGy_r', 'RdPu_r': 'RdPu_r', 'RdYlBu_r': 'RdYlBu_r', 'RdYlGn_r': 'RdYlGn_r', 'Reds_r': 'Reds_r', 'Spectral_r': 'Spectral_r', 'Wistia_r': 'Wistia_r',
                     'YlGn_r': 'YlGn_r', 'YlGnBu_r': 'YlGnBu_r', 'YlOrBr_r': 'YlOrBr_r', 'YlOrRd_r': 'YlOrRd_r', 'afmhot_r': 'afmhot_r', 'autumn_r': 'autumn_r', 'binary_r': 'binary_r',
                     'bone_r': 'bone_r', 'brg_r': 'brg_r', 'bwr_r': 'bwr_r', 'cool_r': 'cool_r', 'coolwarm_r': 'coolwarm_r', 'copper_r': 'copper_r', 'cubehelix_r': 'cubehelix_r',
                     'flag_r': 'flag_r', 'gist_earth_r': 'gist_earth_r', 'gist_gray_r': 'gist_gray_r', 'gist_heat_r': 'gist_heat_r', 'gist_ncar_r': 'gist_ncar_r',
                     'gist_rainbow_r': 'gist_rainbow_r', 'gist_stern_r': 'gist_stern_r', 'gist_yarg_r': 'gist_yarg_r', 'gnuplot_r': 'gnuplot_r', 'gnuplot2_r': 'gnuplot2_r',
                     'gray_r': 'gray_r', 'hot_r': 'hot_r', 'hsv_r': 'hsv_r', 'jet_r': 'jet_r', 'nipy_spectral_r': 'nipy_spectral_r', 'ocean_r': 'ocean_r', 'pink_r': 'pink_r',
                     'prism_r': 'prism_r', 'rainbow_r': 'rainbow_r', 'seismic_r': 'seismic_r', 'spring_r': 'spring_r', 'summer_r': 'summer_r', 'terrain_r': 'terrain_r',
                     'winter_r': 'winter_r', 'Accent_r': 'Accent_r', 'Dark2_r': 'Dark2_r', 'Paired_r': 'Paired_r', 'Pastel1_r': 'Pastel1_r', 'Pastel2_r': 'Pastel2_r', 'Set1_r': 'Set1_r',
                     'Set2_r': 'Set2_r', 'Set3_r': 'Set3_r', 'tab10_r': 'tab10_r', 'tab20_r': 'tab20_r', 'tab20b_r': 'tab20b_r', 'tab20c_r': 'tab20c_r'}
    
    cmap_frm = Frame(main_frame)
    cmap_frm.pack(side='top', pady=5, fill='x')
    Label(cmap_frm, text='Colormap:').pack(side='left', padx=5)
    cmap_sel_var = tk.StringVar(viewer_gui, list(colormap_list.keys())[0])
    cmap_sel_box = ttk.Combobox(cmap_frm, textvariable=cmap_sel_var)
    cmap_sel_box['values'] = list(colormap_list.keys())
    cmap_sel_box.pack(side='right', padx=5, fill='x', expand=True)    

    proj_list = {'Projection: Plate Carrée': ccrs.PlateCarree,
                 'Projection: Robinson': ccrs.Robinson,
                 'Projection: Eckert IV': ccrs.EckertIV,
                 'Projection: AlbersEqualArea': ccrs.AlbersEqualArea,
                 'Projection: Mercator': ccrs.Mercator,
                 'Projection: Mollweide': ccrs.Mollweide,
                 'Projection: InterruptedGoodeHomolosine': ccrs.InterruptedGoodeHomolosine,
                 'Projection: NearsidePerspective': ccrs.NearsidePerspective,
                 'Projection: Aitoff': ccrs.Aitoff,
                 'Projection: EqualEarth': ccrs.EqualEarth,
                 'Projection: Hammer': ccrs.Hammer}
    
    proj_sel_var = tk.StringVar(viewer_gui, 'Projection: Plate Carrée')
    proj_sel_box = ttk.Combobox(main_frame, textvariable=proj_sel_var, width=20)
    proj_sel_box['values'] = list(proj_list.keys())
    proj_sel_box.pack(side='top', padx=5, pady=5, fill='x')    
    
    def cusex():
        if cus_ex_val.get() == 1:
            cus_ymax_ent.config(state='normal')
            cus_xmin_ent.config(state='normal')
            cus_ymin_ent.config(state='normal')
            cus_xmax_ent.config(state='normal')
            proj_sel_var.set('Projection: Plate Carrée')
            proj_sel_box.config(state='disabled')
        else:
            cus_ymax_ent.config(state='disabled')
            cus_xmin_ent.config(state='disabled')
            cus_ymin_ent.config(state='disabled')
            cus_xmax_ent.config(state='disabled')  
            proj_sel_box.config(state='normal')          

    cus_ex_val = IntVar(main_frame, 0)
    cus_ex_cb = Checkbutton(main_frame, variable=cus_ex_val, text='Set Extent', command=cusex)
    cus_ex_cb.pack(side='top', padx=5, pady=5, fill='x')
    
    cus_lab_frm = Frame(main_frame)
    cus_lab_frm.pack(side='top', fill='both', expand=True)
    Label(cus_lab_frm, text='Top', width=6).pack(side='left', padx=5)
    Label(cus_lab_frm, text='Left', width=6).pack(side='left', padx=5)
    Label(cus_lab_frm, text='Bottom', width=6).pack(side='left', padx=5)
    Label(cus_lab_frm, text='Right', width=6).pack(side='left', padx=5)

    cus_frm = Frame(main_frame)
    cus_frm.pack(side='top', fill='both', expand=True)
    cus_ymax = DoubleVar(cus_frm, 90)
    cus_xmin = DoubleVar(cus_frm, -180)
    cus_ymin = DoubleVar(cus_frm, -90)
    cus_xmax = DoubleVar(cus_frm, 180)

    cus_ymax_ent = Entry(cus_frm, textvariable=cus_ymax, state='disabled', width=7)
    cus_xmin_ent = Entry(cus_frm, textvariable=cus_xmin, state='disabled', width=7)
    cus_ymin_ent = Entry(cus_frm, textvariable=cus_ymin, state='disabled', width=7)
    cus_xmax_ent = Entry(cus_frm, textvariable=cus_xmax, state='disabled', width=7)
    cus_ymax_ent.pack(side='left', pady=5, padx=5)
    cus_xmin_ent.pack(side='left', pady=5, padx=5)
    cus_ymin_ent.pack(side='left', pady=5, padx=5)
    cus_xmax_ent.pack(side='left', pady=5, padx=5)

    def on_checkbutton_checked():
        if compare_val.get() == 1:
            fill_tv()
        else:
            empty_cv()
        cmap_sel_var.set(file_colors_dict.get(os.path.splitext(sel_crop_val.get())[0].lower() + ('' if compare_val.get() == 0 else '_diff'), 'viridis'))

    limfabut = Button(main_frame, text='Limitation Analyzer', command=lambda: limfa.limfact_analyzer(config_ini=config_path, current_path=os.path.join(results, get_item_path(treeview_select.selection()[0], treeview_select))), state='disabled')
    limfabut.pack(side='top', fill='x', expand=True, padx=5, pady=5)


    compare_val = IntVar(main_frame, 0)
    compare_cb = Checkbutton(main_frame, variable=compare_val, text='Compare', command=on_checkbutton_checked)
    compare_cb.pack(side='top', padx=5, pady=5, fill='x')
    
    global comp_dir
    comp_dir = results

    def get_comp_directory(initdir):
        dir = filedialog.askdirectory(initialdir=initdir)
        if not dir == '':
            global comp_dir
            comp_dir = dir
            empty_cv()
            fill_tv()

    sel_but2 = Button(main_frame, text=' Select Directory', compound='left', command=lambda: get_comp_directory(results))
    sel_but2.pack(side='top', fill='x', expand=True, padx=5, pady=5)

    frm_treeview_comp = tk.Frame(main_frame)
    frm_treeview_comp.pack(side='top', fill='both', expand=True)

    treeview_compare = ttk.Treeview(frm_treeview_comp)
    treeview_compare.pack(side='left', fill='both', expand=True, padx=5, pady=5)

    scrollbar_compare = ttk.Scrollbar(frm_treeview_comp, orient="vertical", command=treeview_select.yview)
    scrollbar_compare.pack(side='right', fill='y')
    treeview_compare.configure(yscrollcommand=scrollbar_compare.set)

    frm_canvas = tk.Frame(viewer_gui)
    frm_canvas.pack(side='left', fill='both', expand=True)

    canvas = tk.Canvas(master=frm_canvas)
    canvas.pack(side='right', fill='both', expand=True)
    
    def plot_canvas(filename, classified, layer=[0, 0], colormap='viridis'):
        extent = [cus_xmin.get(), cus_xmax.get(), cus_ymin.get(), cus_ymax.get()] if cus_ex_val.get() == 1 else [0, 0, 0, 0]
        data, x_min, x_max, y_min, y_max, nodata = read_data_file(filename, layer, extent=extent) #type:ignore
        data = np.asarray(data, dtype=np.float32)
        data[data == nodata] = np.nan
        fact = int(np.max([(data.shape[0] / 2000), (data.shape[1] / 2000)]))
        if fact > 1:
            data = dt.resample_array_with_nans_custom(data, fact)
        projection = proj_list.get(proj_sel_var.get(), ccrs.PlateCarree)()
        savefig, legend = plot_data(data, filename, x_min, x_max, y_min, y_max, projection=projection, classified=classified, colormap=colormap)
        fig = merge_plot_legend(savefig, legend)
        show_image_centered(canvas, frm_canvas, fig)

    def plot_canvas_compare(filename, compare_file, classified=False, save=False, layer=[0, 0], colormap='viridis'):
        extent = [cus_xmin.get(), cus_xmax.get(), cus_ymin.get(), cus_ymax.get()] if cus_ex_val.get() == 1 else [0, 0, 0, 0]

        data, x_min, x_max, y_min, y_max, nodata = read_data_file(filename, layer, extent=extent) #type:ignore
        comp_data, x_min, x_max, y_min, y_max, nodata = read_data_file(compare_file, layer, extent=extent) #type:ignore
        data, comp_data = np.asarray(data, dtype=np.float32), np.asarray(comp_data, dtype=np.float32)
        data[data == nodata], comp_data[comp_data == nodata] = np.nan, np.nan

        fact = int(np.max([(data.shape[0] / 2000), (data.shape[1] / 2000)]))
        data, comp_data = resample_to_match(data, comp_data)

        if fact > 1:
            data = dt.resample_array_with_nans_custom(data, fact)
            comp_data = dt.resample_array_with_nans_custom(comp_data, fact)

        fn, _ = os.path.splitext(os.path.basename(filename))
        if fn in ['optimal_sowing_date', 'optimal_sowing_date_mc_first', 'optimal_sowing_date_mc_second', 'optimal_sowing_date_mc_third', 'optimal_sowing_date_vernalization']:
            data[np.isnan(comp_data) | (comp_data <= 0)] = np.nan
            comp_data[np.isnan(data) | (data <= 0)] = np.nan

        diff = data - comp_data

        projection = proj_list.get(proj_sel_var.get(), ccrs.PlateCarree)()
        savefig, legend = plot_diff_data(diff, filename, x_min, x_max, y_min, y_max, compare_file, projection=projection, classified=classified, colormap=colormap)
        fig = merge_plot_legend(savefig, legend)
        show_image_centered(canvas, frm_canvas, fig)

    root_id = treeview_select.insert("", 'end', text=os.path.basename(comp_dir), open=True)
    populate_treeview(treeview_select, root_id, comp_dir)

    def fill_tv():
        root_comp_id = treeview_compare.insert("", 'end', text=os.path.basename(comp_dir), open=True)
        populate_treeview(treeview_compare, root_comp_id, comp_dir)

    def empty_cv():
        treeview_compare.delete(*treeview_compare.get_children())

    viewer_gui.mainloop()
    viewer_gui.wait_window(viewer_gui)

def main_gui():
    global plant_param_dir
    plant_param_dir = ''
    main_window = Tk()
    main_window.after(201, lambda :main_window.iconbitmap(os.path.join(os.path.dirname(__file__), 'src', 'cropsuiteicon.ico')))
    x, y = 600, 720
    main_window.geometry(f'{x}x{y}+{(main_window.winfo_screenwidth() - x) // 2}+{(main_window.winfo_screenheight() - y) // 2}')
    main_window.title('CropSuite')
    main_window.resizable(0, 0) #type:ignore
    main_window.focus_force()

    red = '#E40808'
    yellow = '#E4C308'
    green = '#08E426'

    font10 = f'Helvetica 10'
    font12 = f'Helvetica 12'
    font16 = f'Helvetica 16'

    # Head
    header_image = Image.open(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src', 'header.png')))
    header_image = header_image.resize((600, 218))
    header_image = ImageTk.PhotoImage(header_image)
    head_lab = Label(main_window, image=header_image, relief='flat') #type:ignore
    head_lab.pack()
    
    def open_viewer(config):
        viewer_gui(config)

    def open_manual():
        file_path = r'CropSuite_Manual.pdf'
        if platform.system() == "Windows": os.startfile(file_path) #type:ignore
        elif platform.system() == "Darwin": os.system("open " + file_path)
        else: os.system("xdg-open " + file_path)

    def create_cfg():
        global config_dict
        config_dict = {
            "files": {
                "output_dir": ".",
                "orig_temp_min_data": ".",
                "orig_temp_max_data": ".",
                "orig_prec_data": ".",
                "climate_data_dir": ".",
                "plant_param_dir": ".",
                "fine_dem": ".",
                "land_sea_mask": ".",
                "texture_classes": ".",
                "worldclim_precipitation_data_dir": ".",
                "worldclim_temperature_data_dir": "."
            },
            "options": {
                "use_scheduler": "y",
                "irrigation": 0,
                "precipitation_downscaling_method": 2,
                "temperature_downscaling_method": 2,
                "output_format": "geotiff",
                "output_all_startdates": "y",
                "output_grow_cycle_as_doy": "y",
                "downscaling_window_size": 8,
                "downscaling_use_temperature_gradient": "y",
                "downscaling_dryadiabatic_gradient": 0.00976,
                "downscaling_saturation_adiabatic_gradient": 0.007,
                "downscaling_temperature_bias_threshold": 0.0005,
                "downscaling_precipitation_bias_threshold": 0.0001,
                "downscaling_precipitation_per_day_threshold": 0.5,
                "output_all_limiting_factors": "y",
                "remove_interim_results": "y",
                "multiple_cropping_turnaround_time": 21,
                "output_soil_data": "y"
            },
            "extent": {
                "upper_left_x": -5,
                "upper_left_y": 5,
                "lower_right_x": 5,
                "lower_right_y": -5
            },
            "climatevariability": {
                "consider_variability": "y"
            },
            "membershipfunctions": {
                "plot_for_each_crop": "y"
            },
            "parameters.base_saturation": {
                "data_directory": ".",
                "weighting_method": 0,
                "weighting_factors": "1.0,0.0,0.0,0.0,0.0,0.0",
                "conversion_factor": 1.0,
                "no_data": -128.0,
                "interpolation_method": 0,
                "rel_member_func": "base_sat"
            },
            "parameters.coarse_fragments": {
                "data_directory": ".",
                "weighting_method": 2,
                "weighting_factors": "2,1.5,1,0.75,0.5,0.25",
                "conversion_factor": 10,
                "interpolation_method": 0,
                "rel_member_func": "coarsefragments"
            },
            "parameters.clay_content": {
                "data_directory": ".",
                "weighting_method": 2,
                "weighting_factors": "2,1.5,1,0.75,0.5,0.25",
                "conversion_factor": 10,
                "interpolation_method": 0,
                "rel_member_func": "texture"
            },
            "parameters.gypsum": {
                "data_directory": ".",
                "weighting_method": 0,
                "weighting_factors": "2,1.5,1,0.75,0.5,0.25",
                "conversion_factor": 10,
                "interpolation_method": 0,
                "rel_member_func": "gypsum"
            },
            "parameters.pH": {
                "data_directory": ".",
                "weighting_method": 2,
                "weighting_factors": "2.0,1.5,1.0,0.75,0.5,0.25",
                "conversion_factor": 10.0,
                "interpolation_method": 0,
                "rel_member_func": "ph"
            },
            "parameters.salinity": {
                "data_directory": ".",
                "weighting_method": 0,
                "weighting_factors": "2,1.5,1,0.75,0.5,0.25",
                "conversion_factor": 1,
                "interpolation_method": 0,
                "rel_member_func": "elco"
            },
            "parameters.sand_content": {
                "data_directory": ".",
                "weighting_method": 2,
                "weighting_factors": "2,1.5,1,0.75,0.5,0.25",
                "conversion_factor": 10,
                "interpolation_method": 0,
                "rel_member_func": "texture"
            },
            "parameters.soil_organic_carbon": {
                "data_directory": ".",
                "weighting_method": 1,
                "weighting_factors": "2,1.5,1,0.75,0.5,0.25",
                "conversion_factor": 100,
                "interpolation_method": 0,
                "rel_member_func": "organic_carbon"
            },
            "parameters.sodicity": {
                "data_directory": ".",
                "weighting_method": 0,
                "weighting_factors": "2,1.5,1,0.75,0.5,0.25",
                "conversion_factor": 1,
                "interpolation_method": 0,
                "rel_member_func": "esp"
            },
            "parameters.soildepth": {
                "data_directory": ".",
                "weighting_method": 0,
                "weighting_factors": "2,1.5,1,0.75,0.5,0.25",
                "conversion_factor": 100,
                "interpolation_method": 0,
                "rel_member_func": "soildepth"
            },
            "parameters.slope": {
                "data_directory": ".",
                "weighting_method": 0,
                "weighting_factors": "2,1.5,1,0.75,0.5,0.25",
                "conversion_factor": 1,
                "interpolation_method": 0,
                "rel_member_func": "slope"
            }


        }
        fn = filedialog.asksaveasfilename(defaultextension=".ini", filetypes=[("Config.ini", "*.ini"), ("All files", "*.*")], title="New Config.ini")
        if not fn == None:
            rci.write_config(config_dict, fn)
            global current_cfg
            current_cfg = fn
            config_ini_var.set(current_cfg)
            checked = cp.check_config_file(config_dict, gui=True)
            if checked:
                check_cfg.config(text='  ☑  Config File is valid', fg=green)
                viewer_button.config(state='normal')
                preproc_button.config(state='normal')
                edit_cfg.config(state='normal')
                main_window.update()
            else:
                check_cfg.config(text='  ✗  Config file is faulty', fg=red)
                viewer_button.config(state='disabled')
                preproc_button.config(state='disabled')
                edit_cfg.config(state='disabled')
                main_window.update()

    frm = Frame(main_window)
    frm.pack(fill='x')

    rt = Tk()
    rt.withdraw()
    preproc_button = Button(frm, text='Preprocessing', compound='left', command=lambda: ppgui.preproc_gui(tk.Toplevel(rt), config_ini_var.get()))
    preproc_button.pack(side='left', padx=5, pady=5)

    manual_button = Button(frm, text=' Open Manual', command=open_manual, compound='left')
    manual_button.pack(side='right', padx=5, pady=5)

    viewer_button = Button(frm, text=' Open Data Viewer', compound='left', command=lambda: open_viewer(config_ini_var.get()))
    viewer_button.pack(side='right', padx=5, pady=5)

    Label(main_window, text='', font=font10 + ' bold').pack()
    
    global current_cfg
    current_cfg = r'.\config.ini'
    config_ini_var = StringVar(main_window, current_cfg)

    Label(main_window, text='1 - Config File').pack(pady=5, padx=5, anchor='w')

    frame = Frame(main_window, border=4)
    frame.pack()
    
    cfg = Label(frame, text='Path to config.ini:')
    config_ini_entry = Entry(frame, width=30, textvariable=config_ini_var, borderwidth=2, relief='groove')

    #config_ini = Label(frame, width=30, textvariable=config_ini_var, anchor='w', borderwidth=2, relief='groove')
    sel_cfg = Button(frame, text=' Select', compound='left', command=lambda: select_cfg_file())
    new_cfg = Button(frame, text=' Add New', compound='left', command=create_cfg)
    cfg.pack(side='left', padx=5, pady=5)
    config_ini_entry.pack(side='left', padx=5, pady=5)
    new_cfg.pack(side='right', padx=5, pady=5)
    sel_cfg.pack(side='right', padx=5, pady=5)

    global config_dict
    if os.path.exists(config_ini_var.get()):
        config_dict = rci.read_ini_file(config_ini_var.get())
    else:
        config_dict = {}
    try:
        if 'plant_param_dir' in config_dict['files']:
            plant_param_dir = config_dict['files'].get('plant_param_dir') #type:ignore
        else:
            plant_param_dir = 'plant_params'
    except:
        plant_param_dir = ''
    check_cfg = Label(main_window, text='  ☐  Checking Config File', fg=yellow, font=font12 + ' bold', anchor='w')
    check_cfg.pack()
    if os.path.exists(config_ini_var.get()):
        checked = cp.check_config_file(config_dict, gui=True)
    else:
        checked = False
    if checked:
        check_cfg.config(text='  ☑  Config File is valid', fg=green)
        viewer_button.config(state='normal')
        preproc_button.config(state='normal')
        main_window.update()
    else:
        check_cfg.config(text='  ✗  Config file is faulty', fg=red)
        viewer_button.config(state='disabled')
        preproc_button.config(state='disabled')
        main_window.update()
    main_window.update()

    def config_gui_button():
        config_gui(config_ini_var.get())
        config_dict = rci.read_ini_file(config_ini_var.get())
        if 'plant_param_dir' in config_dict['files']:
            plant_param_dir = config_dict['files'].get('plant_param_dir') #type:ignore
        else:
            plant_param_dir = 'plant_params'
        main_window.update()

    if checked:
        edit_cfg = Button(main_window, text=' Config.ini Options', compound='left', command=config_gui_button)
        viewer_button.config(state='normal')
        preproc_button.config(state='normal')
        main_window.update()
    else:
        edit_cfg = Button(main_window, text=' Config.ini Options', compound='left', command=config_gui_button, state='disabled')
        viewer_button.config(state='disabled')
        preproc_button.config(state='disabled')
        main_window.update()
    edit_cfg.pack(pady=5)

    Label(main_window, text='', font=font10 + ' bold').pack()

    def set_config_ini(ini_path):
        config_ini_var.set(ini_path)
        main_window.update()

    def select_cfg_file():
        config_file = ''
        while not str(config_file).endswith('.ini'):
            print('Select config.ini')
            config_file = filedialog.askopenfilename(initialdir=os.getcwd(), title='Select config.ini', filetypes=[("INI files", "*.ini")])
            if config_file:
                set_config_ini(config_file)
            config_dict = rci.read_ini_file(config_ini_var.get())
            checked = cp.check_config_file(config_dict, gui=True)
            if 'plant_param_dir' in config_dict['files']:
                plant_param_dir = config_dict['files'].get('plant_param_dir') #type:ignore
            else:
                plant_param_dir = 'plant_params'
            if checked:
                check_cfg.config(text='  ☑  Config File is valid', fg=green)
                edit_cfg.config(state='normal')
                viewer_button.config(state='normal')
                preproc_button.config(state='normal')
            else:
                check_cfg.config(text='  ✗  Config file is faulty', fg=red)
                edit_cfg.config(state='disabled')
                viewer_button.config(state='disabled')
                preproc_button.config(state='disabled')
            main_window.update()

    def open_plant_gui():
        plant_gui(config_ini_var.get())
        main_window.update()

    def get_config():
        return config_ini_var.get()

    Frame(main_window, bd=5, relief='sunken', height=2).pack(side='top', fill='x')
    Label(main_window, text='', font=font10 + ' bold').pack()
    Label(main_window, text='2 - Crop Selection').pack(pady=5, padx=5, anchor='w')
    frame = Frame(main_window, border=4)
    frame.pack()
    try:
        os.makedirs(plant_param_dir, exist_ok=True)
        no_crops = sum(1 for file in os.listdir(plant_param_dir) if file.endswith('.inf'))
    except:
        no_crops = 0 

    sel_but = Button(frame, text=' Select Crops', compound='left', command=lambda: open_plant_gui())
    sel_but.pack(side='left', padx=10)
    Label(main_window, text='').pack()

    Frame(main_window, bd=5, relief='sunken', height=2).pack(side='top', fill='x')
    Label(main_window, text='', font=font10 + ' bold').pack()
    Label(main_window, text='3 - Parameter Dataset Selection').pack(pady=5, padx=5, anchor='w')
    set_but = Button(main_window, text=' Select Parameter Datasets', compound='left', command=lambda: param_gui(config_ini_var.get()))
    set_but.pack()
    Label(main_window, text='', font=font10 + ' bold').pack()

    def start(ini_path):
        main_window.destroy()
        start_secproc(ini_path)

    Frame(main_window, bd=5, relief='sunken', height=2).pack(side='top', fill='x')
    bottom_frame = Frame(main_window).pack(fill=X)
    Button(bottom_frame, text='START', command=lambda: start(get_config()), font=font16 + ' bold').pack(side='left', fill=X, expand=True)
    Button(bottom_frame, text='EXIT', command=exit_all, font=font16 + ' bold').pack(side='left', fill=X, expand=True)
    main_window.mainloop()

def config_gui(config_file):
    config_window = Tk()
    config_window.focus_force()
    x, y = 600, 800
    config_window.geometry(f'{x}x{y}+{(config_window.winfo_screenwidth() - x) // 2}+{(config_window.winfo_screenheight() - y) // 2}')
    config_window.title('CropSuite - Options')
    config_window.resizable(0, 1) #type:ignore

    font10 = f'Helvetica 10'
    font12 = f'Helvetica 12'
    font14 = f'Helvetica 14'

    Label(config_window, text='Config.ini Options\n', font=font12 + ' bold').pack()
    global config_dict
    config_dict = rci.read_ini_file(config_file)
    tabs = [val for val in list(config_dict.keys()) if not val.startswith('parameters.')]

    tab_parent = ttk.Notebook(config_window)
    tab_list = []
    for idx, tab in enumerate(tabs):
        tab_list.append(Frame(tab_parent))
        tab_parent.add(tab_list[idx], text=str(tab).capitalize())
    tab_parent.pack(expand=1, fill='both')

    # Tab Files
    Label(tab_list[0], text='Adjust Paths here',).pack(anchor='w', pady=5, padx=5)
    label_vals = []

    def check_dir(dir):
        if os.path.exists(dir):
            return True
        else:
            return False

    def select_path(main_key, sub_key):
        path = filedialog.askdirectory(initialdir=os.getcwd())
        if path:
            config_dict[main_key][sub_key] = path
            label_vals[list(config_dict[main_key].keys()).index(sub_key)].config(text=path)
            outline = 'green' if check_dir(path) else 'red'
            label_vals[list(config_dict[main_key].keys()).index(sub_key)].config(highlightthickness=2, highlightbackground=outline)
            config_window.update()

    def select_filepath(main_key, sub_key):
        path = filedialog.askopenfilename(initialdir=os.getcwd())
        if path:
            config_dict[main_key][sub_key] = path
            label_vals[list(config_dict[main_key].keys()).index(sub_key)].config(text=path)
            outline = 'green' if check_dir(path) else 'red'
            label_vals[list(config_dict[main_key].keys()).index(sub_key)].config(highlightthickness=2, highlightbackground=outline)
            config_window.update()

    files_lst = ['fine_dem', 'land_sea_mask', 'texture_classes']
    for key in config_dict.get(tabs[0]).keys(): #type:ignore
        frame = Frame(tab_list[0], border=4)
        frame.pack(fill=X)
        label_key = Label(frame, text=str(key))
        label_value = Label(frame, text=str(config_dict.get(tabs[0]).get(key)), borderwidth=2, relief='groove') #type:ignore
        outline = 'green' if check_dir(str(config_dict.get(tabs[0]).get(key))) else 'red' #type:ignore
        label_value.config(highlightthickness=2, highlightbackground=outline)
        label_vals.append(label_value)
        
        if str(key) in files_lst:
            button_select = Button(frame, text='Change', command=lambda k=key: select_filepath(tabs[0], k))
        else:
            button_select = Button(frame, text='Change', command=lambda k=key: select_path(tabs[0], k))

        label_key.pack(side='left')
        button_select.pack(side='right')
        label_value.pack(side='right')
    

    # Tab Options
    Label(tab_list[1], text='Adjust Options here').pack(anchor='w', pady=5, padx=5)
    try:
        dest_resolution_x, _ = dt.get_geotiff_resolution(config_dict.get(tabs[0]).get('fine_dem')) #type:ignore
    except:
        dest_resolution_x = 0

    resolution = np.round(dest_resolution_x * 3600, 0)
    unit = 'arcsec'
    if resolution > 60:
        resolution = np.round(dest_resolution_x * 60, 0)
        unit = 'arcmin'

    resolution_var = DoubleVar(tab_list[1], resolution)
    frame_resolution = Frame(tab_list[1])
    frame_resolution.pack(fill='x')
    resolution_label = Label(frame_resolution, text='Spatial output resolution: ')
    resolution_entry = Entry(frame_resolution, textvariable=resolution_var, state='disabled')
    resolutionunit_label = Label(frame_resolution, text=f' {unit}')
    resolution_label.pack(side='left')
    resolutionunit_label.pack(side='right')
    resolution_entry.pack(side='right')
    

    # Scheduler Checkbox
    scheduler_var = IntVar(tab_list[1], 1 if config_dict[tabs[1]]['use_scheduler'] else 0)
    use_scheduler_checkbox = Checkbutton(tab_list[1], text='Use Scheduler', variable=scheduler_var)
    use_scheduler_checkbox.pack(anchor='w')

    # Irrigation checkbox
    irrigation_var = IntVar(tab_list[1], 1 if config_dict[tabs[1]]['irrigation'] == '1' else 0)
    irrigation_checkbox = Checkbutton(tab_list[1], text='Irrigation', variable=irrigation_var)
    irrigation_checkbox.pack(anchor='w')
    

    def update_downscaling_options(*args):
        if str(temp_method_var.get()).strip() == 'Height Regression':
            winsize_entry.config(state='normal')
            usetempgrad_checkbox.config(state='normal')
        else:
            winsize_entry.config(state='disabled')
            usetempgrad_checkbox.config(state='disabled')

    frame_temp_method = Frame(tab_list[1], border=4)
    frame_temp_method.pack(fill=X)
    temp_methods = ('Nearest Neighbour', 'Bilinear Interpolation', 'Based on WorldClim Data', 'Height Regression')
    temp_method_var = StringVar(frame_temp_method, temp_methods[int(config_dict[tabs[1]]['temperature_downscaling_method'])])
    temp_method_label = Label(frame_temp_method, text="Temperature Downscaling Method")
    temp_method_dropdown = ttk.Combobox(frame_temp_method, textvariable=temp_method_var, width=40)
    temp_method_dropdown['values'] = temp_methods
    temp_method_label.pack(side='left')
    temp_method_dropdown.pack(side='right')
    temp_method_dropdown.bind("<<ComboboxSelected>>", update_downscaling_options)

    frame_prec_method = Frame(tab_list[1], border=4)
    frame_prec_method.pack(fill=X)
    prec_methods = ('Nearest Neighbour', 'Bilinear Interpolation', 'Based on WorldClim Data')
    prec_method_var = StringVar(frame_prec_method, prec_methods[int(config_dict[tabs[1]]['precipitation_downscaling_method'])])
    prec_method_label = Label(frame_prec_method, text="Precipitation Downscaling Method")
    prec_method_dropdown = ttk.Combobox(frame_prec_method, textvariable=prec_method_var, width=40)
    prec_method_dropdown['values'] = prec_methods
    prec_method_label.pack(side='left')
    prec_method_dropdown.pack(side='right')

    frame_output_type = Frame(tab_list[1], border=4)
    frame_output_type.pack(fill=X)
    output_types = ('GeoTIFF', 'NetCDF4', 'Cloud Optimized GeoTIFF')

    output_var = StringVar(frame_output_type, output_types[[out_dt.lower() for out_dt in output_types].index(config_dict[tabs[1]]['output_format'])])
    output_label = Label(frame_output_type, text='Output data format')
    output_dropdown = ttk.Combobox(frame_output_type, textvariable=output_var, width=40)
    output_dropdown['values'] = output_types
    output_label.pack(side='left')
    output_dropdown.pack(side='right')

    # Downscaling Window Size
    frame_winsize = Frame(tab_list[1])
    frame_winsize.pack(fill=X, padx=5, pady=5)
    winsize_var = IntVar(frame_winsize, int(config_dict[tabs[1]]['downscaling_window_size']))
    winsize_label = Label(frame_winsize, text='Downscaling Window Size')
    winsize_entry = Entry(frame_winsize, textvariable=winsize_var)
    winsize_entry_back = Label(frame_winsize, text=' Px     ')
    winsize_label.pack(side='left')
    winsize_entry_back.pack(side='right')
    winsize_entry.pack(side='right')

    # Use TempGrad
    usetempgrad_var = IntVar(tab_list[1], 1 if config_dict[tabs[1]]['downscaling_use_temperature_gradient'] else 0)
    usetempgrad_checkbox = Checkbutton(tab_list[1], text='Use Temperature Gradient', variable=usetempgrad_var)
    usetempgrad_checkbox.pack(anchor='w')

    # Temp Bias Threshold 
    frame_tbias = Frame(tab_list[1])
    frame_tbias.pack(fill=X, padx=5, pady=5)
    tbias_var = DoubleVar(frame_tbias, float(config_dict[tabs[1]]['downscaling_temperature_bias_threshold']))
    tbias_label = Label(frame_tbias, text='Temperature BIAS Threshold')
    tbias_entry = Entry(frame_tbias, textvariable=tbias_var)
    tbias_label_back = Label(frame_tbias, text=' K      ')
    tbias_label.pack(side='left')
    tbias_label_back.pack(side='right')
    tbias_entry.pack(side='right')

    # Prec Bias Threshold 
    frame_pbias = Frame(tab_list[1])
    frame_pbias.pack(fill=X, padx=5, pady=5)
    pbias_var = DoubleVar(frame_pbias, float(config_dict[tabs[1]]['downscaling_precipitation_bias_threshold']))
    pbias_label = Label(frame_pbias, text='Precipitation BIAS Threshold')
    pbias_entry = Entry(frame_pbias, textvariable=pbias_var)
    pbias_label_back = Label(frame_pbias, text=' mm  ')
    pbias_label.pack(side='left')
    pbias_label_back.pack(side='right')
    pbias_entry.pack(side='right')

    # Daily Prec Threshold 
    frame_pthre = Frame(tab_list[1])
    frame_pthre.pack(fill=X, padx=5, pady=5)
    pthre_var = DoubleVar(frame_pthre, float(config_dict[tabs[1]]['downscaling_precipitation_per_day_threshold']))
    pthre_label = Label(frame_pthre, text='Drizzle Threshold per Day')
    pthre_entry = Entry(frame_pthre, textvariable=pthre_var)
    pthre_label_back = Label(frame_pthre, text=' mm  ')
    pthre_label.pack(side='left')
    pthre_label_back.pack(side='right')
    pthre_entry.pack(side='right')

    # Turnaround Time
    turn_frame = Frame(tab_list[1])
    turn_frame.pack(fill='x', padx=5, pady=5)
    turnaround_lab = Label(turn_frame, text='Processing time between harvest and sowing for multiple cropping')
    turnaround_var = IntVar(turn_frame, config_dict[tabs[1]]['multiple_cropping_turnaround_time'] if 'multiple_cropping_turnaround_time' in config_dict[tabs[1]] else 21)
    turnaround_entry = Entry(turn_frame, textvariable=turnaround_var)
    turnaround_label_back = Label(turn_frame, text=' days ')
    turnaround_lab.pack(side='left')
    turnaround_label_back.pack(side='right')
    turnaround_entry.pack(side='right')

    output_soil = IntVar(tab_list[1], 1 if config_dict[tabs[1]]['output_soil_data'] else 0)
    output_soil_cb = Checkbutton(tab_list[1], text='Output weighted soil datasets', variable=output_soil)
    output_soil_cb.pack(anchor='w')
    
    # Remove Int Results
    remres_var = IntVar(tab_list[1], 1 if config_dict[tabs[1]]['remove_interim_results'] else 0)
    remres_checkbox = Checkbutton(tab_list[1], text='Remove interim results', variable=remres_var)
    remres_checkbox.pack(anchor='w')

    # Remove Downscaled Climate
    remdsclim_var = IntVar(tab_list[1], 1 if config_dict[tabs[1]].get('remove_downscaled_climate', 0) else 0)
    remdsclim_checkbox = Checkbutton(tab_list[1], text='Remove downscaled climate', variable=remdsclim_var)
    remdsclim_checkbox.pack(anchor='w')

    rest_label = Label(tab_list[1], text='')
    rest_label.pack(anchor='w', fill='both', expand=True)

    # Tab Extent
    Label(tab_list[2], text='Adjust Extent here').pack(anchor='w', pady=5, padx=5)
    Label(tab_list[2], text='').pack()
    
    top_coord_var = DoubleVar(tab_list[2], float(config_dict[tabs[2]]['upper_left_y']))
    top_coord_entry = Spinbox(tab_list[2], textvariable=top_coord_var, from_=-90, to=90, justify='center', width=10, increment=0.1)
    top_coord_entry.pack(side='top', padx=5, pady=5)

    frame_ext = Frame(tab_list[2])
    frame_ext.pack(fill='x')

    left_coord_var = DoubleVar(tab_list[2], float(config_dict[tabs[2]]['upper_left_x']))
    left_coord_entry = Spinbox(frame_ext, textvariable=left_coord_var, from_=-90, to=90, justify='center', width=10, increment=0.1)

    canvas = Canvas(frame_ext, relief='ridge')

    fig = plt.figure(figsize=(5, 5), dpi=100)  # 2.5x2.5 inches with 100 DPI to match the 250x250 pixel canvas
    global ax
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    global canvas_agg
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    canvas_agg = FigureCanvasTkAgg(fig, master=canvas)
    canvas_agg.get_tk_widget().pack(side='left')

    def update_map():
        x_min, x_max = float(config_dict[tabs[2]]['upper_left_x']), float(config_dict[tabs[2]]['lower_right_x'])
        y_min, y_max = float(config_dict[tabs[2]]['upper_left_y']), float(config_dict[tabs[2]]['lower_right_y'])
        map_x_min, map_x_max = np.clip(x_min-2.5, -180, x_max), np.clip(x_max+2.5, x_min, 180)
        map_y_min, map_y_max = np.clip(np.min([y_min, y_max])-2.5, -90, 90), np.clip(np.max([y_min, y_max])+2.5, -90, 90)
        ax.clear()
        ax.set_extent([map_x_min, map_x_max, map_y_min, map_y_max]) #type:ignore
        try:
            ax.add_feature(cfeature.COASTLINE)#type:ignore
            ax.add_feature(cfeature.BORDERS)#type:ignore
        except:
            pass
        rect = Rectangle((x_min, y_min), x_max-x_min, y_max-y_min, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        canvas_agg.draw()
    update_map()

    right_coord_var = DoubleVar(tab_list[2], float(config_dict[tabs[2]]['lower_right_x']))
    right_coord_entry = Spinbox(frame_ext, textvariable=right_coord_var, from_=-180, to=180, justify='center', width=10, increment=0.1)
    
    left_coord_entry.pack(side='left', padx=5)
    right_coord_entry.pack(side='right', padx=5)
    canvas.pack(side='left')

    bot_coord_var = DoubleVar(tab_list[2], float(config_dict[tabs[2]]['lower_right_y']))
    bot_coord_entry = Spinbox(tab_list[2], textvariable=bot_coord_var, from_=-180, to=180, justify='center', width=10, increment=0.1)
    bot_coord_entry.pack(side='top', pady=10)

    def on_entry_change(*args):
        brk = False
        try:
            x_min = float(left_coord_var.get())    
            left_coord_entry.config(highlightthickness=2, highlightbackground='green')   
        except:
            left_coord_entry.config(highlightthickness=2, highlightbackground='red')
            brk = True
        try:
            x_max = float(right_coord_var.get()) 
            right_coord_entry.config(highlightthickness=2, highlightbackground='green')          
        except:
            right_coord_entry.config(highlightthickness=2, highlightbackground='red')
            brk = True
        try:
            y_min = float(bot_coord_entry.get())     
            bot_coord_entry.config(highlightthickness=2, highlightbackground='green')      
        except:
            bot_coord_entry.config(highlightthickness=2, highlightbackground='red')
            brk = True
        try:
            y_max = float(top_coord_entry.get())       
            top_coord_entry.config(highlightthickness=2, highlightbackground='green')    
        except:
            top_coord_entry.config(highlightthickness=2, highlightbackground='red')
            brk = True
        
        if not brk:
            if float(x_min) > float(x_max):
                brk = True
                left_coord_entry.config(highlightthickness=2, highlightbackground='red')
                right_coord_entry.config(highlightthickness=2, highlightbackground='red')
            else:
                left_coord_entry.config(highlightthickness=2, highlightbackground='green')
                right_coord_entry.config(highlightthickness=2, highlightbackground='green')

        if not brk:
            if float(y_min) > float(y_max):
                brk = True
                bot_coord_entry.config(highlightthickness=2, highlightbackground='red')
                top_coord_entry.config(highlightthickness=2, highlightbackground='red')
            else:
                bot_coord_entry.config(highlightthickness=2, highlightbackground='green')
                top_coord_entry.config(highlightthickness=2, highlightbackground='green')

        if not brk:
            config_dict[tabs[2]]['upper_left_x'], config_dict[tabs[2]]['lower_right_x'] = float(x_min), float(x_max)
            config_dict[tabs[2]]['upper_left_y'], config_dict[tabs[2]]['lower_right_y'] = float(y_min), float(y_max)
            update_map()


    top_coord_var.trace('w', on_entry_change)
    bot_coord_var.trace('w', on_entry_change)
    left_coord_var.trace('w', on_entry_change)
    right_coord_var.trace('w', on_entry_change)

    # Tab Climate Variability
    Label(tab_list[3], text='Adjust Climate Variability Options here').pack(anchor='w', pady=5, padx=5)
    Label(tab_list[3], text='').pack()

    consvar_var = IntVar(tab_list[3], 1 if config_dict[tabs[3]]['consider_variability'] else 0)
    consvar_checkbox = Checkbutton(tab_list[3], text='Consider Climate Variability', variable=consvar_var)
    consvar_checkbox.pack(anchor='w')

    # Tab Membership Functions
    Label(tab_list[4], text='Adjust Membership Function Options here').pack(anchor='w', pady=5, padx=5)
    Label(tab_list[4], text='').pack()

    plotcrop_var = IntVar(tab_list[4], 1 if config_dict[tabs[4]]['plot_for_each_crop'] else 0)
    plotcrop_checkbox = Checkbutton(tab_list[4], text='Plot Membership Functions for each Crop', variable=plotcrop_var)
    plotcrop_checkbox.pack(anchor='w')


    def save(config_dict):
        config_window.update()
        config_dict['options']['use_scheduler'] = scheduler_var.get()
        config_dict['options']['irrigation'] = '1' if irrigation_var.get() else '0'
        config_dict['options']['temperature_downscaling_method'] = str(temp_methods.index(temp_method_var.get()))
        config_dict['options']['precipitation_downscaling_method'] = str(prec_methods.index(prec_method_var.get()))
        config_dict['options']['output_format'] = 'cog' if str(output_var.get()).lower() == 'cloud optimized geotiff' else str(output_var.get()).lower()
        config_dict['options']['downscaling_window_size'] = winsize_entry.get()
        config_dict['options']['downscaling_use_temperature_gradient'] = usetempgrad_var.get()
        config_dict['options']['downscaling_temperature_bias_threshold'] = tbias_entry.get()
        config_dict['options']['downscaling_precipitation_bias_threshold'] = pbias_entry.get()
        config_dict['options']['downscaling_precipitation_per_day_threshold'] = pthre_entry.get()
        config_dict['options']['remove_interim_results'] = remres_var.get()
        config_dict['options']['remove_downscaled_climate'] = remdsclim_var.get()
        config_dict['options']['output_soil_data'] = output_soil.get()
        config_dict['options']['multiple_cropping_turnaround_time'] = turnaround_entry.get()

        config_dict['extent']['upper_left_x'] = str(np.clip(float(left_coord_entry.get()), -180, 180))
        config_dict['extent']['upper_left_y'] = str(np.clip(float(top_coord_entry.get()), -90, 90))
        config_dict['extent']['lower_right_x'] = str(np.clip(float(right_coord_entry.get()), -180, 180))
        config_dict['extent']['lower_right_y'] = str(np.clip(float(bot_coord_entry.get()), -90, 90))

        config_dict['climatevariability']['consider_variability'] = consvar_var.get()

        config_dict['membershipfunctions']['plot_for_each_crop'] = plotcrop_var.get()

        ini = rci.write_config(config_dict, config_file)
        config_window.destroy()

    def cancel_confirm():
        confirm_win = Tk()
        x, y = 200, 80
        confirm_win.geometry(f'{x}x{y}+{(confirm_win.winfo_screenwidth() - x) // 2}+{(confirm_win.winfo_screenheight() - y) // 2}')
        confirm_win.title('Confirm')
        confirm_win.resizable(0, 0) #type:ignore
        Label(confirm_win, text='Cancel without saving\nConfirm?').pack()
        frame_ex = Frame(confirm_win)
        frame_ex.pack(fill='x')
        def conf():
            config_window.destroy()
            confirm_win.destroy()
        conf_but = Button(frame_ex, text='Confirm', command=conf, font=font10, width=15)
        canc_but = Button(frame_ex, text='Cancel', command=lambda: confirm_win.destroy(), font=font10, width=15)
        conf_but.pack(side='left')
        canc_but.pack(side='right')

    bottom_frame = Frame(config_window)
    bottom_frame.pack(fill=X)
    button_save = Button(bottom_frame, text='Save', command=lambda: save(config_dict), font=font14 + ' bold')
    button_exit = Button(bottom_frame, text='Cancel', command=cancel_confirm, font=font14 + ' bold')
    button_save.pack(side='left', fill=X, expand=True)
    button_exit.pack(side='left', fill=X, expand=True)
    update_downscaling_options()
    config_window.mainloop()

def get_key_by_value(dict, value):
    return list(dict.keys())[list(dict.values()).index(value)]

def get_all_keys(dict):
    return list(dict.keys())

def get_all_membershipfunction_names():
    all_pairs = []
    directory = os.path.join('plant_params', 'available')
    if not os.path.exists(os.path.join(directory, 'available')):
        print(f'No subdirectory "available" in {directory} found!')
        return []
    for filename in os.listdir(directory):
        if filename.endswith('.inf'):  # Assuming the files have a .txt extension
            with open(os.path.join(directory, filename), 'r') as file:
                try:
                    file_content = file.read()
                    pairs = []
                    lines = file_content.split('\n')
                    for line in lines:
                        if '_vals' in line:
                            key = line.split('=')[0].strip()
                            suit_key = key.replace('_vals', '_suit')
                            if any(suit_key in l for l in lines):
                                if key[:-5] not in all_pairs:
                                    all_pairs.append(key[:-5])
                except:
                    pass
    return all_pairs

def check_param_dict(param_dict):
    required_keys = {
        'data_directory': '.',
        'weighting_method': '0',
        'weighting_factors': [2, 1.5, 1, 0.75, 0.5, 0.25],
        'conversion_factor': 1.0,
        'no_data': None,
        'interpolation_method': 0,
        'rel_member_func': 'Not specified'
    }
    for key, default_value in required_keys.items():
        if key not in param_dict:
            param_dict[key] = default_value
    return param_dict

def exit_frame(gui, confirm=False):
    if not confirm:
        gui.destroy()
    else:
        confirm_win = Tk()
        confirm_win.title('Confirm')
        confirm_win.focus_force()
        x, y = 200, 150
        confirm_win.geometry(f'{x}x{y}+{(confirm_win.winfo_screenwidth() - x) // 2}+{(confirm_win.winfo_screenheight() - y) // 2}')
        confirm_win.resizable(0, 0) #type:ignore
        Label(confirm_win, text='Cancel without saving\nConfirm?').pack()
        frame_ex = Frame(confirm_win)
        frame_ex.pack(fill='x')
        def conf():
            gui.destroy()
            confirm_win.destroy()
        conf_but = Button(frame_ex, text='Confirm', command=conf, width=15)
        canc_but = Button(frame_ex, text='Cancel', command=lambda: gui.destroy(), width=15)
        conf_but.pack(side='left')
        canc_but.pack(side='right')

def add_name(label):
    name_win = Tk()
    name_win.title(label)
    name_win.resizable(0, 0) #type:ignore
    name_win.focus_force()
    x, y = 200, 80
    name_win.geometry(f'{x}x{y}+{(name_win.winfo_screenwidth() - x) // 2}+{(name_win.winfo_screenheight() - y) // 2}')
    frm = Frame(name_win)
    frm.pack(anchor='w', padx=5, pady=5, fill='x')
    val = StringVar(name_win, '')
    lab = Label(frm, text=f'{label}: ')
    ent = Entry(frm, textvariable=val, width=25)
    lab.pack(side='left')
    ent.pack(side='right')
    def ret_val():
        global vl
        vl = val.get()
        name_win.destroy()
        return(vl)
    Button(name_win, text='Ok', command=ret_val).pack(pady=5,fill='x')
    name_win.wait_window(name_win)
    return vl

def new_crop_name():
    name_win = Tk()
    name_win.title('New Crop')
    name_win.resizable(0, 0) #type:ignore
    name_win.focus_force()
    x, y = 200, 100
    name_win.geometry(f'{x}x{y}+{(name_win.winfo_screenwidth() - x) // 2}+{(name_win.winfo_screenheight() - y) // 2}')
    frm = Frame(name_win)
    frm.pack(anchor='w', padx=5, pady=5, fill='x')
    val = StringVar(name_win, '')
    lab = Label(frm, text='Crop Name: ')
    ent = Entry(frm, textvariable=val, width=25)
    lab.pack(side='left')
    ent.pack(side='right')

    crops = [crop[1] for crop in get_available_crop_list()]
    frm2 = Frame(name_win)
    frm2.pack(anchor='w', padx=5, pady=5, fill='x')
    copy_val = IntVar(name_win, 0)
    copy_str = StringVar(name_win, crops[0])
    cbut = Checkbutton(frm2, variable=copy_val, text='Copy from ')
    drop = ttk.Combobox(frm2, textvariable=copy_str, width=20)
    drop['values'] = crops
    cbut.pack(side='left')
    drop.pack(side='right')

    def ret_val():
        global vl
        global copy
        vl = val.get()
        if copy_val.get():
            copy = copy_str.get()
        else:
            copy = None
        name_win.destroy()
        return vl, copy
    
    Button(name_win, text='Ok', command=ret_val).pack(pady=5,fill='x')
    name_win.wait_window(name_win)
    return vl, copy

def param_gui(config_file):
    pargui.ParamGUI(config_file=config_file)

def param_gui_bak(config_file):
    param_window = Tk()
    x, y = 450, 580
    param_window.geometry(f'{x}x{y}+{(param_window.winfo_screenwidth() - x) // 2}+{(param_window.winfo_screenheight() - y) // 2}')
    param_window.title('CropSuite - Parameter Datasets')
    param_window.resizable(0, 1) #type:ignore
    param_window.focus_force()
    
    font12 = f'Helvetica 12'

    Label(param_window, text='Parameter Datasets\n', font=font12 + ' bold').pack(anchor='w')
    global config_dict
    config_dict = rci.read_ini_file(config_file)
    vals = [val.split('.')[1] for val in list(config_dict.keys()) if val.startswith('parameters.')]

    param_dict = config_dict.get(f'parameters.{vals[0]}')
    param_dict = check_param_dict(param_dict)

    ttk.Separator(param_window, orient='horizontal').pack(padx=5, pady=5, fill='x')
    
    dropdown_param = StringVar(param_window) 
    dropdown_param.set(str(vals[0]))
    drop = ttk.Combobox(param_window, textvariable=dropdown_param, values=vals, state="readonly")
    #drop = OptionMenu(param_window, dropdown_param, *vals) 
    drop.pack(anchor='w', padx=5, pady=5, fill='x')

    def select_param_dir():
        print('Select Parameter Directory')
        param_dir = filedialog.askdirectory(initialdir=os.getcwd(), title='Select Parameter Directory')
        try:
            tifs = sum(1 for file in os.listdir(param_dir) if file.endswith('.tif'))
            if 0 <= tifs < 3:
                weight_method.set(get_key_by_value(weight_dict, 0))
            if 3 <= tifs < 6:
                weight_method.set(get_key_by_value(weight_dict, 1))
            if tifs >= 6:
                weight_method.set(get_key_by_value(weight_dict, 2))
            update_weighting()
        except:
            pass
        dir_entry_val.set(param_dir)
        param_window.update()

    ttk.Separator(param_window, orient='horizontal').pack(padx=5, pady=5, fill='x')

    dir_frame = Frame(param_window)
    dir_frame.pack(anchor='w', padx=5, pady=5, fill='x')
    dir_label = Label(dir_frame, text='Path: ')
    dir_entry_val = StringVar(dir_frame, param_dict.get('data_directory')) #type:ignore
    dir_entry = Entry(dir_frame, textvariable=dir_entry_val, width=50, state='disabled') 
    dir_sel_but = Button(dir_frame, text='Select Directory', command=select_param_dir)
    dir_label.pack(side='left')
    dir_sel_but.pack(side='right')
    dir_entry.pack(side='right', padx=5)

    ttk.Separator(param_window, orient='horizontal').pack(padx=5, pady=5, fill='x')

    weight_dict = {'First Layer only': 0, 'Topsoil/First Three Layers': 1, 'Full soil/Six Layers': 2}

    weight_frame = Frame(param_window)
    weight_frame.pack(anchor='w', padx=5, pady=5, fill='x')
    weight_label = Label(weight_frame, text='Weighting Method: ')
    weight_method = StringVar(weight_frame, get_key_by_value(weight_dict, int(param_dict.get('weighting_method')))) #type:ignore
    weight_dropdown = ttk.Combobox(weight_frame, textvariable=weight_method, width=40)
    weight_dropdown['values'] = get_all_keys(weight_dict)
    
    ttk.Separator(param_window, orient='horizontal').pack(padx=5, pady=5, fill='x')

    weight_label.pack(side='left')
    weight_dropdown.pack(side='right')

    factors_value1 = DoubleVar(param_window, float(param_dict.get('weighting_factors').split(',')[0])) #type:ignore
    factors_value2 = DoubleVar(param_window, float(param_dict.get('weighting_factors').split(',')[1])) #type:ignore
    factors_value3 = DoubleVar(param_window, float(param_dict.get('weighting_factors').split(',')[2])) #type:ignore
    factors_value4 = DoubleVar(param_window, float(param_dict.get('weighting_factors').split(',')[3])) #type:ignore
    factors_value5 = DoubleVar(param_window, float(param_dict.get('weighting_factors').split(',')[4])) #type:ignore
    factors_value6 = DoubleVar(param_window, float(param_dict.get('weighting_factors').split(',')[5])) #type:ignore

    frame1 = Frame(param_window)
    frame2 = Frame(param_window)
    frame3 = Frame(param_window)
    frame4 = Frame(param_window)
    frame5 = Frame(param_window)
    frame6 = Frame(param_window)

    factors_label1 = Label(frame1, text='Weighting Factor   0 -  25 cm: ')
    factors_entry1 = Entry(frame1, textvariable=factors_value1)
    factors_label2 = Label(frame2, text='Weighting Factor  25 -  50 cm: ')
    factors_entry2 = Entry(frame2, textvariable=factors_value2)
    factors_label3 = Label(frame3, text='Weighting Factor  50 -  75 cm: ')
    factors_entry3 = Entry(frame3, textvariable=factors_value3)
    factors_label4 = Label(frame4, text='Weighting Factor  75 - 100 cm: ')
    factors_entry4 = Entry(frame4, textvariable=factors_value4)
    factors_label5 = Label(frame5, text='Weighting Factor 100 - 125 cm: ')
    factors_entry5 = Entry(frame5, textvariable=factors_value5)
    factors_label6 = Label(frame6, text='Weighting Factor 125 - 200 cm: ')
    factors_entry6 = Entry(frame6, textvariable=factors_value6)

    factors_label1.pack(side='left')
    factors_entry1.pack(side='right')
    factors_label2.pack(side='left')
    factors_entry2.pack(side='right')
    factors_label3.pack(side='left')
    factors_entry3.pack(side='right')
    factors_label4.pack(side='left')
    factors_entry4.pack(side='right')
    factors_label5.pack(side='left')
    factors_entry5.pack(side='right')
    factors_label6.pack(side='left')
    factors_entry6.pack(side='right')

    frame1.pack(anchor='w', padx=5, pady=5, fill='x')
    frame2.pack(anchor='w', padx=5, pady=5, fill='x')
    frame3.pack(anchor='w', padx=5, pady=5, fill='x')
    frame4.pack(anchor='w', padx=5, pady=5, fill='x')
    frame5.pack(anchor='w', padx=5, pady=5, fill='x')
    frame6.pack(anchor='w', padx=5, pady=5, fill='x')

    ttk.Separator(param_window, orient='horizontal').pack(padx=5, pady=5, fill='x')

    frame_convvfact = Frame(param_window)
    frame_convvfact.pack(anchor='w', padx=5, pady=5, fill='x')
    convfact_val = DoubleVar(param_window, float(param_dict.get('conversion_factor'))) #type:ignore
    conv_fact_label = Label(frame_convvfact, text='Conversion Factor: ')
    conv_fact_entry = Entry(frame_convvfact, textvariable=convfact_val, width=40)
    conv_fact_label.pack(side='left')
    conv_fact_entry.pack(side='right')

    nodata_frame = Frame(param_window)
    nodata_frame.pack(anchor='w', padx=5, pady=5, fill='x')
    nodata_val = DoubleVar(param_window, param_dict.get('no_data')) #type:ignore
    nodata_lab = Label(nodata_frame, text='No Data Value: ')
    nodata_entry = Entry(nodata_frame, textvariable=nodata_val, width=40)
    nodata_lab.pack(side='left')
    nodata_entry.pack(side='right')

    interp_dict = {'Linear': 0, 'Cubic': 1, 'Quadratic': 2, 'Spline': 3, 'Polygonal': 4, 'First Order Spline/sLinear': 5}

    interp_frame = Frame(param_window)
    interp_frame.pack(anchor='w', padx=5, pady=5, fill='x')
    interp_label = Label(interp_frame, text='Interpolation Method of Membership Function: ')
    interp_method = StringVar(interp_frame, get_key_by_value(interp_dict, int(param_dict.get('interpolation_method')))) #type:ignore
    interp_dropdown = ttk.Combobox(interp_frame, textvariable=interp_method, width=40)
    interp_dropdown['values'] = get_all_keys(interp_dict)
    
    interp_label.pack(side='left')
    interp_dropdown.pack(side='right')   

    avail_functions = sorted(get_all_membershipfunction_names())
    func_frame = Frame(param_window)
    func_frame.pack(anchor='w', padx=5, pady=5, fill='x')
    func_label = Label(func_frame, text='Corresponding Membership Function: ')
    func_val = StringVar(func_frame, str(param_dict.get('rel_member_func'))) #type:ignore
    func_dropdown = ttk.Combobox(func_frame, textvariable=func_val, width=40)
    func_dropdown['values'] = avail_functions
    
    func_label.pack(side='left')
    func_dropdown.pack(side='right')

    def update_param(*args):
        save()
        param = dropdown_param.get()
        param_dict = config_dict.get(f'parameters.{param}')
        param_dict = check_param_dict(param_dict)
        dir_entry_val.set(param_dict.get('data_directory')) #type:ignore
        weight_method.set(get_key_by_value(weight_dict, int(param_dict.get('weighting_method')))) #type:ignore
        convfact_val.set(float(param_dict.get('conversion_factor'))) #type:ignore
        interp_method.set(get_key_by_value(interp_dict, int(param_dict.get('interpolation_method')))) #type:ignore
        func_val.set(str(param_dict.get('rel_member_func'))) #type:ignore
        nodata_val.set(param_dict.get('no_data')) #type:ignore
        update_weighting()
        param_window.update()

    def update_weighting(*args):
        if int(weight_dict.get(weight_method.get())) == 0: #type:ignore
            factors_value1.set(1)
            factors_value2.set(0)
            factors_value3.set(0)
            factors_value4.set(0)
            factors_value5.set(0)
            factors_value6.set(0)
            factors_entry1.config(state='disabled')
            factors_entry2.config(state='disabled')
            factors_entry3.config(state='disabled')
            factors_entry4.config(state='disabled')
            factors_entry5.config(state='disabled')
            factors_entry6.config(state='disabled')
        elif int(weight_dict.get(weight_method.get())) == 1: #type:ignore
            factors_value1.set(1)
            factors_value2.set(1)
            factors_value3.set(1)
            factors_value4.set(0)
            factors_value5.set(0)
            factors_value6.set(0)
            factors_entry1.config(state='normal')
            factors_entry2.config(state='normal')
            factors_entry3.config(state='normal')
            factors_entry4.config(state='disabled')
            factors_entry5.config(state='disabled')
            factors_entry6.config(state='disabled')
        else:
            factors_value1.set(float(param_dict.get('weighting_factors').split(',')[0])) #type:ignore
            factors_value2.set(float(param_dict.get('weighting_factors').split(',')[1])) #type:ignore
            factors_value3.set(float(param_dict.get('weighting_factors').split(',')[2])) #type:ignore
            factors_value4.set(float(param_dict.get('weighting_factors').split(',')[3])) #type:ignore
            factors_value5.set(float(param_dict.get('weighting_factors').split(',')[4])) #type:ignore
            factors_value6.set(float(param_dict.get('weighting_factors').split(',')[5])) #type:ignore
            factors_entry1.config(state='normal')
            factors_entry2.config(state='normal')
            factors_entry3.config(state='normal')
            factors_entry4.config(state='normal')
            factors_entry5.config(state='normal')
            factors_entry6.config(state='normal')
        param_window.update()

    def save():
        param = dropdown_param.get()
        config_dict[f'parameters.{param}']['data_directory'] = dir_entry_val.get()
        config_dict[f'parameters.{param}']['weighting_method'] = weight_dict[weight_method.get()]
        config_dict[f'parameters.{param}']['weighting_factors'] = f'{factors_value1.get()},{factors_value2.get()},{factors_value3.get()},{factors_value4.get()},{factors_value5.get()},{factors_value6.get()}'
        config_dict[f'parameters.{param}']['conversion_factor'] = convfact_val.get()
        try:
            config_dict[f'parameters.{param}']['no_data'] = nodata_val.get()
        except:
            config_dict[f'parameters.{param}'].pop('no_data', None)
        config_dict[f'parameters.{param}']['interpolation_method'] = interp_dict[interp_method.get()]
        config_dict[f'parameters.{param}']['rel_member_func'] = func_val.get()
        rci.write_config(config_dict, config_file)
        #param_window.destroy()
        #param_gui(config_file)


    def add_param():
        new_name = add_name('Add Parameter Name')
        config_dict[f'parameters.{new_name}'] = {
            'data_directory': '.',
            'weighting_method': '0',
            'weighting_factors': '2,1.5,1,0.75,0.5,0.25',
            'conversion_factor': 1.0,
            'interpolation_method': 0,
            'rel_member_func': 'Not specified'
        }
        save()

    def rem_param():
        def rem():
            for key in get_all_keys(config_dict[f'parameters.{str(val.get())}']):
                config_dict[f'parameters.{str(val.get())}'].pop(key, None)
            config_dict.pop(f'parameters.{str(val.get())}', None) #type:ignore
            remparam_win.destroy()
            rci.write_config(config_dict, config_file)            

        remparam_win = Tk()
        remparam_win.title('Remove Parameter')
        remparam_win.resizable(0, 0) #type:ignore
        frm = Frame(remparam_win)
        frm.pack(anchor='w', padx=5, pady=5, fill='x')
        val = StringVar(remparam_win, sorted(vals)[0])
        lab = Label(frm, text='Remove Parameter: ')
        box = ttk.Combobox(frm, textvariable=val, width=40)
        box['values'] = sorted(vals)
        lab.pack(side='left')
        box.pack(side='right')
        Button(remparam_win, text='Ok', command=rem).pack(pady=5,fill='x')
        save()


    weight_dropdown.bind('<<ComboboxSelected>>', update_weighting)
    dropdown_param.trace("w", update_param)

    ttk.Separator(param_window, orient='horizontal').pack(padx=5, pady=5, fill='x')

    but_frame = Frame(param_window)
    but_frame.pack(anchor='w', padx=5, pady=5, fill='x')
    save_but = Button(but_frame, text='Save', command=save)
    add_but = Button(but_frame, text='Add', command=add_param)
    rem_but = Button(but_frame, text='Remove', command=rem_param)
    exit_but = Button(but_frame, text='Exit', command=lambda: exit_frame(param_window, confirm=False))
    save_but.pack(side='left')
    add_but.pack(side='left', padx=5)
    rem_but.pack(side='left', padx=5)
    exit_but.pack(side='right')
    
    #update_param()
    param_window.mainloop()

def get_available_crop_list():
    try:
        verified = r'U:\web\verified'
        not_verified = r'U:\web\not_verified'
        if not os.path.exists(verified):
            return []
        
        curr_crop_list = [f[:-4] for f in os.listdir(os.path.join('plant_params', 'available')) if str(f).endswith('.inf') and not str(f).startswith('._')]
        ver_crop_list = [f[:-4] for f in os.listdir(verified) if str(f).endswith('.inf') and not str(f).startswith('._')]
        unver_crop_list = [f[:-4] for f in os.listdir(not_verified) if str(f).endswith('.inf') and not str(f).startswith('._')]

        # [Installed, Crop, Verified]
        crop_list = []
        for crop in curr_crop_list:
            if crop in ver_crop_list:
                crop_list.append([True, crop, True])
            else:
                crop_list.append([True, crop, False])
        for crop in ver_crop_list:
            if not any(crop in sublist for sublist in crop_list):
                crop_list.append([False, crop, True])
        for crop in unver_crop_list:
            if not any(crop in sublist for sublist in crop_list):
                crop_list.append([False, crop, False])

        return sorted(crop_list, key=lambda x: x[1])
    except:
        return []
    
def upload():
    try:
        verified = r'U:\web\verified'
        not_verified = r'U:\web\not_verified'
        os.makedirs(verified, exist_ok=True)
        os.makedirs(not_verified, exist_ok=True)
        curr_crop_list = [f for f in os.listdir(os.path.join('plant_params', 'available')) if str(f).endswith('.inf') and not str(f).startswith('._')]
        for crop in curr_crop_list:
            if crop not in [f for f in os.listdir(verified) if str(f).endswith('.inf') and not str(f).startswith('._')]:
                if crop not in [f for f in os.listdir(not_verified) if str(f).endswith('.inf') and not str(f).startswith('._')]:
                    f = os.path.join(os.path.join('plant_params', 'available', crop))
                    shutil.copy(f, not_verified)
    except:
        pass

def update_treeview(tree, new_lst):
    # Clear existing items
    for item in tree.get_children():
        tree.delete(item)
    
    # Insert new items
    for item in new_lst:
        item[0] = '✓' if item[0] else '✗'
        item[1] = str(item[1]).capitalize()
        item[2] = '✓' if item[2] else '✗'
        tree.insert('', 'end', values=item)

def crop_install_gui(config_ini):
    crop_window = Tk()
    x, y = 540, 600
    crop_window.geometry(f'{x}x{y}+{(crop_window.winfo_screenwidth() - x) // 2}+{(crop_window.winfo_screenheight() - y) // 2}')
    crop_window.title('CropSuite - Crop Selection')
    crop_window.resizable(0, 1) #type:ignore
    crop_window.focus_force()
    font12 = f'Helvetica 12'
    font14 = f'Helvetica 14'
    
    #lst = get_available_crop_list()
    lst = []

    Label(crop_window, text='Crops available', font=font14 + ' bold').pack(anchor='w')
    tree = ttk.Treeview(crop_window, columns=('Installed', 'Crop', 'Verified'), show='headings', selectmode='extended')
    tree.heading('Installed', text='Installed')
    tree.heading('Crop', text='Crop')
    tree.heading('Verified', text='Verified')
    tree.column('Installed', width=50, anchor='center')
    tree.column('Crop', width=300, anchor='center')
    tree.column('Verified', width=50, anchor='center')
    update_treeview(tree, lst)
    vsb = ttk.Scrollbar(crop_window, orient='vertical', command=tree.yview)
    tree.configure(yscrollcommand=vsb.set)
    vsb.pack(side='right', fill='y')
    tree.pack(fill='both', expand=True)

    def new():
        plant_param_gui.open_gui(config_ini_path=config_ini, plant_inf=None)
        #upload()
        #lst = get_available_crop_list()
        lst = []
        update_treeview(tree, lst)
        crop_window.update()

    def save():
        selected_items = tree.selection()
        items = [tree.item(item)['values'][1]+'.inf' for item in selected_items]
        #web_source = r'U:\web'
        #for item in items:
        #    try:
        #        shutil.copy(os.path.join(web_source, 'verified', item), os.path.join('plant_params', 'available'))
        #    except:
        #        try:
        #            shutil.copy(os.path.join(web_source, 'not_verified', item), os.path.join('plant_params', 'available'))
        #        except:
        #            pass
        crop_window.destroy()

    def exit():
        crop_window.destroy()

    but_frame = Frame(crop_window)
    but_frame.pack(anchor='w', padx=5, pady=5)

    save_but = Button(but_frame, text='Save', command=save)
    save_but.pack(side='left', padx=5)

    new_but = Button(but_frame, text='Add New', command=new)
    new_but.pack(side='left', padx=5)

    exit_but = Button(but_frame, text='Exit', command=exit)
    exit_but.pack(side='right', padx=5)

    crop_window.wait_window()

def plant_gui(config_ini):
    plant_window = Tk()
    x, y = 540, 650
    plant_window.geometry(f'{x}x{y}+{(plant_window.winfo_screenwidth() - x) // 2}+{(plant_window.winfo_screenheight() - y) // 2}')
    plant_window.title('CropSuite - Crop Selection')
    plant_window.resizable(0, 1) #type:ignore
    plant_window.focus_force()

    font12 = f'Helvetica 12'
    font14 = f'Helvetica 14'

    Label(plant_window, text='Select the desired crops (Multiple selection possible)', font=font14 + ' bold').pack()

    def list_available():
        return sorted([crop.capitalize() for crop in os.listdir(os.path.join('plant_params', 'available')) if crop.endswith('.inf') and not crop.startswith('._')])

    def list_checked():
        # Return a list of indices for items that should be selected
        checked_indices = []
        for i, item in enumerate(sorted([crop for crop in os.listdir(os.path.join('plant_params', 'available')) if crop.endswith('.inf')])):
            if item in sorted([crop for crop in os.listdir(os.path.join('plant_params')) if crop.endswith('.inf')]):
                checked_indices.append(i)
        return checked_indices

    def select_all():
        for i in range(len(list_available())):
            listbox.select_set(i)

    def deselect_all():
        listbox.selection_clear(0, END)

    def open_file(item):
        file_path = os.path.join('plant_params', 'available', item)
        if platform.system() == "Windows": os.startfile(file_path) #type:ignore
        elif platform.system() == "Darwin": os.system("open " + file_path)
        else: os.system("xdg-open " + file_path)

    def edit(item):
        plant_param_gui.open_gui(config_ini_path=config_ini, plant_inf=item)
        #rt = Tk()
        #rt.withdraw()
        #plant_window = plant_param_gui.plant_param_gui(tk.Toplevel(rt), config_ini, item)
        #plant_window.root.wait_window(plant_window.root)
        plant_window.update()

    def new():
        crop_install_gui(config_ini)
        update_listbox(listbox, sorted(list_available()))
        plant_window.update()

    global item
    item = ''

    def on_right_click(event):
        global item
        item = listbox.get(listbox.nearest(event.y))
        try:
            listbox.selection_clear(0, END)
            listbox.selection_set(listbox.nearest(event.y))
            listbox.activate(listbox.nearest(event.y))
        except Exception as e:
            print(e)
        context_menu.post(event.x_root, event.y_root)

    def ok():
        for fn in [crop for crop in os.listdir('plant_params') if crop.endswith('.inf')]:
            try:
                os.remove(os.path.join('plant_params', fn))
            except:
                pass
        indices = listbox.curselection()
        selected_files = [listbox.get(index) for index in indices]
        [shutil.copy(os.path.join('plant_params', 'available', file_name), os.path.join('plant_params', file_name)) for file_name in selected_files]
        dest()

    def dest():
        plant_window.destroy()
    
    def on_close():
        return True

    plant_window.protocol("WM_DELETE_WINDOW", on_close)
    frame = Frame(plant_window)
    frame.pack(padx=20, pady=20)

    def update_listbox(listbox, new_lst):
        # Clear existing items
        listbox.delete(0, tk.END)
        
        # Insert new items
        for item in new_lst:
            listbox.insert(tk.END, item)

    listbox = Listbox(frame, selectmode=MULTIPLE, height=30, width=50)
    listbox.pack(side="left", fill="both", expand=True)

    update_listbox(listbox, list_available())

    #image_edit = PhotoImage(file=os.path.abspath(os.path.join(os.path.dirname(__file__), 'src', 'edit.png')), master=plant_window)
    #image_open = PhotoImage(file=os.path.abspath(os.path.join(os.path.dirname(__file__), 'src', 'notepad.png')), master=plant_window)
    #image_cancel = PhotoImage(file=os.path.abspath(os.path.join(os.path.dirname(__file__), 'src', 'cancel.png')), master=plant_window)

    context_menu = Menu(plant_window, tearoff=0)
    context_menu.add_command(label="Open Config File", compound='left', command=lambda: open_file(item))
    context_menu.add_command(label="Edit", compound='left', command=lambda: edit(item))
    context_menu.add_command(label="Cancel", compound='left', command=context_menu.unpost)

    if sys.platform == 'linux' or sys.platform == 'linux2' or sys.platform == 'win32' or sys.platform == 'win64':
        listbox.bind("<Button-3>", on_right_click)
    else:
        listbox.bind("<Button-2>", on_right_click)
    
    #listbox.bind("<Double-Button-1>", open_file)
    # Create buttons for select all and deselect all
    select_all_button = Button(plant_window, text="Select All", command=select_all)
    select_all_button.pack(side="left", padx=5)
    deselect_all_button = Button(plant_window, text="Deselect All", command=deselect_all)
    deselect_all_button.pack(side="left", padx=10)

    add_button = Button(plant_window, text='Add New', command=new)
    add_button.pack(side='left', padx=5)
    ok_button = Button(plant_window, text="Ok", command=ok)
    ok_button.pack(side="left", padx=10)
    ex_button = Button(plant_window, text="Cancel", command=dest)
    ex_button.pack(side="left", padx=5)
    #plant_window.wait_window()
    plant_window.mainloop()

def get_empty_crop_dict():
    def_plant = os.listdir(os.path.join('plant_params', 'available'))
    def_plant = [plant for plant in def_plant if plant.endswith('.inf')][1]
    def_plant = parse_file_to_dict(os.path.join('plant_params', 'available', def_plant))

    for key in def_plant.keys():
        if key.endswith('_vals') and key not in ['temp_vals', 'prec_vals', 'freqcropfail_vals']:
            def_plant[key] = [0, 1]
            def_plant[key.replace('_vals', '_suit')] = [1, 1]
    name, copy = new_crop_name()
    if copy == None:
        def_plant['name'] = name
        def_plant['growing_cycle'] = 1
        def_plant['temp_vals'] = [10, 15, 20, 25]
        def_plant['temp_suit'] = [0, 0.75, 1, 0]
        def_plant['prec_vals'] = [250, 500, 750, 1000]
        def_plant['prec_suit'] = [0, 0.75, 1, 0]
        def_plant['freqcropfail_vals'] = [0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25]
        def_plant['freqcropfail_vals'] = [1, 0.98, 0.95, 0.88, 0.73, 0.5, 0.27, 0.12, 0.05, 0.02, 0]
    else:
        def_plant = parse_file_to_dict(os.path.join('plant_params', 'available', f'{copy}.inf'))
        def_plant['name'] = name
    new_file = os.path.join('plant_params', 'available', f'{def_plant["name"]}.inf')
    return def_plant, new_file

def convert_to_number(value):
    if value.lower() == 'y': return True
    try:
        if '.' in value: return float(value)
        else:
            try:
                val = int(value)
                return val
            except: return str(value)
    except ValueError: return value

def parse_file_to_dict(file_path):
    data_dict = {}
    with open(file_path, 'r') as file:
        for line in file:
            if '=' in line:
                if line.startswith('AddCon:'):
                    key, value = line.split('=', 1)[0].strip(), line.split('=', 1)[1].strip()
                    data_dict[key] = value
                else:
                    key, value = line.split('=')
                    key, value = key.strip(), value.strip()
                    if ',' in value:
                        value = [convert_to_number(v.strip()) for v in value.split(',')]
                    else:
                        value = convert_to_number(value)
                    data_dict[key] = value
    return data_dict

class add_conditions_window: 
    def __init__(self, root, crop_dict):
        self.root = root

        crop = crop_dict.get('name', '').capitalize()

        x, y = 550, 400
        self.root.geometry(f'{x}x{y}+{(self.root.winfo_screenwidth() - x) // 2}+{(self.root.winfo_screenheight() - y) // 2}')
        self.root.title(f'Additional Conditions - {crop}')
        self.root.resizable(0, 0) #type:ignore
        self.root.focus_force()

        main_frame = tk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=1)

        self.canvas = tk.Canvas(main_frame)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.canvas.bind('<Configure>', lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        self.table_frame = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.table_frame, anchor="nw")
        headers = ["#", "Parameter", "Start Day", "End Day", "Condition", "Value"]

        widths = [3, 12, 5, 5, 5, 8]

        for idx, text in enumerate(headers):
            label = tk.Label(self.table_frame, text=text, padx=10, pady=5, width=widths[idx])
            label.grid(row=0, column=idx)

        self.row_data = []
        self.row_count = 0

        self.button_frame = tk.Frame(root)
        self.button_frame.pack(fill=tk.X)

        cancel_button = tk.Button(self.button_frame, text="Cancel", command=self.root.quit)
        cancel_button.pack(side=tk.LEFT, padx=5, pady=5)

        add_button = tk.Button(self.button_frame, text="Add", command=lambda: self.add_row(crop_dict))
        add_button.pack(side=tk.LEFT, padx=5, pady=5)

        remove_button = tk.Button(self.button_frame, text="Remove", command=self.remove_row)
        remove_button.pack(side=tk.LEFT, padx=5, pady=5)

        save_button = tk.Button(self.button_frame, text="Save", command=self.save_data)
        save_button.pack(side=tk.RIGHT, padx=5, pady=5)

        for i in range(100):
            if f'AddCon:{i}' in crop_dict.keys():
                first = 0 if crop_dict.get(f'AddCon:{i}').split(',')[0] == 'Temperature' else 1
                second = int(crop_dict.get(f'AddCon:{i}').split(',')[1])-1
                third = int(crop_dict.get(f'AddCon:{i}').split(',')[2])-1
                cond = crop_dict.get(f'AddCon:{i}').split(',')[3]
                fourth = 0 if cond == '>' else 1 if cond == '>=' else 2 if cond == '<=' else 3 
                fifth = float(crop_dict.get(f'AddCon:{i}').split(',')[4])
                self.add_row(crop_dict, values=[first, second, third, fourth, fifth])

        #prec_req_after_sow, prec_req_days = int(crop_dict.get('prec_req_after_sow', 20)), int(crop_dict.get('prec_req_days', 14)
        #self.add_row(crop_dict, values=[1, 0, prec_req_days, 1, prec_req_after_sow])

    def add_row(self, crop_dict, values=[0, 0, 0, 0, 0]):
        self.row_count += 1
        row_idx = self.row_count

        lgc = int(crop_dict.get('growing_cycle', 0))
        widths = [3, 12, 5, 5, 5, 8]

        row_num_label = tk.Label(self.table_frame, text=str(row_idx), padx=10, pady=5, width=widths[0])
        row_num_label.grid(row=row_idx, column=0)
        param_combobox = ttk.Combobox(self.table_frame, values=["Temperature", "Precipitation"], width=widths[1])
        param_combobox.grid(row=row_idx, column=1)
        start_combobox = ttk.Combobox(self.table_frame, values=list(range(1, lgc+2)), width=widths[2]) #type:ignore
        start_combobox.grid(row=row_idx, column=2)
        end_combobox = ttk.Combobox(self.table_frame, values=list(range(1, lgc+2)), width=widths[3]) #type:ignore
        end_combobox.grid(row=row_idx, column=3)
        condition_combobox = ttk.Combobox(self.table_frame, values=[">", ">=", "<=", "<"], width=widths[4])
        condition_combobox.grid(row=row_idx, column=4)
        entry_value = tk.Entry(self.table_frame, width=widths[5])
        entry_value.grid(row=row_idx, column=5)

        if values != [0, 0, 0, 0, 0]:
            param_combobox.current(values[0])
            start_combobox.current(values[1])
            end_combobox.current(values[2])
            condition_combobox.current(values[3])
            condition_combobox.current(values[3])
            entry_value.delete(0, END)
            entry_value.insert(0, values[4])
        self.row_data.append([param_combobox, start_combobox, end_combobox, condition_combobox, entry_value])

    def remove_row(self):
        if self.row_data:
            for widget in self.row_data[-1]:
                widget.grid_forget()  # Remove widgets
            self.row_data.pop()  # Remove from data structure
            self.row_count -= 1

    def save_data(self):
        data = []
        for i, row in enumerate(self.row_data, start=1):
            param = row[0].get()
            start = row[1].get()
            end = row[2].get()
            condition = row[3].get()
            val = row[4].get()
            data.append([i, param, start, end, condition, val])
        self.result = data
        self.data_saved = True  # Mark that data was saved
        self.root.destroy()
    
    def cancel(self):
        self.data_saved = False  # Indicate data was not saved
        self.root.destroy()

def exit_all():
    sys.exit()

def start_secproc(ini_path):
    CropSuiteGui(ini_path)

if __name__ == '__main__':

    loading_gui()
    # main_gui()

    # gui = CropSuiteGui()