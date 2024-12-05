from tkinter import * #type:ignore
from tkinter import ttk
import tkinter as tk
import os
import warnings
import numpy as np
try:
    import read_climate_ini as rci
except:
    from src import read_climate_ini as rci
import rasterio
from PIL import Image, ImageTk
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.colors as clr
from datetime import datetime
import matplotlib.cm as cm
import cartopy.crs as ccrs
import cartopy.feature as cfeature

warnings.filterwarnings('ignore')

class limfact_analyzer():
    def __init__(self, config_ini, current_path):
        self.current_path = current_path
        self.fact = 1 if os.name == 'nt' else 1.25
        self.config_file = rci.read_ini_file(config_ini)

        self.root = Tk()
        self.root.title('CropSuite - Data Viewer')
        x, y = 1200, 800
        self.root.geometry(f'{x}x{y}+{(self.root.winfo_screenwidth() - x) // 2}+{(self.root.winfo_screenheight() - y) // 2}')
        self.root.resizable(1, 1) #type:ignore
        self.root.focus_force()

        main_frame = tk.Frame(self.root, borderwidth=1, relief='ridge')
        main_frame.pack(side='left', fill='y')

        self.limiting_factors = self.get_limiting_factors(self.current_path)
        self.data_array, self.x_min, self.x_max, self.y_min, self.y_max, self.nodata = self.get_all_lim_factors(self.current_path)
        self.data_array[self.data_array < 0] = np.nan
        self.data_array[0] = 100 - self.data_array[0]
        self.data_array -= 100
        self.nan_mask = np.isnan(self.data_array)

        Label(main_frame, text='Limitation Analyzer', font='Helvetica 14 bold').pack(side='top', padx=5, pady=5, fill='x')
        Label(main_frame, text=os.path.basename(current_path).capitalize(), font='Helvetica 12 bold').pack(side='top', padx=5, pady=5, fill='x')
        ttk.Separator(main_frame, orient='horizontal').pack(padx=5, pady=5, fill='x')

        self.lim_val = tk.StringVar(self.root, self.limiting_factors[0][1])
        sel_param = ttk.Combobox(main_frame, textvariable=self.lim_val, width=20)
        sel_param['values'] = [b[1] for b in self.limiting_factors] + ['Combined']
        sel_param.pack(side='top', padx=5, pady=5, fill='x')

        self.colormap_list = {'suitability': ['white', 'darkgray', 'darkred', 'yellow', 'greenyellow', 'limegreen', 'darkgreen', 'darkslategray'],
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
        self.cmap_sel_var = tk.StringVar(self.root, 'YlOrRd_r')
        cmap_sel_box = ttk.Combobox(cmap_frm, textvariable=self.cmap_sel_var)
        cmap_sel_box['values'] = list(self.colormap_list.keys())
        cmap_sel_box.pack(side='right', padx=5, fill='x', expand=True)    

        self.proj_list = {'Plate Carrée': ccrs.PlateCarree,
                 'Robinson': ccrs.Robinson,
                 'Eckert IV': ccrs.EckertIV,
                 'AlbersEqualArea': ccrs.AlbersEqualArea,
                 'Mercator': ccrs.Mercator,
                 'Mollweide': ccrs.Mollweide,
                 'InterruptedGoodeHomolosine': ccrs.InterruptedGoodeHomolosine,
                 'NearsidePerspective': ccrs.NearsidePerspective,
                 'Aitoff': ccrs.Aitoff,
                 'EqualEarth': ccrs.EqualEarth,
                 'Hammer': ccrs.Hammer}

        proj_frm = Frame(main_frame)
        proj_frm.pack(side='top', pady=5, fill='x')
        Label(proj_frm, text='Projection:').pack(side='left', padx=5)
        self.proj_sel_var = tk.StringVar(self.root, 'Plate Carrée')
        proj_sel_box = ttk.Combobox(proj_frm, textvariable=self.proj_sel_var, width=20)
        proj_sel_box['values'] = list(self.proj_list.keys())
        proj_sel_box.pack(side='top', padx=5, fill='x')

        ttk.Separator(main_frame, orient='horizontal').pack(padx=5, pady=5, fill='x')

        transect_cb_frm = Frame(main_frame)
        transect_cb_frm.pack(side='top', pady=5, fill='x')
        self.trans_cb = IntVar(self.root, 0)
        self.trans_cb_box = Checkbutton(transect_cb_frm, text='Plot Transect', variable=self.trans_cb)
        self.trans_cb_box.pack(side='left', padx=5, fill='x', anchor='w')

        self.rad_vals = IntVar(self.root, 0)
        rad1_frm = Frame(main_frame)
        rad1_frm.pack(side='top', pady=5, fill='x')
        
        self.r1 = Radiobutton(rad1_frm, text='Vertical', variable=self.rad_vals, value=0, state='disabled')
        self.r1.pack(side='left', padx=5, fill='x', anchor='w')
        self.spb1_val = DoubleVar(self.root, (self.x_min + self.x_max) / 2)
        self.spb1 = Spinbox(rad1_frm, from_=self.x_min, to=self.x_max, width=5, textvariable=self.spb1_val, justify='center', increment=0.1, state='disabled')
        spb1_lab = Label(rad1_frm, text='° E')
        spb1_lab.pack(side='right', padx=5, fill='x', anchor='e')
        self.spb1.pack(side='right', padx=5, fill='x', anchor='e')
        
        rad2_frm = Frame(main_frame)
        rad2_frm.pack(side='top', pady=5, fill='x')

        self.r2 = Radiobutton(rad2_frm, text='Horizontal', variable=self.rad_vals, value=1, state='disabled')
        self.r2.pack(side='left', padx=5, fill='x', anchor='w')
        self.spb2_val = DoubleVar(self.root, (self.y_min + self.y_max) / 2)
        self.spb2 = Spinbox(rad2_frm, from_=self.y_min, to=self.y_max, width=5, textvariable=self.spb2_val, justify='center', increment=0.1, state='disabled')
        spb2_lab = Label(rad2_frm, text='° N')
        spb2_lab.pack(side='right', padx=5, fill='x', anchor='e')
        self.spb2.pack(side='right', padx=5, fill='x', anchor='e')
        
        self.trans_cb.trace("w", self.cb_changed)
        self.rad_vals.trace("w", self.rad_changed)

        smooth_cb_frm = Frame(main_frame)
        smooth_cb_frm.pack(side='top', pady=5, fill='x')
        self.smooth_val = IntVar(self.root, 0)
        self.smooth_cb = Checkbutton(smooth_cb_frm, variable=self.smooth_val, text='Smooth Line', state='disabled')
        self.smooth_cb.pack(side='left', padx=5, fill='x', anchor='w')
        self.smooth_win_val = IntVar(self.root, 100)
        self.smooth_spb = Spinbox(smooth_cb_frm, from_=0, to=500, width=5, textvariable=self.smooth_win_val, justify='center', increment=1, state='disabled')
        smd_lab = Label(smooth_cb_frm, text='Px')
        smd_lab.pack(side='right', padx=5, fill='x', anchor='e')
        self.smooth_spb.pack(side='right', padx=5, fill='x', anchor='e')
        ttk.Separator(main_frame, orient='horizontal').pack(padx=5, pady=5, fill='x')

        self.smooth_val.trace("w", self.smooth_changed)

        pointplot_frm = Frame(main_frame)
        pointplot_frm.pack(side='top', pady=5, fill='x')
        self.pointplot_val = IntVar(self.root, 0)
        self.pointplot_cb = Checkbutton(pointplot_frm, text='Plot data for Point', variable=self.pointplot_val)
        self.pointplot_cb.pack(side='left', padx=5, fill='x', anchor='w')

        ppoint_frm = Frame(main_frame)
        ppoint_frm.pack(side='top', pady=5, fill='x')
        self.y_val = DoubleVar(self.root, (self.y_min + self.y_max) / 2)
        self.y_spb = Spinbox(ppoint_frm, from_=self.y_min, to=self.y_max, width=5, textvariable=self.y_val, justify='center', increment=.1, state='disabled')
        nlab = Label(ppoint_frm, text='°N')
        self.x_val = DoubleVar(self.root, (self.x_min + self.x_max) / 2)
        self.x_spb = Spinbox(ppoint_frm, from_=self.x_min, to=self.x_max, width=5, textvariable=self.x_val, justify='center', increment=.1, state='disabled')
        elab = Label(ppoint_frm, text='°E')

        self.y_spb.pack(side='left', padx=5, fill='x', expand=True)
        nlab.pack(side='left', padx=5)
        self.x_spb.pack(side='left', padx=5, fill='x', expand=True)
        elab.pack(side='left', padx=5)

        self.pointplot_val.trace("w", self.plotpoint_changed)

        ttk.Separator(main_frame, orient='horizontal').pack(padx=5, pady=5, fill='x')

        self.frm_canvas = tk.Frame(self.root)
        self.frm_canvas.pack(side='left', fill='both', expand=True)

        self.canvas = tk.Canvas(master=self.frm_canvas)
        self.canvas.pack(side='right', fill='both', expand=True)

        sel_param.bind('<<ComboboxSelected>>', self.param_changed)

        Button(main_frame, text='Close Window', command=self.root.destroy).pack(side='bottom', padx=5, pady=5, fill='x')
        Button(main_frame, text='Plot', command=lambda: self.plot_current(None)).pack(side='bottom', padx=5, pady=5, fill='x')
        ttk.Separator(main_frame, orient='horizontal').pack(side='bottom', padx=5, pady=5, fill='x')
        
        self.root.mainloop()

    def plotpoint_changed(self, *kwargs):
        if self.pointplot_val.get() == 1:
            self.y_spb.config(state='normal')
            self.x_spb.config(state='normal')
            self.trans_cb.set(0)
            self.trans_cb_box.config(state='disabled')
            self.cb_changed()
        else:
            self.y_spb.config(state='disabled')
            self.x_spb.config(state='disabled')        
            self.trans_cb_box.config(state='normal') 
            self.cb_changed()  

    def smooth_changed(self, *kwargs):
        if self.smooth_val.get() == 1:
            self.smooth_spb.config(state='normal')
        else:
            self.smooth_spb.config(state='disabled')

    def cb_changed(self, *kwargs):
        if self.trans_cb.get() == 1:
            self.r1.config(state='normal')
            self.r2.config(state='normal')
            self.smooth_cb.config(state='normal')
            self.smooth_spb.config(state='normal')
            self.pointplot_val.set(0)
            self.pointplot_cb.config(state='disabled')
            self.x_spb.config(state='disabled')
            self.y_spb.config(state='disabled')
            if self.rad_vals.get() == 1:
                self.spb2.config(state='normal')
                self.spb1.config(state='disabled')
            else:
                self.spb1.config(state='normal')
                self.spb2.config(state='disabled')
        else:
            self.r1.config(state='disabled')
            self.r2.config(state='disabled')
            self.spb1.config(state='disabled')
            self.spb2.config(state='disabled')
            self.smooth_cb.config(state='disabled')
            self.smooth_spb.config(state='disabled')
            self.pointplot_cb.config(state='normal')
 
    def rad_changed(self, *kwargs):
        if self.rad_vals.get() == 1:
            self.spb2.config(state='normal')
            self.spb1.config(state='disabled')
        else:
            self.spb1.config(state='normal')
            self.spb2.config(state='disabled')

    def param_changed(self, event):
        self.cmap_sel_var.set('YlOrRd_r')

    def plot_data(self, vert_line=-999., hor_line=-999., point=[-999, -999]):
        if self.lim_val.get().lower() == 'combined':
            with rasterio.open(os.path.join(self.current_path, 'crop_suitability.tif'), 'r') as src:
                data = src.read(1)
                nodata = src.nodata
            data[data == nodata] = np.nan
            data = data - 100
        else:
            data = self.data_array[{name: number for number, name in self.limiting_factors}.get(self.lim_val.get())]
        min_val = -100
        max_val = 0
        label = f'Degree of Limitation - {self.lim_val.get()}'
        projection = self.proj_list.get(self.proj_sel_var.get(), ccrs.PlateCarree)()
        colormap = self.colormap_list.get(self.cmap_sel_var.get(), 'YlOrRd')
        if isinstance(colormap, list):
            cmap = clr.LinearSegmentedColormap.from_list('', colormap)
        else:
            cmap = cm.get_cmap(colormap)
        
        plt.ioff()
        fig = Figure(figsize=(5,7), facecolor='#f0f0f0', dpi=600)
        fig_legend = plt.figure(figsize=(3, 1))
        ax_legend = fig_legend.add_subplot(111)
        fig.set_facecolor('#f0f0f0')
        ax = fig.add_subplot(111, projection=projection)

        im = ax.imshow(data, extent=(self.x_min, self.x_max, self.y_min, self.y_max), origin='upper', cmap=cmap, transform=ccrs.PlateCarree(), vmin=min_val, vmax=max_val, interpolation='bilinear') #type:ignore
        if (abs(self.x_min) + abs(self.x_max) >= 300) or (abs(self.y_min) + abs(self.y_max) >= 120):
            ax.set_extent((-180, 180, -90, 90), crs=ccrs.PlateCarree())
        else:
            ax.set_extent((self.x_min, self.x_max, self.y_min, self.y_max), crs=ccrs.PlateCarree()) #type:ignore
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
        
        if vert_line != -999.:
            ax.axvline(x=vert_line, color='black', linewidth=3, linestyle='dotted', dash_capstyle = 'round')
        if hor_line != -999.:
            ax.axhline(y=hor_line, color='black', linewidth=3, linestyle='dotted', dash_capstyle = 'round')
        if point != [-999., -999.]:
            #ax.plot(point[1], point[0], color='white', marker='x', markersize=10, mew=5, transform=ccrs.PlateCarree())
            ax.plot(point[1], point[0], color='black', marker='x', markersize=8, mew=3, transform=ccrs.PlateCarree())
            

        ax_legend.axis('off')
        dummy_im = ax_legend.imshow([[min_val, max_val]], cmap=cmap, visible=False)
        cbar = fig_legend.colorbar(dummy_im, ax=ax_legend, orientation='horizontal', fraction=0.15, pad=0.2)
        cbar.set_label(label, fontsize=7)
        cbar.ax.tick_params(labelsize=7)

        title_text = f'{datetime.now().year}-{datetime.now().month}-{datetime.now().day}'+'\nFile: '+str(os.path.join(self.current_path, 'all_suitability_vals.tif'))+'\nCreated by CropSuite v1.0'+'\n© Florian Zabel, Matthias Knüttel 2025'
        ax.set_title(title_text, loc='left', fontsize=4, color='black')
        save_file = os.path.join(self.current_path, f'limitingfactors_{self.lim_val.get()}.png')
        legend_file = os.path.join(self.current_path, f'limitingfactors_{self.lim_val.get()}_legend.png')
        fig.savefig(save_file, bbox_inches='tight', pad_inches=.4, dpi=300)
        fig_legend.savefig(legend_file, bbox_inches='tight', dpi=300, transparent=True)
        fig_legend.clear()
        ax_legend.clear()
        del fig_legend, ax_legend
        return save_file, legend_file

    def plot_current(self, event):
        if self.trans_cb.get() == 1:
            ver_hor_val = self.rad_vals.get()
            merdian = self.spb1_val.get() if ver_hor_val == 0 else self.spb2_val.get()
            self.transect, self.savefig = self.plot_transect(ver_hor_val, merdian)
            currfig = self.merge_plot_transect(self.savefig, self.transect)
        elif self.pointplot_val.get() == 1:
            self.savefig, self.legend = self.plot_data(point=[self.y_val.get(), self.x_val.get()])
            self.bars, self.savefig = self.plot_point_data(y_coord=self.y_val.get(), x_coord=self.x_val.get()) #type:ignore
            currfig = self.merge_plot_transect(self.savefig, self.bars)
        else:
            self.savefig, self.legend = self.plot_data()
            currfig = self.merge_plot_legend()
        self.show_image_centered(currfig)

    def merge_plot_transect(self, fig_img_path, leg_img_path):
        fig_img, leg_image = Image.open(fig_img_path), Image.open(leg_img_path)
        second_width, second_height = fig_img.size
        legend_width, legend_height = leg_image.size
        new_width = second_width + legend_width
        new_height = max(second_height, legend_height)
        new_image = Image.new('RGBA', (new_width, new_height), (240, 240, 240, 255))  # type: ignore
        new_image.paste(fig_img, (0, 0))
        new_image.paste(leg_image, (second_width, 0), leg_image)
        new_image.save(fig_img_path, 'PNG')
        os.remove(leg_img_path)
        return fig_img_path

    def merge_plot_legend(self):
        fig_img, leg_image = Image.open(self.savefig), Image.open(self.legend)
        second_width, second_height = fig_img.size
        legend_width, legend_height = leg_image.size
        new_width, new_height = second_width, second_height + legend_height
        new_image = Image.new('RGBA', (new_width, new_height), (240, 240, 240, 255)) #type:ignore
        new_image.paste(fig_img, (0, 0))
        legend_x, legend_y = (new_width - legend_width) // 2, second_height
        new_image.paste(leg_image, (legend_x, legend_y), leg_image)
        new_image.save(self.savefig, 'PNG')
        os.remove(self.legend)
        return self.savefig

    def show_image_centered(self, img_path):
        image = Image.open(img_path)
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        image_ratio = image.width / image.height
        canvas_ratio = canvas_width / canvas_height
        if image_ratio > canvas_ratio:
            new_width = canvas_width
            new_height = int(new_width / image_ratio)
        else:
            new_height = canvas_height
            new_width = int(new_height * image_ratio)
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        image_tk = ImageTk.PhotoImage(resized_image, master=self.frm_canvas) #type:ignore
        x_center = canvas_width // 2
        y_center = canvas_height // 2
        self.canvas.create_image(x_center, y_center, image=image_tk, anchor='center')
        self.canvas.image = image_tk #type:ignore

    def get_limiting_factors(self, currpath):
        if os.path.exists(os.path.join(currpath, 'limiting_factor.inf')):
            with open(os.path.join(currpath, 'limiting_factor.inf')) as file:
                limiting_factors = [[int(value), factor] for value, factor in 
                                    (line.strip().split(" - ", 1) for line in file.readlines()[1:])]
        else:
            limiting_factors = []
        return limiting_factors

    def get_all_lim_factors(self, currpath):  
        with rasterio.open(os.path.join(currpath, 'all_suitability_vals.tif'), 'r') as src:
            data = src.read()
            bbox = src.bounds
            nodata = src.nodata
        return data, bbox.left, bbox.right, bbox.bottom, bbox.top, nodata

    def get_index(self, meridian, min_coord, max_coord, no_lines):
        num_columns = no_lines
        step = (max_coord - min_coord) / (num_columns - 1)
        column_index = int((meridian - min_coord) / step)
        column_index = max(0, min(column_index, num_columns - 1))
        return column_index

    def smooth_values(self, values, window_size=3):
        smoothed = []
        for i in range(len(values)):
            window = [values[j] for j in range(max(0, i - window_size + 1), min(i + window_size, len(values))) if not np.isnan(values[j])]
            
            if window:
                smoothed.append(np.mean(window))  # Calculate mean of the valid values in the window
            else:
                smoothed.append(np.nan)  # If no valid values, append NaN

        return smoothed

    def get_point_indices(self, coord_x, coord_y, x_min, x_max, y_min, y_max, data_shape):
        dx = (x_max - x_min) / (data_shape[1] - 1)
        dy = (y_max - y_min) / (data_shape[0] - 1)
        col = np.clip(int((coord_x - x_min) / dx), 0, data_shape[1] - 1)
        row = np.clip(int((coord_y - y_min) / dy), 0, data_shape[0] - 1)
        return row, col

    def plot_transect(self, ver_hor_val, meridian):
        color_mapping = {'temperature': 'red', 'precipitation': 'blue', 'climate variability': 'orange'}
        color = color_mapping.get(self.lim_val.get(), 'black')

        self.savefig, self.legend = self.plot_data(vert_line=-999. if ver_hor_val == 1 else meridian, hor_line=-999. if ver_hor_val == 0 else meridian)
        currfig = self.merge_plot_legend()

        if self.lim_val.get().lower() == 'combined':
            with rasterio.open(os.path.join(self.current_path, 'crop_suitability.tif'), 'r') as src:
                data = src.read(1)
                nodata = src.nodata
            data[data == nodata] = np.nan
            data = data - 100
        else:
            data = self.data_array[{name: number for number, name in self.limiting_factors}.get(self.lim_val.get())]

        if ver_hor_val == 0:
            line_number = self.get_index(meridian, self.x_min, self.x_max, data.shape[1])
            data = data[..., line_number]
            if self.lim_val.get().lower() == 'combined':
                mask = self.nan_mask[0, :, line_number]
            else:
                mask = self.nan_mask[{name: number for number, name in self.limiting_factors}.get(self.lim_val.get()), :, line_number]
        else:
            line_number = data.shape[0] - self.get_index(meridian, self.y_min, self.y_max, data.shape[0])
            data = data[line_number]
            if self.lim_val.get().lower() == 'combined':
                mask = self.nan_mask[0, line_number]
            else:
                mask = self.nan_mask[{name: number for number, name in self.limiting_factors}.get(self.lim_val.get()), line_number]
        if self.smooth_val.get() == 1:
            curr_data = data
            curr_data[np.isnan(curr_data)] = 0
            window_size = self.smooth_win_val.get()
            curr_data = np.asarray(self.smooth_values(curr_data, window_size))
        else:
            curr_data = data

        curr_data[mask] = np.nan
        curr_data[mask | (curr_data == 0)] = 1

        plt.figure(figsize=(5, 8), facecolor='#f0f0f0')
        if ver_hor_val == 0:
            coord_values = np.linspace(self.y_min, self.y_max, len(curr_data))
            plt.plot(curr_data, coord_values, color=color, linewidth=1)
            plt.fill_betweenx(coord_values, curr_data, 5, color=color, alpha=0.25) #type:ignore
            plt.ylim(np.min(coord_values), np.max(coord_values))  
            y_ticks = np.arange(self.y_min, self.y_max, 5)
            def format_coord_y(value):
                return f"{abs(value):.0f}° N" if value >= 0 else f"{abs(value):.0f}° S"
            y_tick_labels = [format_coord_y(tick) for tick in y_ticks]
            y_tick_labels = y_tick_labels[::-1]
            plt.yticks(y_ticks, y_tick_labels)
            plt.gca().invert_yaxis()
            plt.xlim([-100, 0])

        else:
            coord_values = np.linspace(self.x_min, self.x_max, len(curr_data))
            plt.plot(coord_values, curr_data, color=color, linewidth=1)
            plt.fill_between(coord_values, curr_data, color=color, alpha=0.25) #type:ignore
            plt.xlim(np.min(coord_values), np.max(coord_values))
            x_ticks = np.arange(self.x_min, self.x_max, 5)
            def format_coord_x(value):
                return f"{abs(value):.0f}° E" if value >= 0 else f"{abs(value):.0f}° W"
            x_tick_labels = [format_coord_x(tick) for tick in x_ticks]
            plt.xticks(x_ticks, x_tick_labels)
            plt.ylim([-100, 0])
        plt.gca().set_facecolor('#f0f0f0')
        plt.grid()
        plt.title(f'Degree of Limitation - {self.lim_val.get()}')
        plt.savefig(os.path.join(self.current_path, f'limitingfactors_{self.lim_val.get()}_transect.png'), bbox_inches='tight', pad_inches=.4, dpi=300)
        plt.close()
        del data
        del curr_data
        return os.path.join(self.current_path, f'limitingfactors_{self.lim_val.get()}_transect.png'), currfig

    def plot_point_data(self, y_coord, x_coord):
        self.savefig, self.legend = self.plot_data(point=[y_coord, x_coord])
        currfig = self.merge_plot_legend()

        y, x = self.get_point_indices(x_coord, y_coord, self.x_min, self.x_max, self.y_min, self.y_max, self.nan_mask[0].shape)
        y = self.nan_mask[0].shape[0] - y
        if y >= self.data_array.shape[1] or x >= self.data_array.shape[2]:
            print('Invalid Coordinates')
            print(f'Given coordinates: {y_coord} °N {x_coord} °E')
            print('Check Format [-]yy.yyyy °N [-]xxx.xxxx °E')
            return
        else:
            data = self.data_array[:, y, x]

            colors = ['red', 'blue', 'orange']
            colors.extend(['gray'] * (len(data) - len(colors)))

            plt.figure(figsize=(5, 8), facecolor='#f0f0f0')
            labels = [factor[1] for factor in self.limiting_factors]
            plt.bar(np.arange(len(data)), data, color=colors, alpha=.5)
            plt.gca().invert_yaxis()
            plt.ylim(-100, 0)
            plt.ylabel('Degree of Limitation')
            plt.xticks(ticks=np.arange(len(data)), labels=labels, rotation=90)
            plt.gca().set_facecolor('#f0f0f0')
            plt.title(f'Point at {y_coord} N {x_coord} E')
            plt.tight_layout()
            plt.savefig(os.path.join(self.current_path, f'limitingfactors_{self.lim_val.get()}_point.png'), bbox_inches='tight', pad_inches=.4, dpi=300)
            return os.path.join(self.current_path, f'limitingfactors_{self.lim_val.get()}_point.png'), currfig


if __name__ == '__main__':
    pass
    #limfact_analyzer(r"U:\Source Code\CropSuite\config.ini", r'U:\Source Code\CropSuite\results\isimip_europe_hrvar_var\Area_57N-4E-43N16E\winterrapeseed')