import tkinter as tk
from tkinter import ttk, filedialog
import os
try:
    import read_climate_ini as rci
    import preproc_gui as ppi
except:
    from src import read_climate_ini as rci
    from src import preproc_gui as ppi
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk #type:ignore
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import sys
from pathlib import Path

class ConfigGUI(tk.Tk):
    def __init__(self, config_ini_path):
        super().__init__()
        
        global config_dict
        self.config_dict = rci.read_ini_file(config_ini_path)
        self.config_ini_path = config_ini_path
        self.focus_force()
        x, y = 900, 850 if os.name == 'nt' else 950
        self.geometry(f'{x}x{y}+{(self.winfo_screenwidth() - x) // 2}+{(self.winfo_screenheight() - y) // 2}')
        self.title('CropSuite - Options')
        self.resizable(1, 1) #type:ignore
        self.create_widgets(config_ini_path)
    
    def create_widgets(self, config_ini_path):
        frame = tk.Frame(self)
        frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(frame, text='Current config file:').pack(side='left', padx=(0, 5))
        self.config_path_var = tk.StringVar(frame, value=config_ini_path)
        config_entry = tk.Entry(frame, state='disabled', textvariable=self.config_path_var)
        config_entry.pack(side='left', fill='x', expand=True)
        
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.path_tab = self.add_tab("Paths and Files")
        self.options_tab = self.add_tab("Options")
        self.extent_tab = self.add_tab("Extent")
        self.downscaling_tab = self.add_tab("Downscaling")
        self.param_tab = self.add_tab("Parameters")        
        self.clim_tab = self.add_tab('Climate Data Preprocessing')
        self.adapt_tab = self.add_tab('Management')

        self.fill_path_tab()
        self.fill_options_tab()
        self.fill_extent_tab()
        self.fill_downscaling_tab()
        self.fill_param_tab()
        self.fill_clim_tab()
        self.fill_adapt_tab()

        self.press_count = 0
        self.selected_extent = None

        button_frame = tk.Frame(self)
        button_frame.pack(fill='x', padx=10, pady=5)
        exit_button = tk.Button(button_frame, text="Cancel", command=self.destroy, width=25)
        exit_button.pack(side='left', padx=5)
        save_button = tk.Button(button_frame, text="Save Config File", command=self.save, width=25)
        save_button.pack(side='right', padx=5)

    def add_tab(self, title):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text=title)
        return frame

    ### PATHS AND FILES ###
    def create_path_entry(self, parent, label_text, var_name, config_key, ftype=False):
        frame = tk.Frame(parent)
        frame.pack(fill='x', padx=10, pady=5)
        tk.Label(frame, text=label_text).pack(side='left', padx=(0, 5))
        setattr(self, var_name, tk.StringVar(self, os.path.join(self.config_dict['files'].get(config_key, ''))))
        entry = tk.Entry(frame, textvariable=getattr(self, var_name))
        self.set_path_check(entry)
        entry.pack(side='left', fill='x', expand=True)
        tk.Button(frame, text='Change', command=lambda e=entry: self.select_path(e, config_key, ftype)).pack(side='left', padx=5)

    def fill_path_tab(self):
        self.create_path_entry(self.path_tab, 'Output directory:', 'output_dir_var', 'output_dir', ftype=False)
        self.create_path_entry(self.path_tab, 'Processed climate data:', 'climate_dir_var', 'climate_data_dir', ftype=False)
        self.create_path_entry(self.path_tab, 'Plant parametrization directory:', 'plant_param_dir_var', 'plant_param_dir', ftype=False)
        self.create_path_entry(self.path_tab, 'Digital elevation model:', 'fine_dem_var', 'fine_dem', ftype=True)
        self.create_path_entry(self.path_tab, 'Land sea mask:', 'land_sea_mask_var', 'land_sea_mask', ftype=True)
        self.create_path_entry(self.path_tab, 'Texture class configuration:', 'texture_classes_var', 'texture_classes', ftype=True)
        self.create_path_entry(self.path_tab, 'WorldClim temperature directory:', 'worldclim_temp_var', 'worldclim_temperature_data_dir', ftype=False)
        self.create_path_entry(self.path_tab, 'WorldClim precipitation directory:', 'worldclim_prec_var', 'worldclim_precipitation_data_dir', ftype=False)

    def select_path(self, entry, config_key, ftype=True):
        path = filedialog.askopenfilename(title="Select file") if ftype else filedialog.askdirectory(title="Select directory")
        if path:
            entry.delete(0, tk.END)
            entry.insert(0, path)
        self.set_path_check(entry)
        self.config_dict['files'][config_key] = os.path.join(entry.get())

    def set_path_check(self, entry):
        entry.config(highlightthickness=2, highlightbackground='green' if self.check_path(entry.get()) else 'red')

    def check_path(self, dir):
        return os.path.exists(dir)

    ### OPTIONS ###
    def checkbox_to_dict(self, var_name, section, config_key):
        self.config_dict[section][config_key] = getattr(self, var_name).get()

    def add_checkbox(self, parent, label, var_name, section, config_key):
        frame = tk.Frame(parent)
        frame.pack(fill='x', padx=10, pady=5)
        setattr(self, var_name, tk.IntVar(self, self.config_dict[section].get(config_key, 'n')))
        cb = tk.Checkbutton(frame, variable=getattr(self, var_name), text=label, command=lambda: self.checkbox_to_dict(var_name, section, config_key))
        cb.pack(side='left')
        return cb
    
    def combobox_to_dict(self, var_name, section, config_key, by_index=False, options=[]):
        if var_name == 'temp_downs_var':
            if ['Nearest Neighbour', 'Bilinear Interpolation', 'WorldClim', 'Height Regression'].index(getattr(self, 'temp_downs_var').get()) == 3:
                self.dswins.config(state='normal')
                self.tmpgrd.config(state='normal')
            else:
                self.dswins.config(state='disabled')
                self.tmpgrd.config(state='disabled')
        if by_index:
            self.config_dict[section][config_key] = options.index(getattr(self, var_name).get())
        else:
            self.config_dict[section][config_key] = (getattr(self, var_name).get()).lower()

    def add_combobox(self, parent, label, var_name, section, config_key, options, by_index=False):
        frame = tk.Frame(parent)
        frame.pack(fill='x', padx=10, pady=5)
        tk.Label(frame, text=label).pack(side='left', padx=5)
        if by_index:
            setattr(self, var_name, tk.StringVar(parent, value=options[int(self.config_dict[section].get(config_key, options[0]))]))
        else:
            setattr(self, var_name, tk.StringVar(parent, value=self.config_dict[section].get(config_key, options[0])))
        combo = ttk.Combobox(frame, textvariable=getattr(self, var_name), values=options, state='readonly', width=40)
        combo.pack(side='left')
        combo.bind("<<ComboboxSelected>>", lambda e: self.combobox_to_dict(var_name, section, config_key, by_index, options))
        return combo
    
    def spinbox_to_dict(self, var, section, config_key, min_val, max_val):
        try:
            if not min_val <= float(var.get()) <= max_val:
                raise ValueError
        except ValueError:
            var.set(min_val if float(var.get()) < min_val else max_val)
        self.config_dict[section][config_key] = var.get()

    def options_spinbox_events(self, spinbox, var, min_val, max_val, section, config_key):
        spinbox.bind("<FocusOut>", lambda e: self.spinbox_to_dict(var, section, config_key, min_val, max_val))

    def add_spinbox(self, parent, label, var_name, section, config_key, from_ = 1, to = 365):
        frame = tk.Frame(parent)
        frame.pack(fill='x', padx=10, pady=5)
        tk.Label(frame, text=label).pack(side='left', padx=5)
        initial_value = self.config_dict[section].get(config_key, 21)
        setattr(self, var_name, tk.IntVar(parent, value=initial_value)) 
        sb = ttk.Spinbox(frame, textvariable=getattr(self, var_name), width=10, increment=1, from_=from_, to=to)
        sb.pack(side='left')
        self.options_spinbox_events(sb, getattr(self, var_name), 1, 365, section, config_key)
        return sb

    def snap_to_nearest_resolution(self, value):
        getattr(self, 'resolution_slider').set(min([0, 1, 2, 3, 4, 5, 6], key=lambda x: abs(x - float(value))))
        self.config_dict['options']['resolution'] = getattr(self, 'resolution_slider').get()

    def add_radiobuttons(self, parent, label, var_name, section, config_key, options):
       
        initial_value = int(self.config_dict[section].get(config_key, 0))
        var = tk.IntVar(self, value=initial_value)
        setattr(self, var_name, var)
        
        for text, value in options:
            rb = tk.Radiobutton(parent, text=text, variable=var, value=value,
                                command=lambda: self.checkbox_to_dict(var_name, section, config_key))
            rb.pack(anchor='w', padx=5, pady=2)


    def fill_options_tab(self):
        gen_frm = ttk.LabelFrame(self.options_tab, text='General Options')
        gen_frm.pack(fill='x', padx=10, pady=5)
        self.add_checkbox(gen_frm, 'Use scheduler', 'use_scheduler_var', 'options', 'use_scheduler')
        self.add_checkbox(gen_frm, 'Output all limiting factors', 'output_all_lim_var', 'options', 'output_all_limiting_factors')
        self.add_checkbox(gen_frm, 'Remove interim results', 'rem_interims_var', 'options', 'remove_interim_results')
        self.add_checkbox(gen_frm, 'Output aggregated soil data', 'out_agg_soil_var', 'options', 'output_soil_data')

        msf_frm = ttk.LabelFrame(self.options_tab, text='Membership Functions')
        msf_frm.pack(fill='x', padx=10, pady=5)
        self.add_checkbox(msf_frm, 'Plot membership functions for each crop', 'plot_each_crop_var', 'membershipfunctions', 'plot_for_each_crop')

        cvr_frm = ttk.LabelFrame(self.options_tab, text='Climate Variability')
        cvr_frm.pack(fill='x', padx=10, pady=5)
        self.add_radiobuttons(cvr_frm, 'Consider climate variability:', 'cons_variability_var', 'climatevariability', 'consider_variability',\
                              [('Without variability', 0), ('With variability', 1), ('With and without variability', 2)] )

        out_frm = ttk.LabelFrame(self.options_tab, text='Data Format')
        out_frm.pack(fill='x', padx=10, pady=5)
        self.add_combobox(out_frm, 'Output data format', 'out_format_var', 'options', 'output_format', ['GeoTIFF', 'NetCDF4', 'Cloud Optimized GeoTIFF (COG)'])

        resolution_frm = ttk.LabelFrame(self.options_tab, text='Output Resolution')
        resolution_frm.pack(fill='x', padx=10, pady=5)
        setattr(self, 'resolution_slider', tk.IntVar(self, self.config_dict['options'].get('resolution', '0')))
        slider_frame = tk.Frame(resolution_frm)
        slider_frame.pack(fill=tk.X)
        # 0 = 0.5, 1 = 0.25, 2 = 0.1, 3 = 5 arcmin, 4 = 2.5 arcmin, 5 = 30 arcsec, 6 = 7.5 arcsec
        reso_slid = ttk.Scale(master=slider_frame, from_=0, to=6, variable=getattr(self, 'resolution_slider'), orient='horizontal',
                              length=650, command=self.snap_to_nearest_resolution)
        reso_slid.pack(pady=2)
        labels_frame = tk.Frame(slider_frame)
        labels_frame.pack(fill=tk.X, padx=50)
        labels = ["0.5°", "0.25°", '0.1°', '5 arcmin', '2.5 arcmin', '30 arcsec', '7.5 arcsec\nOnly\nSoil Data']
        for i, text in enumerate(labels):
            label = tk.Label(labels_frame, text=text, width=8)
            label.grid(row=0, column=i, sticky="")
            labels_frame.columnconfigure(i, weight=1)

        tur_frm = ttk.LabelFrame(self.options_tab, text='Multiple Cropping')
        tur_frm.pack(fill='x', padx=10, pady=5)
        self.add_checkbox(tur_frm, 'Consider crop rotations (for all combinations of selected crops)', 'crop_rot', 'options', 'consider_crop_rotation')
        self.add_spinbox(tur_frm, 'Processing time between harvest and\nsowing for multiple cropping [Days]', 'proc_time_var', 'options', 'multiple_cropping_turnaround_time')

    ### EXTENT ###

    def fill_extent_tab(self):
        x_max, x_min = float(self.config_dict['extent'].get('lower_right_x', '180')), float(self.config_dict['extent'].get('upper_left_x', '-180'))
        y_min, y_max = float(self.config_dict['extent'].get('lower_right_y', '-90')), float(self.config_dict['extent'].get('upper_left_y', '90'))
        self.x_min, self.x_max = tk.DoubleVar(self.extent_tab, value=x_min), tk.DoubleVar(self.extent_tab, value=x_max)
        self.y_min, self.y_max = tk.DoubleVar(self.extent_tab, value=y_min), tk.DoubleVar(self.extent_tab, value=y_max)

        self.rect_x_min, self.rect_x_max = tk.DoubleVar(self.extent_tab, value=x_min), tk.DoubleVar(self.extent_tab, value=x_max)
        self.rect_y_min, self.rect_y_max = tk.DoubleVar(self.extent_tab, value=y_min), tk.DoubleVar(self.extent_tab, value=y_max)

        self.rect = None
        self.start_x = None
        self.start_y = None

        self.create_extent_selection(self.extent_tab)
        self.create_canvas(self.extent_tab)

    def create_extent_selection(self, parent):
        frame = tk.LabelFrame(parent, text='Selected Extent')
        frame.pack(fill='x', padx=10, pady=5)

        nw_frame = tk.LabelFrame(frame, text='Upper-Left Corner')
        nw_frame.pack(side='left', padx=10, pady=5)
        self._create_spinbox(nw_frame, "Lat.", self.rect_y_max, -90, 90, "y_max")
        self._create_spinbox(nw_frame, "Lon.", self.rect_x_min, -180, 180, "x_min")

        label_frame = tk.Frame(frame)
        label_frame.pack(side='left', expand=True)
        tk.Label(label_frame, text='Draw a rectangle on the map by pressing the left mouse button.\nPlease make sure that no interactive tools are active for selecting an extent.').pack()
        
        se_frame = tk.LabelFrame(frame, text='Lower-Right Corner')
        se_frame.pack(side='right', padx=10, pady=5)
        self._create_spinbox(se_frame, "Lat.", self.rect_y_min, -90, 90, "y_min")
        self._create_spinbox(se_frame, "Lon.", self.rect_x_max, -180, 180, "x_max")

    def _create_spinbox(self, parent, label_text, var, min_val, max_val, var_name):
        frame = tk.Frame(parent)
        frame.pack(padx=5, pady=5)
        tk.Label(frame, text=label_text).pack(side='left', padx=5)
        spinbox = ttk.Spinbox(frame, textvariable=var, width=6, increment=0.1, from_=min_val, to=max_val)
        spinbox.pack(side='left', padx=5)
        spinbox.bind("<Return>", lambda e: self.validate_spbxs(var_name))
        spinbox.bind("<FocusOut>", lambda e: self.validate_spbxs(var_name))
        spinbox.bind("<<Increment>>", lambda e: self.validate_spbxs(var_name))
        spinbox.bind("<<Decrement>>", lambda e: self.validate_spbxs(var_name))
    
    def cartopy_shapefile_exists(self):
        if sys.platform == "win32":
            base_path = Path(os.environ["USERPROFILE"]) / ".local" / "share" / "cartopy" / "shapefiles" / "natural_earth" / "cultural"
        else:
            base_path = Path.home() / ".local" / "share" / "cartopy" / "shapefiles" / "natural_earth" / "cultural"
        shp_path = base_path / "ne_10m_admin_0_boundary_lines_land.shp"
        return shp_path.exists()

    def create_canvas(self, parent):
        frame = tk.Frame(parent)
        frame.pack(fill='both', expand=True)
        self.fig, self.ax = plt.subplots(figsize=(4, 4), dpi=100, subplot_kw={'projection': ccrs.PlateCarree()})
        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.fig.tight_layout(pad=1) #type: ignore
        self.fig.patch.set_facecolor(tuple(c / 65535 for c in frame.winfo_rgb(frame.cget("bg")))) #type: ignore
        self.ax.set_facecolor('white')

        if self.cartopy_shapefile_exists():
            try:
                #cartopy.io.shapereader.natural_earth(resolution='110m', category='physical', name='land')
                self.ax.coastlines() #type:ignore
                self.ax.add_feature(cfeature.LAND, facecolor='beige')#type:ignore
                self.ax.add_feature(cfeature.OCEAN, facecolor='skyblue')#type:ignore
                self.ax.add_feature(cfeature.BORDERS, edgecolor='black')#type:ignore
            except:
                pass

        self.canvas = FigureCanvasTkAgg(self.fig, master=frame)
        self.canvas.get_tk_widget().pack(side='bottom', fill='both', expand=True)
        self.canvas.get_tk_widget().bind("<MouseWheel>", self.zoom)
        self.canvas.get_tk_widget().bind("<Button-4>", self.zoom)
        self.canvas.get_tk_widget().bind("<Button-5>", self.zoom)

        self.canvas.mpl_connect("button_press_event", self.start_rectangle)
        self.motion_event_id = self.canvas.mpl_connect("motion_notify_event", self.update_rectangle)
        self.canvas.mpl_connect("button_release_event", self.end_rectangle)

        self.update_map_extent()
        self.toolbar = NavigationToolbar2Tk(self.canvas, frame)
        self.toolbar.update()
        self.toolbar.pack(side="top", fill="x")

        self.add_rectangle()

    def zoom(self, event, zoom_factor=1.2):
        scale = 1 / zoom_factor if event.delta > 0 else zoom_factor
        x_min, x_max = self.x_min.get(), self.x_max.get()
        y_min, y_max = self.y_min.get(), self.y_max.get()
        x_center, y_center = (x_max + x_min) / 2, (y_max + y_min) / 2
        x_range = (x_max - x_min) * scale
        y_range = (y_max - y_min) * scale
        self.x_min.set(max(x_center - x_range / 2, -180))
        self.x_max.set(min(x_center + x_range / 2, 180))
        self.y_min.set(max(y_center - y_range / 2, -90))
        self.y_max.set(min(y_center + y_range / 2, 90))
        self.update_map_extent()

    def start_rectangle(self, event):
        if self.ax.get_navigate_mode()in ['PAN', 'ZOOM']:
            return
        self.motion_event_id = self.canvas.mpl_connect("motion_notify_event", self.update_rectangle)
        self.start_x, self.start_y = event.xdata, event.ydata
        if self.start_x is None or self.start_y is None:
            return
        if hasattr(self, 'rect') and self.rect:
            self.rect.remove()
        self.rect = Rectangle((self.start_x, self.start_y), 0, 0, 
                              linewidth=2, edgecolor='red', facecolor='none')
        self.ax.add_patch(self.rect)
        self.canvas.draw()

    def update_rectangle(self, event):
        if self.ax.get_navigate_mode()in ['PAN', 'ZOOM']:
            return
        if self.start_x is None or self.start_y is None or event.xdata is None or event.ydata is None:
            return
        width = event.xdata - self.start_x
        height = event.ydata - self.start_y
        self.rect.set_width(width) #type:ignore
        self.rect.set_height(height) #type:ignore
        self.canvas.draw()

    def end_rectangle(self, event):
        if self.ax.get_navigate_mode()in ['PAN', 'ZOOM']:
            return
        if self.start_x is None or self.start_y is None or event.xdata is None or event.ydata is None:
            return
        x_min, x_max = sorted([self.start_x, event.xdata])
        y_min, y_max = sorted([self.start_y, event.ydata])
        self.rect.set_x(x_min) #type:ignore
        self.rect.set_y(y_min) #type:ignore
        self.rect.set_width(x_max - x_min) #type:ignore
        self.rect.set_height(y_max - y_min) #type:ignore
        self.rect_x_min.set(max(x_min, -180))
        self.rect_x_max.set(min(x_max, 180))
        self.rect_y_min.set(max(y_min, -90))
        self.rect_y_max.set(min(y_max, 90))
        self.x_min.set(max(x_min, -180))
        self.x_max.set(min(x_max, 180))
        self.y_min.set(max(y_min, -90))
        self.y_max.set(min(y_max, 90))
        self.update_map_extent()
        self.canvas.mpl_disconnect(self.motion_event_id)
        self.canvas.draw()

        self.config_dict['extent']['upper_left_x'] = self.rect_x_min.get()
        self.config_dict['extent']['upper_left_y'] = self.rect_y_max.get()
        self.config_dict['extent']['lower_right_x'] = self.rect_x_max.get()
        self.config_dict['extent']['lower_right_y'] = self.rect_y_min.get()

    def update_map_extent(self):
        x_min, x_max = self.x_min.get() - 2, self.x_max.get() + 2
        y_min, y_max = self.y_min.get() - 2, self.y_max.get() + 2
        self.ax.set_extent([max(x_min, -180), min(x_max, 180), max(y_min, -90), min(y_max, 90)])#type:ignore
        self.canvas.draw()

    def update_map_extent_spbxs(self):
        x_min, x_max = self.rect_x_min.get() - 2, self.rect_x_max.get() + 2
        y_min, y_max = self.rect_y_min.get() - 2, self.rect_y_max.get() + 2
        self.ax.set_extent([max(x_min, -180), min(x_max, 180), max(y_min, -90), min(y_max, 90)])#type:ignore
        self.canvas.draw()

    def validate_spbxs(self, var_name):
        constraints = {"y_max": (self.rect_y_min.get() + 0.1, 90), "y_min": (-90, self.rect_y_max.get() - 0.1),
                       "x_max": (self.rect_x_min.get() + 0.1, 180), "x_min": (-180, self.rect_x_max.get() - 0.1)}
        var = getattr(self, var_name)
        min_val, max_val = constraints[var_name]
        var.set(max(min(var.get(), max_val), min_val))
        self.update_map_extent_spbxs()
        self.add_rectangle()

        self.config_dict['extent']['upper_left_x'] = self.rect_x_min.get()
        self.config_dict['extent']['upper_left_y'] = self.rect_y_max.get()
        self.config_dict['extent']['lower_right_x'] = self.rect_x_max.get()
        self.config_dict['extent']['lower_right_y'] = self.rect_y_min.get()

    def add_rectangle(self):
        if hasattr(self, 'rect') and self.rect:
            self.rect.remove()
        x_min, x_max = self.rect_x_min.get(), self.rect_x_max.get()
        y_min, y_max = self.rect_y_min.get(), self.rect_y_max.get()
        self.rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor='red', facecolor='none')
        self.ax.add_patch(self.rect)
        self.canvas.draw()

        self.config_dict['extent']['upper_left_x'] = self.rect_x_min.get()
        self.config_dict['extent']['upper_left_y'] = self.rect_y_max.get()
        self.config_dict['extent']['lower_right_x'] = self.rect_x_max.get()
        self.config_dict['extent']['lower_right_y'] = self.rect_y_min.get()

    ### DOWNSCALING ###

    def set_inactive_height(self, event=None):
        if ['Nearest Neighbour', 'Bilinear Interpolation', 'WorldClim', 'Height Regression'].index(getattr(self, 'temp_downs_var').get()) == 3:
            self.dswins.config(state='normal')
            self.tmpgrd.config(state='normal')
        else:
            self.dswins.config(state='disabled')
            self.tmpgrd.config(state='disabled')

    def fill_downscaling_tab(self):
        met_frm = ttk.LabelFrame(self.downscaling_tab, text='Downscaling Methods')
        met_frm.pack(fill='x', padx=10, pady=5)
        self.add_combobox(met_frm, 'Temperature downscaling method', 'temp_downs_var', 'options', 'temperature_downscaling_method',
                          ['Nearest Neighbour', 'Bilinear Interpolation', 'WorldClim', 'Height Regression'], by_index = True)
        self.add_combobox(met_frm, 'Precipitation downscaling method', 'prec_downs_var', 'options', 'precipitation_downscaling_method',
                          ['Nearest Neighbour', 'Bilinear Interpolation', 'WorldClim'], by_index = True)
        self.add_combobox(met_frm, 'RRPCF file downscaling method', 'rrpcf_downs_var', 'options', 'rrpcf_interpolation_method',
                          ['Linear', 'Nearest', 'Cubic'])
        hgr_frm = ttk.LabelFrame(self.downscaling_tab, text='Temperature Downscaling: Height Regression Options')
        hgr_frm.pack(fill='x', padx=10, pady=5)

        dwwins_frm = tk.Frame(hgr_frm)
        dwwins_frm.pack(fill='x', padx=10, pady=5)
        tk.Label(dwwins_frm, text='Downscaling window size [Px]').pack(side='left', padx=5)
        setattr(self, 'downs_wind_size_var', tk.IntVar(self.downscaling_tab, value=self.config_dict['options'].get('downscaling_window_size', 5))) 
        self.dswins = ttk.Spinbox(dwwins_frm, textvariable=getattr(self, 'downs_wind_size_var'), width=10, increment=1, from_=1, to=24,
                                  command=lambda: self.spinbox_to_dict(getattr(self, 'downs_wind_size_var'), 'options', 'downscaling_window_size', 1, 24))
        self.dswins.pack(side='left')

        self.tmpgrd = self.add_checkbox(hgr_frm, 'Check for physical limits of temperature gradient', 'use_temp_grad_var', 'options', 'downscaling_use_temperature_gradient')
        #tcb.bind("<<ComboboxSelected>>", self.set_inactive_height)
        self.set_inactive_height()

        gop_frm = ttk.LabelFrame(self.downscaling_tab, text='Downscaling Options')
        gop_frm.pack(fill='x', padx=10, pady=5)

        lpr_frm = tk.Frame(gop_frm)
        lpr_frm.pack(fill='x', padx=10, pady=5)
        tk.Label(lpr_frm, text='Drizzle precipitation threshold per day [mm]').pack(side='left', padx=5)
        setattr(self, 'prec_thres_var', tk.DoubleVar(self.downscaling_tab, value=self.config_dict['options'].get('downscaling_precipitation_per_day_threshold', 1.0))) 
        self.prtres = ttk.Spinbox(lpr_frm, textvariable=getattr(self, 'prec_thres_var'), width=10, increment=.25, from_=0, to=5, format="%.2f",
                                  command=lambda: self.spinbox_to_dict(getattr(self, 'prec_thres_var'), 'options', 'downscaling_precipitation_per_day_threshold', 0, 5))
        self.prtres.pack(side='left')
        self.add_checkbox(gop_frm, 'Remove downscaled climate data', 'remdsclim', 'options', 'remove_downscaled_climate')

    ### ADAPTION ###

    def snap_to_nearest(self, value):
        getattr(self, 'liming_slider').set(min([0, 1, 2, 3], key=lambda x: abs(x - float(value))))
        self.config_dict['options']['simulate_calcification'] = getattr(self, 'liming_slider').get()

    def update_irrigation_main(self, value_string):
        if all(c in '0' for c in value_string):
            self.irrig_var.set(0)
            self.irrig_checkbox.state(['!selected', '!alternate'])
            self.config_dict['options']['irrigation'] = '0'
        elif all(c in '1' for c in value_string):
            self.irrig_var.set(1)
            self.irrig_checkbox.state(['selected', '!alternate'])
            self.config_dict['options']['irrigation'] = '1'
        else:
            self.irrig_var.set(-1)
            self.irrig_checkbox.state(['alternate'])
            self.config_dict['options']['irrigation'] = value_string

    def create_monthly_irrigs(self, frame):
        state_string = list('111111111111' if str(self.config_dict['options']['irrigation']).lower() in ['true', 't', '1'] else '000000000000')

        months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

        self.vars = []
        string_display = tk.StringVar()
        string_display.set("State: " + "".join(state_string))

        def update_state_string():
            for i in range(12):
                state_string[i] = '1' if self.vars[i].get() else '0'
            current = "".join(state_string)
            string_display.set("State: " + "".join(state_string))
            self.update_irrigation_main(current)

        for i, month in enumerate(months):
            var = tk.IntVar(frame, value=int(state_string[i]))
            cb = ttk.Checkbutton(frame, text=month, variable=var, command=update_state_string, state='DISABLED')
            cb.grid(row=i, column=0, sticky="w", padx=20, pady=2)
            self.vars.append(var)

        label = ttk.Label(frame, textvariable=string_display)
        label.grid(row=13, column=0, pady=10)

    def on_irrig_var_changed(self):
        bool_val = self.irrig_var.get()
        [var.set(bool_val) for var in self.vars]
        state_string = list('000000000000')
        for i, var in enumerate(self.vars):
            state_string[i] = '1' if var.get() else '0'
        current = "".join(state_string)
        self.update_irrigation_main(current)

    def create_irrigation(self, frame):
        self.irrig_var = tk.IntVar(self)
        irrigation_val = str(self.config_dict['options'].get('irrigation', 0)).lower()
        ttk.Style().theme_use('clam')

        cb_frame = tk.Frame(frame)
        cb_frame.pack(fill='x')

        self.irrig_checkbox = ttk.Checkbutton(cb_frame, text="Irrigation", variable=self.irrig_var, command=self.on_irrig_var_changed)
        self.irrig_checkbox.pack(side='left', padx=5, pady=5)

        if irrigation_val in ['y', 'yes', 'true', '1']:
            irrigation_val = '1'
        elif irrigation_val in ['n', 'no', 'false', '0']:
            irrigation_val = '0'

        if irrigation_val == '0':
            self.irrig_var.set(0)
            self.irrig_checkbox.state(['!selected', '!alternate'])
        elif irrigation_val == '1':
            self.irrig_var.set(1)
            self.irrig_checkbox.state(['selected', '!alternate'])
        else:
            self.irrig_var.set(-1)
            self.irrig_checkbox.state(['alternate'])

        monthly_frame = tk.Frame(frame)
        monthly_frame.pack(fill='x')

        self.create_monthly_irrigs(monthly_frame)
        #self.irrig_var.trace_add('write', self.on_irrig_var_changed)

    def fill_adapt_tab(self):

        irg_frm = ttk.LabelFrame(self.adapt_tab, text='Irrigation')
        irg_frm.pack(fill='x', padx=10, pady=5)
        self.add_checkbox(irg_frm, 'Irrigation', 'irrigation_var', 'options', 'irrigation')

        #self.create_irrigation(irg_frm)

        lime_frm = ttk.LabelFrame(self.adapt_tab, text='Liming')
        lime_frm.pack(fill='x', padx=10, pady=5)
        setattr(self, 'liming_slider', tk.IntVar(self, self.config_dict['options'].get('simulate_calcification', '0')))
        slider_frame = tk.Frame(lime_frm)
        slider_frame.pack(fill=tk.X)
        qual_slid = ttk.Scale(master=slider_frame, from_=0, to=3, variable=getattr(self, 'liming_slider'), orient='horizontal', length=650, command=self.snap_to_nearest)
        qual_slid.pack(pady=2)
        labels_frame = tk.Frame(slider_frame)
        labels_frame.pack(fill=tk.X)
        labels = ["No Liming", "Low\npH increase of\nup to 0.5", "Medium\npH increase of\nup to 1.0", "High\npH increase of\nup to 1.5"]
        for i, text in enumerate(labels):
            label = tk.Label(labels_frame, text=text, width=20)
            label.grid(row=0, column=i, sticky="")
            labels_frame.columnconfigure(i, weight=1)

    ### PARAMETERS ###
    def fill_param_tab(self):
        param_frm = ttk.LabelFrame(self.param_tab, text='Parameters')
        param_frm.pack(fill='x', padx=10, pady=5)
        self.get_parameters = [key.split('.')[1] for key in self.config_dict if key.startswith('parameters.')]
        self.parameter_var = tk.StringVar(self)
        self.combobox = ttk.Combobox(param_frm, textvariable=self.parameter_var, values=self.get_parameters, state="readonly")
        self.combobox.pack(fill='x', padx=5, pady=5)
        if self.get_parameters:
            self.parameter_var.set(self.get_parameters[0])
            self.combobox.current(0)
        self.combobox.bind("<<ComboboxSelected>>", self.on_parameter_change)
        self.combobox.bind("<Return>", self.on_parameter_change)
        self.curr_param_frame = tk.Frame(param_frm)
        self.curr_param_frame.pack(fill='x', padx=10, pady=5)
        self.fill_current_param()

    def on_parameter_change(self, event=None):
        self.fill_current_param()

    def get_param_value(self, key, default=""):
        """Get the parameter value from the config_dict."""
        current_param = self.parameter_var.get()
        param_key = f'parameters.{current_param}'
        return self.config_dict[param_key].get(key)

    def set_param_value(self, key, value):
        """Set the parameter value in the config_dict."""
        current_param = self.parameter_var.get()
        param_key = f'parameters.{current_param}'
        self.config_dict.setdefault(param_key, {})[key] = value

    def create_labeled_entry(self, parent, label_text, var, **entry_kwargs):
        """Create a labeled entry widget."""
        row = tk.Frame(parent)
        row.pack(fill='x', pady=2)
        tk.Label(row, text=label_text).pack(side='left', padx=5)
        entry = tk.Entry(row, textvariable=var, **entry_kwargs)
        entry.pack(side='right', fill='x', expand=True, padx=5)
        return row

    def get_all_membershipfunction_names(self):
        all_pairs = []
        directory = os.path.join(self.config_dict['files'].get('plant_param_dir', 'plant_params'), 'available')
        for filename in os.listdir(directory):
            if filename.endswith('.inf'):
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

    def fill_current_param(self):
        """Fill the current parameter frame with widgets based on the current parameter."""
        
        def get_param_value(key, default=""):
            """Get the parameter value from the config_dict."""
            current_param = self.parameter_var.get()
            param_key = f'parameters.{current_param}'
            return self.config_dict.get(param_key, {}).get(key, default)

        def set_param_value(key, value):
            """Set the parameter value in the config_dict."""
            current_param = self.parameter_var.get()
            param_key = f'parameters.{current_param}'
            self.config_dict.setdefault(param_key, {})[key] = value

        def create_labeled_entry(parent, label_text, var, **entry_kwargs):
            """Create a labeled entry widget."""
            row = tk.Frame(parent)
            row.pack(fill='x', pady=2)
            tk.Label(row, text=label_text).pack(side='left', padx=5)
            entry = tk.Entry(row, textvariable=var, **entry_kwargs)
            entry.pack(side='right', fill='x', expand=True, padx=5)
            return row
        
        # === Prepare ===
        for widget in self.curr_param_frame.winfo_children():
            widget.destroy()

        current_param = self.parameter_var.get()
        param_key = f'parameters.{current_param}'

        # === Path Selection ===
        path_frame = tk.Frame(self.curr_param_frame)
        path_frame.pack(fill='x', pady=5)

        tk.Label(path_frame, text="Path:").pack(side="left", padx=(5, 2))

        self.path_var = tk.StringVar(master=self.curr_param_frame, value=get_param_value('data_directory'))
        tk.Entry(path_frame, textvariable=self.path_var).pack(side="left", fill='x', expand=True, padx=(0, 5))
        tk.Button(path_frame, text="Select", command=lambda: self.path_var.set(filedialog.askdirectory())).pack(side="right", padx=5)
        self.path_var.trace_add("write", lambda *_: set_param_value('data_directory', self.path_var.get()))

        # === Weighting Method ===
        self.weight_vars = []
        weighting_frame = tk.Frame(self.curr_param_frame)
        weighting_frame.pack(fill='x', pady=5)

        tk.Label(weighting_frame, text="Weighting Method:").pack(side="left", padx=(5, 2))

        method_options = {
            "0": "First Layer only",
            "1": "Top Soil/First Three Layers",
            "2": "All six Soil Layers"
        }
        reverse_method_options = {v.lower(): k for k, v in method_options.items()}

        method_code = get_param_value('weighting_method', '0')
        self.method_var = tk.StringVar(master=self.curr_param_frame, value=method_options.get(method_code, "First Layer only"))
        self.method_combobox = ttk.Combobox(
            weighting_frame,
            textvariable=self.method_var,
            values=list(method_options.values()),
            state="readonly",
            width=30
        )
        self.method_combobox.pack(side="right", padx=5)
        
        # === Weighting Factors Section ===
        self.weights_frame = tk.Frame(self.curr_param_frame)
        self.weights_frame.pack(fill='x', pady=(10, 5))

        def show_layer_weights(*_):
            layer_labels = [
                "Weighting Factor 0 - 25 cm",
                "Weighting Factor 25 - 50 cm",
                "Weighting Factor 50 - 75 cm",
                "Weighting Factor 75 - 100 cm",
                "Weighting Factor 100 - 125 cm",
                "Weighting Factor 125 - 200 cm"
            ]

            for widget in self.weights_frame.winfo_children():
                widget.destroy()
            method_code = int(reverse_method_options[self.method_var.get().lower()])
            if method_code == 2:
                stored_weights = get_param_value('weighting_factors', '2.0,1.5,1.0,0.75,0.5,0.25').split(',')                
                for i, label_text in enumerate(layer_labels):
                    var = tk.StringVar(master=self.curr_param_frame, value=stored_weights[i] if i < len(stored_weights) else '0.0')
                    create_labeled_entry(self.weights_frame, label_text, var)
                    self.weight_vars.append(var)

                    def update_weights(*_):
                        set_param_value('weighting_factors', ','.join(v.get() for v in self.weight_vars))
                    set_param_value('weighting_method', '2')
                    var.trace_add("write", update_weights)
            elif method_code == 1:
                set_param_value('weighting_method', '1')
                set_param_value('weighting_factors', '1.0,1.0,1.0,0.0,0.0,0.0')
            else:
                set_param_value('weighting_method', '0')
                set_param_value('weighting_factors', '1.0,0.0,0.0,0.0,0.0,0.0')

        self.method_combobox.bind("<<ComboboxSelected>>", show_layer_weights)
        self.method_combobox.bind("<Return>", show_layer_weights)
        show_layer_weights()

        # === Conversion Factor ===
        conv_frame = tk.Frame(self.curr_param_frame)
        conv_frame.pack(fill='x', pady=5)

        tk.Label(conv_frame, text="Conversion Factor:").pack(side="left", padx=(5, 2))
        self.conv_var = tk.StringVar(master=self.curr_param_frame, value=get_param_value('conversion_factor'))
        tk.Entry(conv_frame, textvariable=self.conv_var, width=30).pack(side="right", padx=(0, 5))
        self.conv_var.trace_add("write", lambda *_: set_param_value('conversion_factor', self.conv_var.get()))

        # === Interpolation Method ===
        interp_frame = tk.Frame(self.curr_param_frame)
        interp_frame.pack(fill='x', pady=5)

        tk.Label(interp_frame, text="Interpolation Method:").pack(side="left", padx=(5, 2))
        self.conv_var = tk.StringVar(master=self.curr_param_frame, value=get_param_value('interpolation_method'))

        interp_methods = {0: 'Linear', 1: 'Cubic', 2: 'Quadratic', 3: 'Spline', 4: 'Poly', 5: 'sLinear'}
        interp_values = list(interp_methods.values())
        self.conv_var = tk.StringVar(master=self.curr_param_frame, value=interp_methods.get(int(get_param_value('interpolation_method')), 'Linear'))
        interp_combobox = ttk.Combobox(interp_frame, textvariable=self.conv_var, values=interp_values, state="readonly", width=30)
        interp_combobox.pack(side="right", padx=(2, 5))
        
        def on_interpmethod_change(event=None):
            selected_method = interp_combobox.get()
            method_code = next((k for k, v in interp_methods.items() if v.lower() == selected_method.lower()), None)
            if method_code is not None:
                set_param_value('interpolation_method', str(method_code))

        interp_combobox.bind("<<ComboboxSelected>>", on_interpmethod_change)
        interp_combobox.bind("<Return>", on_interpmethod_change)
        
        # === Relating Membership Functions ===
        available_mfs = self.get_all_membershipfunction_names()
        mfs_frame = tk.Frame(self.curr_param_frame)
        mfs_frame.pack(fill='x', pady=5)

        tk.Label(mfs_frame, text="Corresponding Membership Function:").pack(side="left", padx=(5, 2))
        self.mfs_var = tk.StringVar(master=self.curr_param_frame, value=get_param_value('rel_member_func'))
        mfs_combobox = ttk.Combobox(mfs_frame, textvariable=self.mfs_var, values=available_mfs, state="readonly", width=30)
        mfs_combobox.pack(side="right", padx=(2, 5))   

        def on_mfs_change(event=None):
            selected_mfs = mfs_combobox.get()
            if selected_mfs in available_mfs:
                set_param_value('rel_member_func', selected_mfs)

        mfs_combobox.bind("<<ComboboxSelected>>", on_mfs_change)
        mfs_combobox.bind("<Return>", on_mfs_change)

        but_frm = tk.Frame(self.curr_param_frame)
        but_frm.pack(fill='x', pady=5)

        tk.Button(but_frm, text='Add new Parameter', command=self.add_param).pack(side='left', padx=5)
        tk.Button(but_frm, text='Remove selected Parameter', command=self.rem_param).pack(side='left', padx=5)

    #pgi.ParamGUI(self.config_ini_path, parent_frame=param_frm)

    def add_param(self):
        new_name = self.add_name('Add Parameter Name')
        self.config_dict[f'parameters.{new_name}'] = {
            'data_directory': '.',
            'weighting_method': '0',
            'weighting_factors': '1.0,0.0,0.0,0.0,0.0,0.0',
            'rel_member_func': 'None',
            'conversion_factor': 1.0,
            'interpolation_method': '0',
        }
        self.get_parameters.append(new_name)
        self.parameter_var.set(new_name)
        self.fill_current_param()
           
    def rem_param(self):
        curr_param = self.parameter_var.get()
        param_key = f'parameters.{curr_param}'
        if param_key in self.config_dict:
            del self.config_dict[param_key]
        self.parameter_var.set(self.get_parameters[0] if self.get_parameters else '')
        self.fill_current_param()

    def add_name(self, label):
        name_win = tk.Tk()
        name_win.title(label)
        name_win.resizable(0, 0) #type:ignore
        name_win.focus_force()
        x, y = 200, 80
        name_win.geometry(f'{x}x{y}+{(name_win.winfo_screenwidth() - x) // 2}+{(name_win.winfo_screenheight() - y) // 2}')
        frm = tk.Frame(name_win)
        frm.pack(anchor='w', padx=5, pady=5, fill='x')
        val = tk.StringVar(name_win, '')
        lab = tk.Label(frm, text=f'{label}: ')
        ent = tk.Entry(frm, textvariable=val, width=25)
        lab.pack(side='left')
        ent.pack(side='right')
        def ret_val():
            global vl
            vl = val.get()
            name_win.destroy()
            return(vl)
        tk.Button(name_win, text='Ok', command=ret_val).pack(pady=5,fill='x')
        name_win.wait_window(name_win)
        return vl

    ### CLIMATE ###
    def fill_clim_tab(self):
        clim_frm = ttk.LabelFrame(self.clim_tab, text='Climate Data')
        clim_frm.pack(fill='x', padx=10, pady=5)
        ppi.preproc_gui(clim_frm, self.config_ini_path, parent=clim_frm, config_ini_dict=self.config_dict)

    def reset(self):
        if self.rect:
            self.rect.remove()
            self.rect = None
        self.press_count = 0
        self.canvas.draw()

    def save(self):
        rci.write_config(self.config_dict, self.config_ini_path)
        print('Config written to file')
        self.destroy()


if __name__ == "__main__":
    #pass
    ConfigGUI(os.path.join('U:\\Source Code\\CropSuite\\config.ini')).mainloop()
    #app.mainloop()
