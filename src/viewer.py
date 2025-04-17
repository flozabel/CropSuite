import os
from tkinter import * #type:ignore
from tkinter import ttk
import tkinter as tk
from tkinter import filedialog
import numpy as np
import rasterio
from pathlib import Path
import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk #type:ignore
import matplotlib.cm as cm
from matplotlib.patches import Patch
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import matplotlib.colors as clr
from datetime import datetime

try:
    import limfact_analyzer as lfa
    import data_tools as dt
    import viewer_plot as vp
except:
    from src import limfact_analyzer as lfa
    from src import data_tools as dt
    from src import viewer_plot as vp

version = '1.2.0'

class ViewerGUI(tk.Toplevel):
    def __init__(self, start_path, master=None):
        super().__init__()
        self.current_path = Path(start_path)
        self.qual_val = tk.DoubleVar(master=self, value=50)
        self.focus_force()
        x, y = 1200, 800
        self.geometry(f'{x}x{y}+{(self.winfo_screenwidth() - x) // 2}+{(self.winfo_screenheight() - y) // 2}')
        self.title('CropSuite Viewer - Beta')
        self.resizable(1, 1) #type:ignore
        self.create_widgets()
        self.cmap_getter = vp.ColormapGetter()        
        
    def create_widgets(self):
        self.create_control_panel()
        self.create_map_frame()
    
    """
    def add_tab(self, title):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text=title)
        return frame
    """

    def create_control_panel(self):
        self.bcolor = self.winfo_rgb(self.cget("bg"))
        self.bcolor = "#{:02x}{:02x}{:02x}".format(self.bcolor[0] // 256, self.bcolor[1] // 256, self.bcolor[2] // 256)

        c_panel_frm = tk.Frame(self, width=280, height=600, bg=self.bcolor)
        c_panel_frm.pack(side='left', fill='both')

        #style = ttk.Style()
        #style.configure("TNotebook", tabposition="n")
        #self.notebook = ttk.Notebook(c_panel_frm, style="TNotebook")
        #self.notebook.pack(fill='both', expand=True, padx=5, pady=5)
        #self.main_tab = self.add_tab("Results")
        #self.climate_tab = self.add_tab("Climate Data")

        control_frame = tk.Frame(c_panel_frm, width=280, height=600, bg=self.bcolor)
        control_frame.pack(side='left', fill='y')
        self.fill_control_panel(control_frame)

        #climate_frame = tk.Frame(self.climate_tab, bg=self.bcolor)
        #climate_frame.pack(side='left', fill='both', expand=True)
        #self.fill_climate_frame(climate_frame)

    def create_map_frame(self):
        self.map_frame = tk.Frame(self)
        self.map_frame.pack(side='right', fill='both', expand=True)
        self.fig, self.ax = plt.subplots()
        self.bcolor = self.winfo_rgb(self.cget("bg"))
        self.bcolor = "#{:02x}{:02x}{:02x}".format(self.bcolor[0] // 256, self.bcolor[1] // 256, self.bcolor[2] // 256)
        self.fig.patch.set_facecolor(self.bcolor)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.map_frame)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.map_frame)
        self.toolbar.update()
        self.toolbar.pack(side="top", fill="x")
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        self.plot_empty()

    def fill_control_panel(self, parent):
        filesel_frame = ttk.LabelFrame(parent, text='File Selection')
        filesel_frame.pack()
        self.create_path_entry(filesel_frame)
        self.create_treeview_new(filesel_frame)
        up_button = ttk.Button(filesel_frame, text='Directory Up', command=self.navigate_up)
        up_button.pack(fill=tk.X, padx=5, pady=5)

        qual_frame = ttk.LabelFrame(parent, text='Display Quality')
        qual_frame.pack(fill='x')
        self.add_quality_slider(qual_frame)

        compare_frame = ttk.LabelFrame(parent, text='Compare')
        compare_frame.pack(fill='x')
        self.compare_options(compare_frame)

        plotopt_frame = ttk.LabelFrame(parent, text='Plotting Options')
        plotopt_frame.pack(fill='x')
        self.create_plotting_options(plotopt_frame)

        lfa_frame = ttk.LabelFrame(parent, text='Limiting Factors')
        lfa_frame.pack(fill='x')
        self.lfa_but = tk.Button(lfa_frame, text='Limiting Factor Analyzer', state='disabled', command=lambda: lfa.limfact_analyzer(os.path.dirname(self.path_entry.get())))
        self.lfa_but.pack(fill='x', padx=5, pady=5)

    ### FILE SELECTOR ###

    def create_treeview_new(self, parent):
        frame = tk.Frame(parent)
        frame.pack(fill=tk.BOTH, expand=True)

        self.h_scrollbar = ttk.Scrollbar(frame, orient="horizontal")
        self.h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)

        self.treeview = ttk.Treeview(frame, show="tree", xscrollcommand=self.h_scrollbar.set)
        self.treeview.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.treeview.column("#0", width=250)

        self.h_scrollbar.config(command=self.treeview.xview)
        
        v_scrollbar = ttk.Scrollbar(frame, orient="vertical", command=self.treeview.yview)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.treeview.configure(yscrollcommand=v_scrollbar.set)
        self.treeview.bind("<<TreeviewSelect>>", self.item_selected)

        try:
            self.file_image = tk.PhotoImage(master=self, file="src/file.png")
            self.folder_image = tk.PhotoImage(master=self, file="src/folder.png")
        except tk.TclError:
            self.file_image = tk.PhotoImage(master=self, width=16, height=16)
            self.folder_image = tk.PhotoImage(master=self, width=16, height=16)

        parent_iid = self.treeview.insert(parent='', index='0', text=os.getcwd(), open=True, image=self.folder_image) #type:ignore
        start_path = os.path.expanduser(os.getcwd())
        start_dir_entries = os.listdir(start_path)
        self.recursive_fill_treeview(parent_path=start_path, directory_entries=start_dir_entries,
                                     parent_iid=parent_iid, f_image=self.file_image, d_image=self.folder_image, treeview=self.treeview)

    def recursive_fill_treeview(self, parent_path, directory_entries, parent_iid, f_image, d_image, treeview):
        for name in sorted(directory_entries):
            item_path = parent_path+os.sep+name
            if os.path.isdir(item_path):
                subdir_iid = treeview.insert(parent=parent_iid, index='end', text=name, image=d_image)
                try:
                    subdir_entries = os.listdir(item_path)
                    self.recursive_fill_treeview(parent_path=item_path, directory_entries=subdir_entries, parent_iid=subdir_iid,
                                                 f_image=f_image, d_image=d_image, treeview=treeview)
                except PermissionError:
                    pass
            else:
                treeview.insert(parent=parent_iid, index='end', text=name, image=f_image)        

    def create_path_entry(self, parent):
        path_frame = ttk.Frame(parent)
        path_frame.pack(fill=tk.X, pady=5)
        self.path_entry = tk.Entry(path_frame, justify='right')
        self.path_entry.pack(fill=tk.X, padx=5, pady=5)
        self.path_entry.insert(0, str(self.current_path))
        self.path_entry.bind("<Return>", self.update_path)

    def load_tree_new(self, parent):
            self.treeview.delete(*self.treeview.get_children())
            parent_iid = self.treeview.insert(parent='', index='0', text=str(parent), open=True, image=self.folder_image) #type:ignore
            start_path = os.path.expanduser(parent)
            start_dir_entries = os.listdir(start_path)
            self.recursive_fill_treeview(parent_path=start_path, directory_entries=start_dir_entries,
                                        parent_iid=parent_iid, f_image=self.file_image, d_image=self.folder_image, treeview=self.treeview)
            self.treeview.xview_moveto(0)

    def update_path(self, event):
        new_path = Path(self.path_entry.get())
        if new_path.exists() and new_path.is_dir():
            self.current_path = new_path
            self.treeview.delete(*self.treeview.get_children())
            self.load_tree_new(self.current_path)

    def navigate_up(self):
        if self.current_path.parent.exists() and self.current_path != self.current_path.parent:
            self.current_path = self.current_path.parent
            self.load_tree_new(self.current_path)
            #self.load_tree(self.current_path)
            self.treeview.xview_moveto(0)
            self.path_entry.delete(0, tk.END)
            self.path_entry.insert(0, str(self.current_path))

    def get_full_path(self, item_iid):
        path_parts = []
        while item_iid:
            item_text = self.treeview.item(item_iid, "text")
            path_parts.insert(0, item_text)
            item_iid = self.treeview.parent(item_iid)
        return os.path.join(*path_parts)

    def item_selected(self, event):
        try:
            selected_item = self.treeview.selection()
            if not selected_item:
                return

            item_iid = selected_item[0]
            selected_path = self.get_full_path(item_iid)

            #selected_iid = self.treeview.selection()[0]
            #selected_path = self.fsobjects[selected_iid]
            self.path_entry.delete(0, tk.END)
            self.path_entry.insert(0, str(selected_path))
            self.set_default_values(os.path.basename(selected_path))

            allfacts = os.path.join(os.path.dirname(str(selected_path)), 'all_suitability_vals.tif')
            if os.path.exists(allfacts):
                self.lfa_but.config(state='normal')
            else:
                self.lfa_but.config(state='disabled')
            
            if os.path.splitext(os.path.basename(selected_path))[1] in ['.nc4', '.nc', '.tif', '.tiff']:
                self.plot_button.config(state='normal')
                self.check_extent_proj(selected_path)

                if self.comp_val.get() == 1:
                    av_mods = vp.get_models(str(selected_path))
                    if len(av_mods) > 0:
                        self.models_combobox.config(values = av_mods, state='normal')
                        self.models_combobox['values'] = av_mods
                        self.models_combobox.current(0)
                        self.mod_combo_changed(None, selected_value=str(av_mods[0]))
                    else:
                        old_path = self.models_combobox.get()
                        new_path = os.path.join(os.path.dirname(old_path), os.path.basename(selected_path))
                        if os.path.exists(new_path):
                            self.mod_combo_changed(None, selected_value=new_path)
                            self.models_combobox.config(state='normal')
                            self.models_combobox['values'] = [new_path]
                            self.models_combobox.after(100, lambda: self.models_combobox.current(0))
                            self.activate_compare()
                            #self.models_combobox.config(values=[new_path], state='normal')
                            #self.models_combobox.set(new_path)
            
                            self.mod_lab.config(text=f'Parent: {os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(new_path))))}')
                            self.are_lab.config(text=f'Directory: {os.path.basename(os.path.dirname(os.path.dirname(new_path)))}')
                            self.crp_lab.config(text=f'Crop: {os.path.basename(os.path.dirname(new_path))}')
                            self.fnm_lab.config(text=f'Filename: {os.path.basename(new_path)}')
                        else:
                            self.comp_val.set(0)
                            self.models_combobox.config(values = [], state='disabled')
                            self.selfile_but.config(state='disabled')
                else:
                    self.models_combobox.config(values = [], state='disabled')
                    self.selfile_but.config(state='disabled')
                    self.mod_lab.config(text='Parent:')
                    self.are_lab.config(text='Directory:')
                    self.crp_lab.config(text='Crop:')
                    self.fnm_lab.config(text='Filename:')
            else:
                self.plot_button.config(state='disabled')

        except:
            pass

    ### QUALITY SLIDER ###

    def snap_to_nearest(self, value):
        self.qual_val.set(min([0, 25, 50, 75, 100], key=lambda x: abs(x - float(value))))

    def add_quality_slider(self, parent):
        slider_frame = tk.Frame(parent)
        slider_frame.pack(fill=tk.X)
        qual_slid = ttk.Scale(master=slider_frame, from_=0., to=100., variable=self.qual_val, orient='horizontal', length=240, command=self.snap_to_nearest)
        qual_slid.pack(pady=2)
        labels_frame = tk.Frame(parent)
        labels_frame.pack(fill=tk.X)
        left_label = tk.Label(labels_frame, text="Quality")
        mid_label = tk.Label(labels_frame, text="         Auto")
        right_label = tk.Label(labels_frame, text="Performance")
        left_label.grid(row=0, column=0, sticky="w")    # Left aligned
        mid_label.grid(row=0, column=1, sticky="n")     # Center aligned
        right_label.grid(row=0, column=2, sticky="e")   # Right aligned
        labels_frame.columnconfigure(0, weight=1)
        labels_frame.columnconfigure(1, weight=1)
        labels_frame.columnconfigure(2, weight=1)


    ### COMPARE OPTIONS ###

    def compare_options(self, parent):
        comp_togram = vp.ToggledFrame(parent, 'Comparison options')
        comp_togram.pack(fill=tk.X)
        self.comp_val = tk.IntVar(self, 0)
        comp_cb = tk.Checkbutton(comp_togram.sub_frame, text='Compare', variable=self.comp_val, command=self.activate_compare)
        comp_cb.pack(padx=5, anchor='w')
        
        self.models_combobox = ttk.Combobox(comp_togram.sub_frame, values=[], state='disabled')
        self.models_combobox.pack(padx=5, anchor='w', fill='x', expand=True)
        self.models_combobox.bind('<<ComboboxSelected>>', self.mod_combo_changed)

        self.mod_lab = tk.Label(comp_togram.sub_frame, text='Parent: ')
        self.are_lab = tk.Label(comp_togram.sub_frame, text='Directory: ')
        self.crp_lab = tk.Label(comp_togram.sub_frame, text='Crop: ')
        self.fnm_lab = tk.Label(comp_togram.sub_frame, text='Filename: ')
        self.mod_lab.pack(padx=5, anchor='w')
        self.are_lab.pack(padx=5, anchor='w')
        self.crp_lab.pack(padx=5, anchor='w')
        self.fnm_lab.pack(padx=5, anchor='w')

        self.selfile_but = tk.Button(comp_togram.sub_frame, text='Select File', command=self.select_file, state='disabled')
        self.selfile_but.pack(padx=5, fill='x')

    def select_file(self):
        filetypes = [('Geospatial Datasets', '*.tif;*.tiff;*.nc;*.nc4')]
        file_path = filedialog.askopenfilename(title="Select a file", filetypes=filetypes)
        if file_path:
            self.models_combobox['values'] = [file_path]
            self.models_combobox.config(state='disabled')
            self.mod_combo_changed(None, selected_value=file_path)
            self.models_combobox.set(file_path)
            self.activate_compare()
        else:
            return None

    def activate_compare(self):
        if self.comp_val.get() == 1:
            self.selfile_but.config(state='normal')
            av_models = vp.get_models(str(self.path_entry.get()))
            if len(av_models) > 0:
                self.models_combobox.config(values = av_models, state='readonly')

            fname = os.path.splitext(os.path.basename(self.path_entry.get()))[0]
            if fname in ['climate_suitability', 'crop_suitability', 'soil_suitability', 'climate_suitability_mc']:
                self.cmap_box.current(22)
                self.min_val, self.max_val = -50, 50
                self.cbar_label = 'Suitability Change []'
                self.labels = ['Strong Decrease', 'Slight Decrease', 'No Changes', 'Slight Increase', 'Strong Increase']
            elif fname in ['multiple_cropping']:
                self.cmap_box.current(22)
                self.min_val, self.max_val = -2, 2
                self.cbar_label = 'Change in Potential Harvests []'
                self.labels = ['2 Harvests less', '1 Harvest less', 'Unchanged', '1 Harvest more', '2 Harvests more']
                self.nodata_patch = 'none'  
                self.default_classified = True          
            elif fname in ['optimal_sowing_date', 'optimal_sowing_date_mc_first', 'optimal_sowing_date_mc_second', 'optimal_sowing_date_mc_third',
                   'optimal_sowing_date_vernalization', 'optimal_sowing_date_with_vernalization', 'start_growing_cycle_after_vernalization']:
                self.cmap_box.current(24)
                self.min_val, self.max_val = -105, 105
                self.cbar_label = 'Shift of optimal Sowing Date [days]'
                self.labels = ['> 3 Months earlier', '2 Months earlier', '1 Month earlier', 'No significant change', '1 Month later', '2 Months later', '> 3 Months later']
            elif fname in ['suitable_sowing_days']:
                self.cmap_box.current(22)
                self.min_val, self.max_val = -365, 365
                self.cbar_label = 'Change in Suitable sowing days [days]'
                self.labels = ['Strong Decrease', 'Slight Decrease', 'No Changes', 'Slight Increase', 'Strong Increase']
            
        else:
            self.models_combobox.config(values = [], state='disabled')
            self.mod_lab.config(text='Parent:')
            self.are_lab.config(text='Directory:')
            self.crp_lab.config(text='Crop:')
            self.fnm_lab.config(text='Filename:')
            self.selfile_but.config(state='disabled')
            self.set_default_values(os.path.basename(self.path_entry.get()))

    def mod_combo_changed(self, event, selected_value=None):
        if selected_value is None:
            selected_value = self.models_combobox.get()
        curr_crop = os.path.basename(os.path.dirname(selected_value))
        curr_modl = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(selected_value))))
        curr_area = os.path.basename(os.path.dirname(os.path.dirname(selected_value)))
        self.mod_lab.config(text=f'Parent: {curr_modl}')
        self.are_lab.config(text=f'Directory: {curr_area}')
        self.crp_lab.config(text=f'Crop: {curr_crop}')
        self.fnm_lab.config(text=f'Filename: {os.path.basename(selected_value)}')


    ### PLOTTING OPTIONS ###

    def create_plotting_options(self, parent):
        cmaps = vp.ColormapGetter()
        cmaps = cmaps.get_all_colormaps()

        projs = vp.ProjectionGetter()
        projs = projs.get_all_projections()

        class_frame = tk.Frame(parent)
        class_frame.pack(fill=tk.X, pady=2, expand=True)
        self.class_val = tk.IntVar(self, False)
        self.class_cb = tk.Checkbutton(class_frame, variable=self.class_val, text='Classify Data')
        self.class_cb.pack(padx=5, side='left')

        cmap_frame = tk.Frame(parent)
        cmap_frame.pack(fill=tk.X, pady=2, expand=True)

        self.cmap_var = tk.StringVar(value=cmaps[0])
        self.proj_var = tk.StringVar(value=projs[0])

        tk.Label(cmap_frame, text="Colormap:").pack(side=tk.LEFT)
        self.cmap_box = ttk.Combobox(cmap_frame, values=cmaps, state="readonly")
        self.cmap_box.pack(side=tk.LEFT, padx=5, fill='x', expand=True)
        self.cmap_box.current(0)

        proj_frame = tk.Frame(parent)
        proj_frame.pack(fill=tk.X, pady=2, expand=True)
        tk.Label(proj_frame, text="Projection:").pack(side=tk.LEFT)
        self.proj_box = ttk.Combobox(proj_frame, values=projs, state="readonly")
        self.proj_box.pack(side=tk.LEFT, padx=5, fill='x', expand=True)
        self.proj_box.current(0)

        self.plot_button = ttk.Button(parent, text='Plot', command=self.plot, state='disabled')
        self.plot_button.pack(fill=tk.X, padx=5, pady=5)

    def check_extent_proj(self, filepath):
        extent = dt.get_extent(str(filepath))        
        x_range = extent.right - extent.left
        if x_range >= 160:
            self.proj_box.current(1)
        else:
            self.proj_box.current(0)

    def plot(self):
        filepath = self.path_entry.get()
        filename = os.path.basename(filepath)
        print(filepath)
        data, bounds = vp.read_geotiff(filepath=filepath, resolution=self.canvas.get_width_height()[0], resolution_mode=int(self.qual_val.get()))

        if self.comp_val.get() == 1 and os.path.exists(self.models_combobox.get()):
            comp_data, bounds = vp.read_geotiff(filepath=self.models_combobox.get(), resolution=self.canvas.get_width_height()[0], resolution_mode=int(self.qual_val.get()))
            if os.path.splitext(os.path.basename(self.path_entry.get()))[0] in ['optimal_sowing_date', 'optimal_sowing_date_mc_first', 'optimal_sowing_date_mc_second', 'optimal_sowing_date_mc_third',
                   'optimal_sowing_date_vernalization', 'optimal_sowing_date_with_vernalization', 'start_growing_cycle_after_vernalization']:
                data[(np.isnan(comp_data)) | (comp_data == 0)] = np.nan
                comp_data[(np.isnan(data)) | (data == 0)] = np.nan
                data = comp_data - data
                data[data > 182] = data[data > 182] - 180
                data[data < -182] = data[data < -182] + 180
            else:
                data = comp_data - data

        colormap = vp.ColormapGetter().get_colormap(self.cmap_box.get())
        projection = vp.ProjectionGetter().get_projection(self.proj_box.get())
        if isinstance(colormap, list):
            cmap = clr.LinearSegmentedColormap.from_list('', colormap)
            color_list = colormap
        else:
            cmap = cm.get_cmap(colormap)
            color_list = cmap(np.linspace(0, 1, 254))

        x_range = bounds.right - bounds.left
        grid_x = 5 if x_range <= 20 else 10 if x_range <= 60 else 30 if x_range <= 180 else 60
        
        y_range = bounds.top - bounds.bottom
        grid_y = 5 if y_range <= 20 else 10 if y_range <= 60 else 30

        if self.class_val.get():
            self.plot_on_canvas_discrete(data, bounds, color_list, self.labels, projection, grid_x, grid_y, self.nodata_patch, 3)
        else:
            self.plot_on_canvas_continous(data, bounds, cmap, projection, grid_x, grid_y)

    def set_default_values(self, f):
        f = os.path.splitext(f)[0]
        self.nodata_patch = 'none'
        self.nodata_val = -9999
        self.default_classified = False
        if f in ['climate_suitability', 'crop_suitability', 'soil_suitability', 'climate_suitability_mc']:
            if self.comp_val.get() == 1:
                self.cmap_box.current(22)
            else:
                self.cmap_box.current(1)
            self.class_cb.config(state='normal')
            self.class_val.set(False)
            self.min_val, self.max_val = 0, 100
            self.cbar_label = 'Suitability []'
            self.nodata_patch = 'first'
            self.labels = ['Unsuitable', 'Marginally Suitable', 'Moderately Suitable', 'Highly Suitable', 'Very highly Suitable']
        elif f in ['crop_limiting_factor']:
            self.cmap_box.current(21)
            self.class_val.set(True)
            self.class_cb.config(state='disabled')
            self.nodata_patch = 'last'
            self.min_val, self.max_val = 0, 13
            self.labels = vp.get_limiting_factors(self.path_entry.get()) + ['No Data']
            self.default_classified = True
        elif f in ['multiple_cropping']:
            if self.comp_val.get() == 1:
                self.cmap_box.current(22)
            else:
                self.cmap_box.current(3)
            self.class_val.set(True)
            self.class_cb.config(state='disabled')
            self.min_val, self.max_val = 0, 3
            self.cbar_label = 'Potential Harvests []'
            self.nodata_patch = 'first'
            self.default_classified = True
            self.labels = ['Unsuitable', 'One Harvest', 'Two Harvests', 'Three Harvests']
        elif f in ['optimal_sowing_date', 'optimal_sowing_date_mc_first', 'optimal_sowing_date_mc_second', 'optimal_sowing_date_mc_third',
                   'optimal_sowing_date_vernalization', 'optimal_sowing_date_with_vernalization', 'start_growing_cycle_after_vernalization']:
            if self.comp_val.get() == 1:
                self.cmap_box.current(24)
            else:
                self.cmap_box.current(7)
            self.class_cb.config(state='normal')
            self.class_val.set(False)
            self.min_val, self.max_val = 0, 365
            self.cbar_label = 'Optimal Sowing Date [doy]'
            self.labels = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
        elif f in ['suitable_sowing_days']:
            if self.comp_val.get() == 1:
                self.cmap_box.current(22)
            else:
                self.cmap_box.current(4)
            self.class_cb.config(state='normal')
            self.class_val.set(False)
            self.min_val, self.max_val = 0, 365
            self.cbar_label = 'Suitable Sowing Days [days]'
            self.labels = ['Less than 1 Month'] + [f'{i} Months' for i in range(2, 12)] + ['All year round']
            self.nodata_val = 0
        elif f in ['base_saturation', 'base_saturation_combined']:
            if self.comp_val.get() == 1:
                self.cmap_box.current(22)
            else:
                self.cmap_box.current(11)
            self.class_cb.config(state='normal')
            self.class_val.set(False)
            self.min_val, self.max_val = 0, 100
            self.cbar_label = 'Base Saturation [%]'
            self.labels = ['0 - 10 %', '10 - 20 %', '20 - 30 %', '30 - 40 %', '40 - 50 %', '50 - 60 %', '60 - 70 %', '70 - 80 %', '80 - 90 %', '90 - 100 %']
        elif f in ['coarse_fragments', 'coarse_fragments_combined']:
            if self.comp_val.get() == 1:
                self.cmap_box.current(22)
            else:
                self.cmap_box.current(12)
            self.class_cb.config(state='normal')
            self.class_val.set(False)
            self.min_val, self.max_val = 0, 100
            self.cbar_label = 'Coarse Fragments [%]'
            self.labels = ['0 - 10 %', '10 - 20 %', '20 - 30 %', '30 - 40 %', '40 - 50 %']
        elif f in ['gypsum', 'gypsum_combined']:
            if self.comp_val.get() == 1:
                self.cmap_box.current(22)
            else:
                self.cmap_box.current(13)
            self.class_cb.config(state='normal')
            self.class_val.set(False)
            self.min_val, self.max_val = 0, 100
            self.cbar_label = 'Gypsum Content [%]'
            self.labels = ['0 %', '1 %', '2 %', '3 %', '>4 %']
        elif f in ['pH', 'pH_combined']:
            if self.comp_val.get() == 1:
                self.cmap_box.current(22)
            else:
                self.cmap_box.current(14)
            self.class_cb.config(state='normal')
            self.class_val.set(False)
            self.min_val, self.max_val = 3, 11
            self.cbar_label = 'Soil pH []'
            self.labels = ['3-4', '4-5', '5-6', '6-7', '7-8', '8-9', '9-10', '10-11']
        elif f in ['salinity', 'salinity_combined']:
            if self.comp_val.get() == 1:
                self.cmap_box.current(22)
            else:
                self.cmap_box.current(15)
            self.class_cb.config(state='normal')
            self.class_val.set(False)
            self.min_val, self.max_val = 0, 16
            self.cbar_label = 'Salinity [ds/m]'
            self.labels = ['0 ds/m', '4 ds/m', '8 ds/m', '12 ds/m', '>16 ds/m']
        elif f in ['slope', 'slope_combined']:
            if self.comp_val.get() == 1:
                self.cmap_box.current(22)
            else:
                self.cmap_box.current(16)
            self.class_cb.config(state='normal')
            self.class_val.set(False)
            self.min_val, self.max_val = 0, 90
            self.cbar_label = 'Slope [°]'
            self.labels = ['0°', '5°', '10°', '>15°']
        elif f in ['sodicity', 'sodicity_combined']:
            if self.comp_val.get() == 1:
                self.cmap_box.current(22)
            else:
                self.cmap_box.current(17)
            self.class_cb.config(state='normal')
            self.class_val.set(False)
            self.min_val, self.max_val = 0, 15
            self.cbar_label = 'Sodicity [%]'
            self.labels = ['0 %', '5 %', '>10 %']
        elif f in ['organic_carbon', 'organic_carbon_combined', 'soil_organic_carbon_combined']:
            if self.comp_val.get() == 1:
                self.cmap_box.current(22)
            else:
                self.cmap_box.current(18)
            self.class_cb.config(state='normal')
            self.class_val.set(False)
            self.min_val, self.max_val = 0, 8
            self.cbar_label = 'Soil Organic Carbon Content [%]'
            self.labels = ['0 %', '2.5 %', '5 %', '7.5 %', '> 10 %']
        elif f in ['soildepth', 'soildepth_combined']:
            if self.comp_val.get() == 1:
                self.cmap_box.current(22)
            else:
                self.cmap_box.current(19)
            self.class_cb.config(state='normal')
            self.class_val.set(False)
            self.min_val, self.max_val = 0, 5
            self.cbar_label = 'Soil Depth [m]'
            self.labels = ['0.5 m', '1 m', '1.5 m', '2 m', '2.5 m', '3 m', '3.5 m', '4 m', '4.5 m', '> 5 m']
        elif f in ['texture', 'texture_combined']:
            self.cmap_box.current(20)
            self.class_val.set(True)
            self.class_cb.config(state='disabled')
            self.labels = ['Sand', 'Loamy Sand', 'Sandy Loam', 'Sandy Clay Loam', 'Loam', 'Sandy Clay', 'Silty Loam',
                           'Silt', 'Clay Loam', 'Silty Clay Loam', 'Clay', 'Silty Clay', 'Heavy Clay']
            
        elif f in ['limiting_factor']:
            self.cmap_box.current(25)
            self.class_val.set(True)
            self.class_cb.config(state='disabled')
            self.nodata_patch = 'last'
            self.min_val, self.max_val = 0, 3
            self.labels = ['Temperature', 'Precipitation', 'Climate Variability', 'Photoperiod', 'No Data']
            self.default_classified = True
            
    def plot_empty(self):
        self.ax.clear()
        self.fig.clear()

    def plot_on_canvas_continous(self, data, bounds, colormap, projection=None, griddist_x=30, griddist_y=30):
        if not hasattr(self, 'min_val') or not hasattr(self, 'max_val'):
            self.min_val = np.nanmin(data)
            self.max_val = np.nanmax(data)
        if not hasattr(self, 'cbar_label'):
            self.cbar_label = ''

        self.ax.clear()
        self.fig.clear()
        self.ax = self.fig.add_subplot(111, projection=projection)
        im = self.ax.imshow(data, extent=(bounds.left, bounds.right, bounds.bottom, bounds.top), origin='upper', transform=ccrs.PlateCarree(),
                            cmap=colormap, vmin=self.min_val, vmax=self.max_val)
        self.ax.coastlines() #type:ignore
        self.ax.add_feature(cfeature.OCEAN,facecolor='lightsteelblue') #type:ignore
        self.ax.add_feature(cfeature.BORDERS, edgecolor='black') #type:ignore
        self.ax.set_extent((bounds.left, bounds.right, bounds.bottom, bounds.top)) #type:ignore
        gl = self.ax.gridlines(draw_labels=True, linewidth=0.5, color='k', alpha=.33, linestyle='--', x_inline=False, y_inline=False) #type:ignore
        gl.xlocator = plt.FixedLocator(range(-180, 181, griddist_x)) #type:ignore
        gl.ylocator = plt.FixedLocator(range(-90, 91, griddist_y)) #type:ignore
        gl.xlabel_style = {'size': 8}
        gl.ylabel_style = {'size': 8}
        self.colorbar = self.fig.colorbar(im, ax=self.ax, pad=0.05, shrink=.4)
        self.colorbar.set_label(self.cbar_label, fontsize=10)
        self.colorbar.ax.tick_params(labelsize=10)
        #self.ax.axis('off')
        self.ax.set_title(f'Creation Date: {datetime.today().strftime("%Y-%m-%d")}\nFile: {self.path_entry.get()}\nCreated by CropSuite v{version}\n© Florian Zabel, Matthias Knüttel {datetime.today().strftime("%Y")}', loc='left', fontsize=6, color='black')
        self.fig.tight_layout(pad=1) #type:ignore
        self.canvas = self.fig.canvas
        self.canvas.draw()

    def classify_2d_array(self, data, num_classes=4):
        nan_mask = np.isnan(data)
        boundaries = np.linspace(self.min_val, self.max_val, num_classes)
        classified_data = np.zeros_like(data, dtype=float)
        for i, boundary in enumerate(boundaries):
            classified_data[data > boundary] = i + 1
        classified_data[nan_mask] = np.nan
        return classified_data

    def plot_on_canvas_discrete(self, data, bounds, color_list, label_list, projection=None, griddist_x=30, griddist_y=30,
                                patch_no_data='none', ncol=3):
        if not hasattr(self, 'min_val') or not hasattr(self, 'max_val'):
            self.min_val = np.nanmin(data)
            self.max_val = np.nanmax(data)

        self.bcolor = self.winfo_rgb(self.cget("bg"))
        self.bcolor = "#{:02x}{:02x}{:02x}".format(self.bcolor[0] // 256, self.bcolor[1] // 256, self.bcolor[2] // 256)

        if not self.default_classified:
            data = self.classify_2d_array(data, len(self.labels))
        cmap = clr.LinearSegmentedColormap.from_list('', color_list)
        color_list = cmap(np.linspace(0, 1, len(label_list)))
        VariableLimits = np.arange(len(label_list))
        norm = clr.BoundaryNorm(np.append(VariableLimits, 1000), ncolors=254)
        patches = [Patch(color=color, label=label) for label, color in zip(label_list, color_list)] 

        self.ax.clear()
        self.fig.clear()

        self.ax = self.fig.add_subplot(111, projection=projection)
        im = self.ax.imshow(data, extent=(bounds.left, bounds.right, bounds.bottom, bounds.top), origin='upper', transform=ccrs.PlateCarree(),
                            cmap=cmap, norm=norm, interpolation='nearest')
        self.ax.coastlines() #type:ignore
        self.ax.add_feature(cfeature.OCEAN,facecolor='lightsteelblue') #type:ignore
        self.ax.add_feature(cfeature.BORDERS, edgecolor='black') #type:ignore
        self.ax.set_extent((bounds.left, bounds.right, bounds.bottom, bounds.top)) #type:ignore
        gl = self.ax.gridlines(draw_labels=True, linewidth=0.5, color='k', alpha=.33, linestyle='--', x_inline=False, y_inline=False) #type:ignore
        gl.xlocator = plt.FixedLocator(range(-180, 181, griddist_x)) #type:ignore
        gl.ylocator = plt.FixedLocator(range(-90, 91, griddist_y)) #type:ignore
        gl.xlabel_style = {'size': 8}
        gl.ylabel_style = {'size': 8}

        for i, patch in enumerate(patches):
            if patch_no_data == 'last':
                if i == len(patches) - 1:
                    patch.set_edgecolor('black')
                else:
                    patch.set_edgecolor('none')
            elif patch_no_data == 'first':
                if i == 0:
                    patch.set_edgecolor('black')
                else:
                    patch.set_edgecolor('none')
            else:
                patch.set_edgecolor('none')

        self.ax.legend(handles=patches, loc='center', bbox_to_anchor=(0.5, -0.3), ncol=ncol, edgecolor='none', fontsize=10, frameon=False)
        #self.ax.axis('off')
        self.ax.set_title(f'Creation Date: {datetime.today().strftime("%Y-%m-%d")}\nFile: {self.path_entry.get()}\nCreated by CropSuite v{version}\n© Florian Zabel, Matthias Knüttel {datetime.today().strftime("%Y")}', loc='left', fontsize=6, color='black')
        self.fig.tight_layout(pad=1) #type:ignore
        self.canvas = self.fig.canvas
        self.canvas.draw()


    ### CLIMATE DATA OPTIONS ###

    def select_multi_file(self):
        filetypes = [('Geospatial Datasets', '*.tif;*.tiff;*.nc;*.nc4')]
        file_paths = filedialog.askopenfilenames(title="Select a file", filetypes=filetypes)
        if file_paths:
            self.climate_paths = file_paths
            self.file_listbx.delete(0, tk.END)
            for path in file_paths:
                self.file_listbx.insert(tk.END, path)
            av_years = list(vp.get_years_from_name(file_paths))
            self.start_time_cbox.config(values=av_years, state='normal')
            self.end_time_cbox.config(values=av_years, state='normal')
            self.start_time_cbox.current(0)
            self.end_time_cbox.current(len(av_years)-1)
            self.plot_climbutton.config(state='normal')
    
    def fill_climate_frame(self, parent):
        filesel_frame = ttk.LabelFrame(parent, text='File Selection')
        filesel_frame.pack(fill="x", padx=5, pady=5)

        fs_but = tk.Button(filesel_frame, text='Select Files', command=self.select_multi_file)
        fs_but.pack(fill='x')

        listbox_frame = tk.Frame(filesel_frame)
        listbox_frame.pack(fill="both", expand=True, padx=5, pady=5)
        v_scroll = tk.Scrollbar(listbox_frame, orient="vertical")
        v_scroll.pack(side="right", fill="y")
        h_scroll = tk.Scrollbar(listbox_frame, orient="horizontal")
        h_scroll.pack(side="bottom", fill="x")
        self.file_listbx = tk.Listbox(listbox_frame, yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)
        self.file_listbx.pack(fill="both", expand=True)
        v_scroll.config(command=self.file_listbx.yview)
        h_scroll.config(command=self.file_listbx.xview)

        trange_frame = ttk.LabelFrame(parent, text='Time Range')
        trange_frame.pack(fill="x", padx=5, pady=5)

        label1 = tk.Label(trange_frame, text="Start:")
        label1.pack(side='left', padx=5, pady=5)
        self.start_time_cbox = ttk.Combobox(trange_frame, values=["Option 1", "Option 2", "Option 3"], width=8, state='disabled')
        self.start_time_cbox.pack(side='left', padx=5, pady=5)
        label2 = tk.Label(trange_frame, text="End:")
        self.end_time_cbox = ttk.Combobox(trange_frame, values=["Choice A", "Choice B", "Choice C"], width=8, state='disabled')
        self.end_time_cbox.pack(side='right', pady=5)
        label2.pack(side='right', pady=5, padx=5)

        plot_frame = ttk.LabelFrame(parent, text='Plot')
        plot_frame.pack(fill="x", padx=5, pady=5)

        self.plot_climbutton = ttk.Button(plot_frame, text='Plot', command=self.clim_plot, state='disabled')
        self.plot_climbutton.pack(fill=tk.X, padx=5, pady=5)

    def clim_plot(self):
        print()

        var = 'tas'

        bounds = dt.get_extent(self.climate_paths[0])
        data = np.asarray(vp.get_netcdf_over_time_range(self.climate_paths, self.start_time_cbox.get(), self.end_time_cbox.get(), 'mean' if var == 'tas' else 'sum'))
        
        if var == 'tas':
            data -= 273.15

        colormap = vp.ColormapGetter().get_colormap(self.cmap_box.get())
        projection = vp.ProjectionGetter().get_projection(self.proj_box.get())
        
        cmap = clr.LinearSegmentedColormap.from_list('', vp.ColormapGetter().get_colormap('temperature' if var == 'tas' else 'precipitation'))
        color_list = colormap
        
        x_range = bounds.right - bounds.left
        grid_x = 5 if x_range <= 20 else 10 if x_range <= 60 else 30 if x_range <= 180 else 60
        
        y_range = bounds.top - bounds.bottom
        grid_y = 5 if y_range <= 20 else 10 if y_range <= 60 else 30

        self.plot_on_canvas_continous(data, bounds, cmap, projection, grid_x, grid_y)


if __name__ == '__main__':
    startpath = os.getcwd()
    app = ViewerGUI(startpath)
    app.mainloop()