#! usr/bin/env python

from tkinter import * #type:ignore
from tkinter import filedialog
from tkinter import ttk
import tkinter as tk
from tkinter import filedialog
import sys
import os
import warnings
import numpy as np
try:
    import preproc_tools as pret
    import read_climate_ini as rci
except:
    from src import preproc_tools as pret
    from src import read_climate_ini as rci
warnings.filterwarnings('ignore')


class preproc_gui():
    def __init__(self, root, config_ini):
        self.root = root

        x, y = 400 if os.name == 'nt' else 600, 850
        self.root.geometry(f'{x}x{y}+{(self.root.winfo_screenwidth() - x) // 2}+{(self.root.winfo_screenheight() - y) // 2}')
        self.root.title(f'CropSuite - Preprocessing')
        self.root.resizable(0, 0) #type:ignore
        self.root.focus_force()

        title_frm = tk.Frame(root)
        title_frm.pack(fill='x', expand=0)

        self.title = tk.Label(title_frm, text='Preprocessing', font='Helvetica 14')
        self.title.pack(side='left', padx=5, pady=5)
        ttk.Separator(root, orient='horizontal').pack(padx=5, pady=5, fill='x')

        self.sel_temp_files_but = tk.Button(root, text='Select Temperature Files', command=self.select_temp_nc_files)
        self.sel_temp_files_but.pack(pady=5, padx=5)

        self.listbox_temp = tk.Listbox(root, height=5, state='disabled')
        self.listbox_temp.pack(pady=5, fill='x')

        self.cbx_par_temp_frm = Frame(root)
        self.cbx_par_temp_frm.pack(fill='x')
        self.tvar_param_temp = StringVar(root, '')
        self.cbx_par_temp_lab = Label(self.cbx_par_temp_frm, text='Parameter: ').pack(side='left', pady=5, padx=5)
        self.combobox_param_temp = ttk.Combobox(self.cbx_par_temp_frm, textvariable=self.tvar_param_temp)
        self.combobox_param_temp.pack(side='right', pady=5, padx=5, fill='x', expand=True)
        
        ttk.Separator(root, orient='horizontal').pack(padx=5, pady=5, fill='x')

        self.sel_prec_files_but = tk.Button(root, text='Select Precipitation Files', command=self.select_prec_nc_files)
        self.sel_prec_files_but.pack(pady=5, padx=5)

        self.listbox_prec = tk.Listbox(root, height=5, state='disabled')
        self.listbox_prec.pack(pady=5, fill='x')

        self.cbx_par_prec_frm = Frame(root)
        self.cbx_par_prec_frm.pack(fill='x')
        self.tvar_param_prec = StringVar(root, '')
        self.cbx_par_prec_lab = Label(self.cbx_par_prec_frm, text='Parameter: ').pack(side='left', pady=5, padx=5)
        self.combobox_param_prec = ttk.Combobox(self.cbx_par_prec_frm, textvariable=self.tvar_param_prec)
        self.combobox_param_prec.pack(side='right', pady=5, padx=5, fill='x', expand=True)
        
        self.combobox_param_temp.bind('<<ComboboxSelected>>', self.on_combobox_select)
        self.combobox_param_prec.bind('<<ComboboxSelected>>', self.on_combobox_select)

        ttk.Separator(root, orient='horizontal').pack(padx=5, pady=5, fill='x')        

        Label(root, text='Time period').pack(padx=5, pady=5)

        self.tp_start_val = IntVar(root)
        self.tp_end_val = IntVar(root)
        self.tperiod_frm = Frame(root)
        self.tperiod_frm.pack(fill='x')
        self.tp_start_lab = Label(self.tperiod_frm, text='Start: ')
        self.cbox_start = ttk.Combobox(self.tperiod_frm, textvariable=self.tp_start_val, state='disabled')
        self.tp_end_lab = Label(self.tperiod_frm, text='End: ', width=7)
        self.cbox_end = ttk.Combobox(self.tperiod_frm, textvariable=self.tp_end_val, state='disabled')

        self.tp_start_lab.pack(side='left', padx=5, pady=5)
        self.cbox_start.pack(side='left', padx=5, pady=5)
        self.cbox_end.pack(side='right', padx=5, pady=5)
        self.tp_end_lab.pack(side='right', padx=5, pady=5)

        self.cbox_start.bind('<<ComboboxSelected>>', self.timeframe_changed)
        self.cbox_end.bind('<<ComboboxSelected>>', self.timeframe_changed)

        ttk.Separator(root, orient='horizontal').pack(padx=5, pady=5, fill='x')

        Label(root, text='Processing Extent').pack(padx=5, pady=5)

        self.extent_var = IntVar(root, 0)
        self.ext_frm = Frame(root)
        self.ext_frm.pack(fill='x')
        self.ext_rbut_1 = Radiobutton(self.ext_frm, variable=self.extent_var, value=0, text='Extent specified in config.ini', command=self.radiobutton_change)
        self.ext_rbut_2 = Radiobutton(self.ext_frm, variable=self.extent_var, value=1, text='Whole World', command=self.radiobutton_change)
        self.ext_rbut_1.pack(side='left', padx=5, pady=5)
        self.ext_rbut_2.pack(side='right', padx=5, pady=5)

        ttk.Separator(root, orient='horizontal').pack(padx=5, pady=5, fill='x')

        Label(root, text='Variability').pack(padx=5, pady=5)

        self.cb_var_val = IntVar(root, 1)
        self.cb_var = Checkbutton(root, variable=self.cb_var_val, text='Create variability files for crops')
        self.cb_var.pack(anchor='w', padx=5, pady=5)
        self.cb_var_val_ds = IntVar(root, 0)
        self.cb_var_frm = Frame(root, highlightbackground='purple1', highlightthickness=2, bd=0)
        self.cb_var_frm.pack(fill='x', padx=5)
        self.cb_var_ds = Checkbutton(self.cb_var_frm, variable=self.cb_var_val_ds,
                                     text='Use downscaled climate data for creation of climate variability files\nCaution:\nThis function is memory-intensive and may take a significant\namount of time to complete and is only recommended for\nsmall areas',
                                     fg='purple1', justify='left', command=self.downs_checked)
        self.cb_var_ds.pack(anchor='w', padx=5, pady=5)

        ttk.Separator(root, orient='horizontal').pack(padx=5, pady=5, fill='x')

        config_dict = rci.read_ini_file(config_ini)
        clim_data_path = config_dict['files'].get('climate_data_dir')

        out_frm = Frame(root)
        out_frm.pack(fill=tk.X)

        out_ent_val = StringVar(root, clim_data_path)
        out_lab = Label(out_frm, text='Output Directory: ')
        out_ent = Entry(out_frm, textvariable=out_ent_val, state='disabled')
        out_lab.pack(side='left', padx=5, pady=5)
        out_ent.pack(side='right', pady=5, padx=5, fill='x', expand=True)

        self.button_frame = tk.Frame(root)
        self.button_frame.pack(fill=tk.X)

        cancel_button = tk.Button(self.button_frame, text="Exit", command=self.root.destroy)
        cancel_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.save_button = tk.Button(self.button_frame, text="Start", command=
                                     lambda: self.start_preproc(config_ini, self.listbox_temp.get(0,tk.END), self.listbox_prec.get(0,tk.END),
                                                                (int(self.cbox_start.get()), int(self.cbox_end.get())), self.extent_var.get(),
                                                                self.cb_var_val.get(), self.combobox_param_temp.get(), self.combobox_param_prec.get(),
                                                                self.cb_var_val_ds.get()), state='disabled',
                                                                bg='red3', fg='white')
        self.save_button.pack(side=tk.RIGHT, padx=5, pady=5)

    def start_preproc(self, config_ini, temp_files, prec_files, time_range, extent, proc_varfiles, varname_temp, varname_prec, downscaling):
        pret.preprocessing_main(config_ini, temp_files, prec_files, time_range, extent, proc_varfiles, varname_temp, varname_prec, downscaling==1)
        self.root.destroy

    def downs_checked(self):
        if self.cb_var_val_ds.get() == 0:
            self.ext_rbut_1.config(state='normal')
            self.ext_rbut_2.config(state='normal')
        else:
            self.extent_var.set(0)
            self.ext_rbut_1.config(state='disabled')
            self.ext_rbut_2.config(state='disabled')

    def radiobutton_change(self):
        if self.extent_var.get() == 0:
            self.cb_var_ds.config(state='normal')
            self.cb_var_frm.config(highlightbackground='purple1', highlightthickness=2, bd=0)
        else:
            self.cb_var_val_ds.set(0)
            self.cb_var_ds.config(state='disabled')
            self.cb_var_frm.config(highlightbackground=self.root.cget('bg'), highlightthickness=2, bd=0)

    def timeframe_changed(self, event):
        start_val = self.cbox_start.get()
        _, end_year = pret.get_time_range(self.listbox_temp.get(0,tk.END))
        avail_years = list(np.arange(int(start_val), end_year))
        self.cbox_end['values'] = avail_years        

    def on_combobox_select(self, event):
        sel_temp_var = self.tvar_param_temp.get()
        sel_prec_var = self.tvar_param_prec.get()
        if (sel_temp_var != '' and sel_prec_var != '') and (sel_temp_var != None and sel_prec_var != None):
            self.fill_values()
            self.save_button.config(state='normal', bg='medium sea green', fg='black')

    def fill_values(self):
        self.cbox_start.config(state='normal')
        self.cbox_end.config(state='normal')

        start_year, end_year = pret.get_time_range(self.listbox_temp.get(0,tk.END))

        avail_years = list(np.arange(start_year, end_year))
        self.cbox_start['values'] = avail_years
        self.cbox_end['values'] = avail_years
        self.tp_start_val.set(avail_years[0])
        self.tp_end_val.set(avail_years[-1])

    def select_temp_nc_files(self):
        self.listbox_temp.config(state='normal')
        file_paths = filedialog.askopenfilenames(filetypes=[("NetCDF Files", "*.nc")])
        self.listbox_temp.delete(0, tk.END)
        for file in sorted(file_paths):
            self.listbox_temp.insert(tk.END, file)
        vals = pret.get_available_parameters(self.listbox_temp.get(0,tk.END))
        self.combobox_param_temp['values'] = vals
        if len(vals) == 1:
            self.tvar_param_temp.set(vals[0])
        self.on_combobox_select(self)
        self.listbox_temp.config(state='disabled')
        
    def select_prec_nc_files(self):
        self.listbox_prec.config(state='normal')
        file_paths = filedialog.askopenfilenames(filetypes=[("NetCDF Files", "*.nc")])
        self.listbox_prec.delete(0, tk.END)
        for file in sorted(file_paths):
            self.listbox_prec.insert(tk.END, file)
        self.combobox_param_prec['values'] = pret.get_available_parameters(self.listbox_prec.get(0,tk.END))
        vals = pret.get_available_parameters(self.listbox_prec.get(0,tk.END))
        self.combobox_param_prec['values'] = vals
        if len(vals) == 1:
            self.tvar_param_prec.set(vals[0]) 
        self.on_combobox_select(self)
        self.listbox_prec.config(state='disabled')

def parse_arguments():
    """Parse command-line arguments using sys.argv"""
    if not '-nogui' in sys.argv:
        return None, None, None, None, 0, 0, [0, 0, 0, 0], False
    if len(sys.argv) < 8:
        print("Usage: preproc_gui.py -nogui -continue -config.ini -[temperature_files] -[precipitation_files] -[startyear, endyear] -[y_max, x_min, y_min, x_max]")
        sys.exit(1)
    
    nogui = '-nogui' in sys.argv
    continue_cropsuite = '-continue' in sys.argv
    config_file = sys.argv[sys.argv.index('-config.ini') + 1]
    
    temp_start = sys.argv.index('-temperature_files') + 1
    temp_end = sys.argv.index('-precipitation_files')
    temperature_files = sys.argv[temp_start:temp_end]
    
    precip_start = sys.argv.index('-precipitation_files') + 1
    precip_end = sys.argv.index('-years')
    precipitation_files = sys.argv[precip_start:precip_end]
    
    years_start = sys.argv.index('-years') + 1
    startyear = int(sys.argv[years_start])
    endyear = int(sys.argv[years_start + 1])
    
    bbox_start = sys.argv.index('-bbox') + 1
    y_max, x_min, y_min, x_max = map(float, sys.argv[bbox_start:bbox_start + 4])
    
    return nogui, config_file, temperature_files, precipitation_files, startyear, endyear, (y_max, x_min, y_min, x_max), continue_cropsuite

def startup():
        print('''\
        
        =======================================================
        |                                                     |
        |                                                     |    
        |                      CropSuite                      |
        |                                                     |
        |                     Version 0.95                    |
        |                      2024-10-13                     |
        |                                                     |
        |     - Create Files for Climate and Variability -    |
        |                                                     |
        |                      2024-10-13                     |
        |                                                     |
        |                                                     |
        |        Based on the work of Zabel et al. 2014       |
        |                                                     |
        |                                                     |
        |                     Florian Zabel                   |
        |                   Matthias Knüttel                  |
        |                         2024                        |
        |                                                     |  
        |                  University of Basel                |
        |                                                     |
        |                                                     |
        |                © All rights reserved                |
        |                                                     |
        =======================================================
        
        ''')

if __name__ == '__main__':
    startup()
    nogui, config_file, temperature_files, precipitation_files, startyear, endyear, (y_max, x_min, y_min, x_max), continue_cropsuite = parse_arguments()
    if nogui:
        pret.preprocessing_main(config_ini=config_file,
                                temp_files=temperature_files,
                                prec_files=precipitation_files,
                                time_range=(startyear, endyear),
                                extent=0,
                                proc_varfiles=True,
                                varname_temp='tas',
                                varname_pr='pr',
                                downscaling=True)
        if continue_cropsuite:
            import subprocess
            try:
                command = ['python', 'CropSuite.py', '-silent', '-config', f'{config_file}']
                subprocess.run(command)
            except:
                command = ['python3', 'CropSuite.py', '-silent', '-config', f'{config_file}']
                subprocess.run(command)

    else:
        if os.path.exists('config.ini'):
            config_ini_path = 'config.ini'
        else:
            print('Select config.ini file')
            config_ini_path = filedialog.askopenfilename(title="Select a config.ini file", filetypes=[("ini files", "*.ini")], initialdir=os.getcwd())
        root = Tk()
        root.withdraw()
        preproc_window = preproc_gui(tk.Toplevel(root), config_ini_path)
        root.wait_window(preproc_window.root)