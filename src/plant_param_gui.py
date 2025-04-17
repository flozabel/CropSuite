from tkinter import * #type:ignore
from tkinter import ttk
import tkinter as tk
import os
import warnings
import numpy as np
try:
    import read_climate_ini as rci
    import read_plant_params as rpp
except:
    from src import read_climate_ini as rci
    from src import read_plant_params as rpp
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from scipy.integrate import cumulative_trapezoid

warnings.filterwarnings('ignore')

class plant_param_gui(tk.Frame):
    def __init__(self, root, config_ini, plant):
        ### MAIN ###

        self.fact = 1 if os.name == 'nt' else 1.25

        self.config_file = rci.read_ini_file(config_ini)

        if plant == None:
            self.crop_dict, self.plant = self.get_empty_crop_dict()
            crop = self.crop_dict.get('name')
        else:
            crop = os.path.splitext(plant)[0]
            self.plant = plant
            self.crop_dict = rpp.read_single_crop_parameterizations_files(os.path.join(self.config_file['files'].get('plant_param_dir'), 'available', self.plant))[crop.lower()]
        if crop == None:
            crop = ''
        if self.crop_dict == None:
            self.crop_dict = {}

        self.param_keys = [key[:-5] for key in self.crop_dict if key.endswith('_vals')]
        self.label_dict = {'temp': 'Temperature', 'prec': 'Precipitation', 'slope': 'Slope', 'soildepth': 'Soil Depth', 'base_sat': 'Base Saturation', 'texture': 'Texture Class',
                    'coarsefragments': 'Coarse Fragments', 'gypsum': 'Gypsum Content', 'ph': 'Soil pH', 'organic_carbon': 'Soil Organic Carbon Content', 'elco': 'Electric Conductivity/Salinity', 
                    'esp': 'Exchangable Sodium Percentage/Sodicity', 'freqcropfail': 'Recurrence Rate of Potential Crop Failures'}
        self.params = [self.label_dict.get(param, param) for param in self.param_keys]

        self.root = root
        x, y = int(550 * self.fact), int(800 * self.fact)
        self.root.geometry(f'{x}x{y}+{(self.root.winfo_screenwidth() - x) // 2}+{(self.root.winfo_screenheight() - y) // 2}')
        self.root.title(f'CropSuite - {crop.capitalize()} Parameterization')
        self.root.resizable(1, 1) #type:ignore
        self.root.focus_force()

        ### TITLE ###

        title_frm = tk.Frame(root)
        title_frm.pack(fill='x', expand=0)

        title = tk.Label(title_frm, text=f'Parameterization {crop.capitalize()}', font='Helvetica 14')
        title.pack(side='left', padx=5, pady=5)
        ttk.Separator(root, orient='horizontal').pack(padx=5, pady=5, fill='x')
        
        ### GROWING CYCLE ###

        growing_cycle_frm = Frame(root)
        growing_cycle_frm.pack(fill='x', pady=5)
        growing_cycle_label = Label(growing_cycle_frm, text='Length of growing cycle:')
        self.growing_cycle_var = IntVar(root, int(self.crop_dict.get('growing_cycle', 0)))
        self.growing_cycle_perennial_var = IntVar(root, int(self.crop_dict.get('growing_cycle', 0)) == 365)
        self.growing_cycle_entry = Entry(growing_cycle_frm, width=7, textvariable=self.growing_cycle_var,
                                    state='disabled' if int(self.crop_dict.get('growing_cycle', 0)) == 365 else 'normal')
        self.growing_cycle_entry.bind("<FocusOut>", lambda event: self.check_lgc_value())
        growing_cycle_unit_label = Label(growing_cycle_frm, text='days')
        growing_cycle_perennial = Checkbutton(growing_cycle_frm, variable=self.growing_cycle_perennial_var, text='Crop is a perennial crop', command=self.crop_perennial)

        growing_cycle_label.pack(side='left', padx=5, pady=5)
        self.growing_cycle_entry.pack(side='left', padx=5, pady=5)
        growing_cycle_unit_label.pack(side='left', padx=5, pady=5)
        growing_cycle_perennial.pack(side='right', padx=5, pady=5)

        ttk.Separator(root, orient='horizontal').pack(padx=5, pady=5, fill='x')   
    
        ### MEMBERSHIP VALUES ###

        memship_frm = Frame(root)
        memship_frm.pack(fill='x', pady=5)
        membership_label = Label(memship_frm, text='Current membership function:')
        self.memship_var = StringVar(root)
        self.memship_cbox = ttk.Combobox(memship_frm, width=40, state='normal', textvariable=self.memship_var)
        self.add_memship_but = Button(memship_frm, width=10, text='Add', command=self.add_memship)
        membership_label.pack(side='left', pady=5, padx=5)
        self.memship_cbox.pack(side='left', pady=5, padx=5)
        self.add_memship_but.pack(side='right', pady=5, padx=5)
        self.memship_cbox['values'] = [self.label_dict.get(k, k.capitalize()) for k in self.param_keys]
        self.memship_var.set([self.label_dict.get(k, k.capitalize()) for k in self.param_keys][0]) #type:ignore
        self.memship_cbox.bind('<<ComboboxSelected>>', self.membership_changed)

        xvals_frm = Frame(root)
        xvals_frm.pack(fill='x', pady=1)
        xvals_lab = Label(xvals_frm, text='x-Axis values:')
        self.xvals_var = StringVar(root, ', '.join(str(e) for e in self.crop_dict.get(self.param_keys[0]+'_vals', 'temp_vals')))
        self.xvals_ent = Entry(xvals_frm, textvariable=self.xvals_var, width=30)
        xvals_lab.pack(side='left', pady=1, padx=5)
        self.xvals_ent.pack(side='left', pady=1, padx=5, fill='x', expand=True)

        yvals_frm = Frame(root)
        yvals_frm.pack(fill='x', pady=1)
        yvals_lab = Label(yvals_frm, text='Suitability:      ')
        self.yvals_var = StringVar(root, ', '.join(str(e) for e in self.crop_dict.get(self.param_keys[0]+'_suit', 'temp_suit')))
        self.yvals_ent = Entry(yvals_frm, textvariable=self.yvals_var, width=30)
        yvals_lab.pack(side='left', pady=1, padx=5)
        self.yvals_ent.pack(side='left', pady=1, padx=5, fill='x', expand=True)
        self.yvals_ent.bind('<FocusOut>', self.check_val_lengths)
        self.xvals_ent.bind('<FocusOut>', self.check_val_lengths)

        canvas_frm = Frame(root)
        canvas_frm.pack(pady=5)

        left_frame = Frame(canvas_frm)
        left_frame.pack(side='left', padx=5)

        Label(left_frame, text='Lower\nSuitability\nThreshold:').pack(pady=5)

        self.spinbox_low_var = DoubleVar(root, 0.05)
        self.spinbox_lower = Spinbox(left_frame, from_=0, to=0.3, width=7, textvariable=self.spinbox_low_var, command=self.plot_graph, increment=.01, justify='center')
        self.spinbox_lower.pack(pady=5)

        Label(left_frame, text='Value:').pack(pady=5)

        self.varval_lower = DoubleVar(root, 0)
        self.varval_left = Entry(left_frame, width=5, textvariable=self.varval_lower, state='disabled', justify='center')
        self.varval_left.pack(pady=5)

        #self.fig, self.ax = plt.subplots(figsize=(4, 3))
        #self.fig.set_facecolor('#f0f0f0')
        self.fig = Figure(figsize=(4, 3), facecolor='#f0f0f0')
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_frm)
        self.canvas.get_tk_widget().pack(side='left', padx=5, pady=5)   

        right_frame = Frame(canvas_frm)
        right_frame.pack(side='left', padx=5)

        Label(right_frame, text='Upper\nSuitability\nThreshold:').pack(pady=5)

        self.spinbox_up_var = DoubleVar(root, 0.05)
        self.spinbox_upper = Spinbox(right_frame, from_=0, to=0.3, width=7, textvariable=self.spinbox_up_var, command=self.plot_graph, increment=.01, justify='center')
        self.spinbox_upper.pack(pady=5)

        Label(right_frame, text='Value:').pack(pady=5)

        self.varval_upper = IntVar(root, 0)
        self.varval_right = Entry(right_frame, width=5, textvariable=self.varval_upper, state='disabled', justify='center')
        self.varval_right.pack(pady=5)       

        if next((k for k, v in self.label_dict.items() if v == self.memship_var.get()), '') in ['temp', 'prec']:
            self.spinbox_lower.config(state='normal')
            self.spinbox_upper.config(state='normal')
        else:
            self.spinbox_lower.config(state='disabled')
            self.spinbox_upper.config(state='disabled')   
        self.check_val_lengths(None)

        ttk.Separator(root, orient='horizontal').pack(padx=5, pady=5, fill='x')

        ### CROP REQUIREMENTS ### 

        Label(root, text='Crop-specific growth requirements:').pack(anchor='w', padx=5, pady=5)

        self.germ_frm = Frame(root)
        self.germ_frm.pack(fill='x', padx=5)

        self.germ_cb_val = IntVar(root, int(self.crop_dict.get('prec_req_after_sow', 0 if self.growing_cycle_var.get() >= 365 else 1) in [1, 'y']))
        self.germ_cb = Checkbutton(self.germ_frm, variable=self.germ_cb_val, text='Germination Requirements', command=self.germination_cb_switched)
        self.germ_but = Button(self.germ_frm, text='Set', width=10, state='normal' if self.germ_cb_val.get() == 1 else 'disabled', command=self.set_germination_requirements)
        self.germ_cb.pack(side='left', padx=5, pady=5)
        self.germ_but.pack(side='right', padx=5, pady=5)
        self.germination_cb_switched()

        self.lethals_frm = Frame(root)
        self.lethals_frm.pack(fill='x', padx=5)

        self.lethals_cb_val = IntVar(root, int(self.crop_dict.get('lethal_thresholds', 0) in [1, 'y']))
        self.lethals_cb = Checkbutton(self.lethals_frm, variable=self.lethals_cb_val, text='Consider Lethal Thresholds', command=self.lethal_cb_switched)
        self.lethals_but = Button(self.lethals_frm, text='Set', width=10, state='normal' if self.lethals_cb_val.get() == 1 else 'disabled', command=self.set_lethal_thresholds)
        self.lethals_cb.pack(side='left', padx=5, pady=5)
        self.lethals_but.pack(side='right', padx=5, pady=5)
        self.lethal_cb_switched()

        self.photo_frm = Frame(root)
        self.photo_frm.pack(fill='x', padx=5)

        self.photo_cb_val = IntVar(root, int(self.crop_dict.get('photoperiod', 'n') in [1, 'y']))
        self.photo_cb = Checkbutton(self.photo_frm, variable=self.photo_cb_val, text='Consider Photoperiod', command=self.photo_cb_switched)
        self.photo_but = Button(self.photo_frm, text='Set', width=10, state='normal' if self.photo_cb_val.get() == 1 else 'disabled', command=self.set_photoperiod_settings)
        self.photo_cb.pack(side='left', padx=5, pady=5)
        self.photo_but.pack(side='right', padx=5, pady=5)
        self.photo_cb_switched()

        self.winter_frm = Frame(root)
        self.winter_frm.pack(fill='x', padx=5)

        self.winter_cb_val = IntVar(root, 1 if self.crop_dict.get('wintercrop') in [1, 'y'] else 0)
        self.winter_cb = Checkbutton(self.winter_frm, variable=self.winter_cb_val, text='Crop is a Wintercrop', command=self.winter_cb_switched)
        self.winter_but = Button(self.winter_frm, text='Set', width=10, state='normal' if self.winter_cb_val.get() == 1 else 'disabled', command=self.set_vernalization_params)
        self.winter_cb.pack(side='left', padx=5, pady=5)
        self.winter_but.pack(side='right', padx=5, pady=5)
        self.winter_cb_switched()

        add_cond_frm = Frame(root)
        add_cond_frm.pack(fill='x', padx=5)

        self.additionals_but = Button(add_cond_frm, text='Configure additional requirements', width=30, state='normal',
                                      command=self.add_conditions_window)
        self.additionals_but.pack(side='right', padx=5, pady=5)
        ttk.Separator(root, orient='horizontal').pack(padx=5, pady=5, fill='x')

        ### BOTTOM FRAME ###
        self.botton_frame = tk.Frame(root)
        self.botton_frame.pack(fill=tk.X, side='bottom')

        cancel_button = tk.Button(self.botton_frame, text="Exit", command=self.root.destroy)
        cancel_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.save_button = tk.Button(self.botton_frame, text="Save", command=lambda: [self.save(), self.root.destroy()])
        self.save_button.pack(side=tk.RIGHT, padx=5, pady=5)

    def set_germination_requirements(self):
        self.germreq_root = Tk()

        crop = os.path.splitext(self.plant)[0]

        x, y = int(500 * self.fact), int(150 * self.fact)
        self.germreq_root.geometry(f'{x}x{y}+{(self.germreq_root.winfo_screenwidth() - x) // 2}+{(self.germreq_root.winfo_screenheight() - y) // 2}')
        self.germreq_root.title(f'Germination Requirements - {crop}')
        self.germreq_root.resizable(0, 0) #type:ignore
        self.germreq_root.focus_force()

        title_frm = tk.Frame(self.germreq_root)
        title_frm.pack(fill='x', expand=0)
        title = tk.Label(title_frm, text=f'Germination Requirements {crop.capitalize()}', font='Helvetica 14')
        title.pack(side='left', padx=5, pady=5)
        ttk.Separator(self.germreq_root, orient='horizontal').pack(padx=5, pady=5, fill='x')

        frm1 = Frame(self.germreq_root)
        frm1.pack(fill='x', expand=0)
        Label(frm1, text='At least').pack(side='left', padx=5, pady=5)
        self.ent_sow_dur_var = IntVar(self.germreq_root, self.crop_dict.get('temp_for_sow_duration', 7))
        sbx_sow_dur = Spinbox(frm1, from_=1, to=int(self.crop_dict.get('growing_cycle', 1)), width=7, textvariable=self.ent_sow_dur_var, justify='center')
        sbx_sow_dur.pack(side='left', padx=5, pady=5)
        Label(frm1, text='days above').pack(side='left', padx=5, pady=5)
        self.ent_sow_temp_var = IntVar(self.germreq_root, self.crop_dict.get('temp_for_sow', 5))
        ent_sow_temp = Entry(frm1, textvariable=self.ent_sow_temp_var, width=7, justify='center')
        ent_sow_temp.pack(side='left', padx=5, pady=5)
        Label(frm1, text='°C').pack(side='left', padx=5, pady=5)

        frm2 = Frame(self.germreq_root)
        frm2.pack(fill='x', expand=0)
        Label(frm2, text='More than').pack(side='left', padx=5, pady=5)
        self.ent_req_prec_var = IntVar(self.germreq_root, self.crop_dict.get('prec_req_after_sow', 20))
        ent_req_prec = Entry(frm2, textvariable=self.ent_req_prec_var, width=7, justify='center')
        ent_req_prec.pack(side='left', padx=5, pady=5)
        Label(frm2, text='mm required within the first').pack(side='left', padx=5, pady=5)
        self.ent_req_prec_dur_var = IntVar(self.germreq_root, self.crop_dict.get('prec_req_days', 14))
        sbx_req_prec = Spinbox(frm2, from_=1, to=int(self.crop_dict.get('growing_cycle', 1)), width=7, textvariable=self.ent_req_prec_dur_var, justify='center')
        sbx_req_prec.pack(side='left', padx=5, pady=5)
        Label(frm2, text='days after sowing').pack(side='left', padx=5, pady=5)        

        botton_frame = tk.Frame(self.germreq_root)
        botton_frame.pack(fill=tk.X, side='bottom')
        cancel_button = tk.Button(botton_frame, text="Exit", command=self.germreq_root.destroy)
        cancel_button.pack(side=tk.LEFT, padx=5, pady=5)
        save_button = tk.Button(botton_frame, text="Save", command=self.germination_requirements_to_dict)
        save_button.pack(side=tk.RIGHT, padx=5, pady=5)  

    def germination_requirements_to_dict(self):
        temp_for_sow_duration = self.ent_sow_dur_var.get()
        temp_for_sow = self.ent_sow_temp_var.get()
        prec_req_days = self.ent_req_prec_dur_var.get()
        prec_req_after_sow = self.ent_req_prec_var.get()
        add_dict = {'prec_req_after_sow': prec_req_after_sow,
                    'prec_req_days': prec_req_days,
                    'temp_for_sow_duration': temp_for_sow_duration,
                    'temp_for_sow': temp_for_sow}  
        for key, value in add_dict.items():
            self.crop_dict[key] = value
        self.germreq_root.destroy()

    def set_lethal_thresholds(self):
        self.letthres_root = Tk()

        crop = os.path.splitext(self.plant)[0]

        x, y = int(500 * self.fact), int(250 * self.fact)
        self.letthres_root.geometry(f'{x}x{y}+{(self.letthres_root.winfo_screenwidth() - x) // 2}+{(self.letthres_root.winfo_screenheight() - y) // 2}')
        self.letthres_root.title(f'Lethal Thresholds - {crop}')
        self.letthres_root.resizable(0, 0) #type:ignore
        self.letthres_root.focus_force()

        title_frm = tk.Frame(self.letthres_root)
        title_frm.pack(fill='x', expand=0)
        title = tk.Label(title_frm, text=f'Lethal Thresholds {crop.capitalize()}', font='Helvetica 14')
        title.pack(side='left', padx=5, pady=5)
        ttk.Separator(self.letthres_root, orient='horizontal').pack(padx=5, pady=5, fill='x')

        frm1 = Frame(self.letthres_root)
        frm1.pack(fill='x', expand=0)
        Label(frm1, text='Not more than').pack(side='left', padx=5, pady=5)
        self.ent_consec_days_below_dur = IntVar(self.letthres_root, self.crop_dict.get('lethal_min_temp_duration', 7))
        sbx_temp_below_dur = Spinbox(frm1, from_=1, to=int(self.crop_dict.get('growing_cycle', 1)), width=7, textvariable=self.ent_consec_days_below_dur, justify='center')
        sbx_temp_below_dur.pack(side='left', padx=5, pady=5)
        Label(frm1, text='consecutive days below').pack(side='left', padx=5, pady=5)
        self.ent_consec_days_below_tmp = IntVar(self.letthres_root, self.crop_dict.get('lethal_min_temp', 5))
        ent_temp_below_tmp = Entry(frm1, textvariable=self.ent_consec_days_below_tmp, width=7, justify='center')
        ent_temp_below_tmp.pack(side='left', padx=5, pady=5)
        Label(frm1, text='°C').pack(side='left', padx=5, pady=5)

        frm2 = Frame(self.letthres_root)
        frm2.pack(fill='x', expand=0)
        Label(frm2, text='Not more than').pack(side='left', padx=5, pady=5)
        self.ent_consec_days_above_dur = IntVar(self.letthres_root, self.crop_dict.get('lethal_max_temp_duration', 5))
        sbx_temp_above_dur = Spinbox(frm2, from_=1, to=int(self.crop_dict.get('growing_cycle', 1)), width=7, textvariable=self.ent_consec_days_above_dur, justify='center')
        sbx_temp_above_dur.pack(side='left', padx=5, pady=5)
        Label(frm2, text='consecutive days above').pack(side='left', padx=5, pady=5)
        self.ent_consec_days_above_tmp = IntVar(self.letthres_root, self.crop_dict.get('lethal_max_temp', 32))
        ent_temp_above_tmp = Entry(frm2, textvariable=self.ent_consec_days_above_tmp, width=7, justify='center')
        ent_temp_above_tmp.pack(side='left', padx=5, pady=5)
        Label(frm2, text='°C').pack(side='left', padx=5, pady=5)      

        frm3 = Frame(self.letthres_root)
        frm3.pack(fill='x', expand=0)
        Label(frm3, text='Not more than').pack(side='left', padx=5, pady=5)
        self.ent_consec_dry_days_dur = IntVar(self.letthres_root, self.crop_dict.get('lethal_min_prec_duration', 21))
        sbx_temp_above_dur = Spinbox(frm3, from_=1, to=int(self.crop_dict.get('growing_cycle', 1)), width=7, textvariable=self.ent_consec_dry_days_dur, justify='center')
        sbx_temp_above_dur.pack(side='left', padx=5, pady=5)
        Label(frm3, text='consecutive days below').pack(side='left', padx=5, pady=5)
        self.ent_consec_dry_days_value = IntVar(self.letthres_root, self.crop_dict.get('lethal_min_prec', 1))
        ent_prec_below = Entry(frm3, textvariable=self.ent_consec_dry_days_value, width=7, justify='center')
        ent_prec_below.pack(side='left', padx=5, pady=5)
        Label(frm3, text='mm/day').pack(side='left', padx=5, pady=5)

        frm4 = Frame(self.letthres_root)
        frm4.pack(fill='x', expand=0)
        Label(frm4, text='Not more than').pack(side='left', padx=5, pady=5)
        self.max_prec_val = IntVar(self.letthres_root, self.crop_dict.get('lethal_max_prec', 100))
        self.max_prec_dur = IntVar(self.letthres_root, self.crop_dict.get('lethal_max_prec_duration', 3))
        ent_prev_dur = Spinbox(frm4, from_=1, to=int(self.crop_dict.get('growing_cycle', 1)), textvariable=self.max_prec_dur, width=7, justify='center')
        ent_prev_dur.pack(side='left', padx=5, pady=5)
        Label(frm4, text='consecutive days above').pack(side='left', padx=5, pady=5)
        sbx_prec_above_val = Spinbox(frm4, from_=1, to=2000, width=7, textvariable=self.max_prec_val, justify='center')
        sbx_prec_above_val.pack(side='left', padx=5, pady=5)
        Label(frm4, text='mm/day').pack(side='left', padx=5, pady=5)   

        botton_frame = tk.Frame(self.letthres_root)
        botton_frame.pack(fill=tk.X, side='bottom')
        cancel_button = tk.Button(botton_frame, text="Exit", command=self.letthres_root.destroy)
        cancel_button.pack(side=tk.LEFT, padx=5, pady=5)
        save_button = tk.Button(botton_frame, text="Save", command=self.lethal_thresholds_to_dict)
        save_button.pack(side=tk.RIGHT, padx=5, pady=5)      

    def lethal_thresholds_to_dict(self):
        lethal_min_duration = self.ent_consec_days_below_dur.get()
        lethal_min_temp = self.ent_consec_days_below_tmp.get()
        lethal_max_duration = self.ent_consec_days_above_dur.get()
        lethal_max_temp = self.ent_consec_days_above_tmp.get()        
        consecutive_dry_days = self.ent_consec_dry_days_dur.get()
        dry_day_prec = self.ent_consec_dry_days_value.get()
        max_prec_val = self.max_prec_val.get()
        max_prec_dur = self.max_prec_dur.get()

        add_dict = {'lethal_min_temp_duration': lethal_min_duration, 'lethal_min_temp': lethal_min_temp,
                    'lethal_max_temp_duration': lethal_max_duration, 'lethal_max_temp': lethal_max_temp,
                    'lethal_min_prec_duration': consecutive_dry_days, 'lethal_min_prec': dry_day_prec, 'lethal_max_prec': max_prec_val,
                    'lethal_max_prec_duration': max_prec_dur}  
        
        self.crop_dict['lethal_thresholds'] = self.lethals_cb_val.get()
        for key, value in add_dict.items():
            self.crop_dict[key] = value

        if self.lethals_cb_val.get() == 0:
            for key in add_dict.keys():
                self.crop_dict.pop(key, None)

        self.letthres_root.destroy()

    def set_photoperiod_settings(self):
        self.photset_root = Tk()

        crop = os.path.splitext(self.plant)[0]

        x, y = int(400 * self.fact), int(180 * self.fact)
        self.photset_root.geometry(f'{x}x{y}+{(self.photset_root.winfo_screenwidth() - x) // 2}+{(self.photset_root.winfo_screenheight() - y) // 2}')
        self.photset_root.title(f'Photoperiod Settings - {crop}')
        self.photset_root.resizable(0, 0) #type:ignore
        self.photset_root.focus_force()

        title_frm = tk.Frame(self.photset_root)
        title_frm.pack(fill='x', expand=0)
        title = tk.Label(title_frm, text=f'Photoperiod Settings {crop.capitalize()}', font='Helvetica 14')
        title.pack(side='left', padx=5, pady=5)
        ttk.Separator(self.photset_root, orient='horizontal').pack(padx=5, pady=5, fill='x')

        frm1 = Frame(self.photset_root)
        frm1.pack(fill='x', expand=0)
        Label(frm1, text='Day length longer than').pack(side='left', padx=5, pady=5)

        self.ent_day_lon = IntVar(self.photset_root, self.crop_dict.get('minimum_sunlight_hours', 8))
        sbx_day_lon = Spinbox(frm1, from_=0, to=24, width=7, textvariable=self.ent_day_lon, justify='center', command=self.daylength_changed1)
        sbx_day_lon.pack(side='left', padx=5, pady=5)
        Label(frm1, text='hours').pack(side='left', padx=5, pady=5)

        frm2 = Frame(self.photset_root)
        frm2.pack(fill='x', expand=0)
        Label(frm2, text='Day length shorter than').pack(side='left', padx=5, pady=5)

        self.ent_day_sho = IntVar(self.photset_root, self.crop_dict.get('maximum_sunlight_hours', 16))
        sbx_day_sho = Spinbox(frm2, from_=0, to=24, width=7, textvariable=self.ent_day_sho, justify='center', command=self.daylength_changed2)
        sbx_day_sho.pack(side='left', padx=5, pady=5)
        Label(frm2, text='hours').pack(side='left', padx=5, pady=5)

        botton_frame = tk.Frame(self.photset_root)
        botton_frame.pack(fill=tk.X, side='bottom')
        cancel_button = tk.Button(botton_frame, text="Exit", command=self.photset_root.destroy)
        cancel_button.pack(side=tk.LEFT, padx=5, pady=5)
        save_button = tk.Button(botton_frame, text="Save", command=self.photo_settings_to_dict)
        save_button.pack(side=tk.RIGHT, padx=5, pady=5)       

    def daylength_changed1(self):
        self.ent_day_sho.set(24 - self.ent_day_lon.get())

    def daylength_changed2(self):
        self.ent_day_lon.set(24 - self.ent_day_sho.get())        

    def photo_settings_to_dict(self):
        minimum_sunlight_hours = self.ent_day_sho.get()
        maximum_sunlight_hours = self.ent_day_lon.get()

        add_dict = {'photoperiod': 1, 'minimum_sunlight_hours': minimum_sunlight_hours, 'maximum_sunlight_hours': maximum_sunlight_hours}  
        for key, value in add_dict.items():
            self.crop_dict[key] = value
        self.photset_root.destroy()

    def set_vernalization_params(self):
        self.vernparam_root = Tk()

        crop = os.path.splitext(self.plant)[0]

        x, y = int(600 * self.fact), int(200 * self.fact)
        self.vernparam_root.geometry(f'{x}x{y}+{(self.vernparam_root.winfo_screenwidth() - x) // 2}+{(self.vernparam_root.winfo_screenheight() - y) // 2}')
        self.vernparam_root.title(f'Vernalization Parameters - {crop}')
        self.vernparam_root.resizable(0, 0) #type:ignore
        self.vernparam_root.focus_force()

        title_frm = tk.Frame(self.vernparam_root)
        title_frm.pack(fill='x', expand=0)
        title = tk.Label(title_frm, text=f'Vernalization Parameters {crop.capitalize()}', font='Helvetica 14')
        title.pack(side='left', padx=5, pady=5)
        ttk.Separator(self.vernparam_root, orient='horizontal').pack(padx=5, pady=5, fill='x')

        frm1 = Frame(self.vernparam_root)
        frm1.pack(fill='x', expand=0)
        self.vern_effect_days = IntVar(self.vernparam_root, self.crop_dict.get('vernalization_effective_days', 50))
        sbx_vern_effect_days = Spinbox(frm1, from_=1, to=120, width=7, textvariable=self.vern_effect_days, justify='center')
        sbx_vern_effect_days.pack(side='left', padx=5, pady=5)
        Label(frm1, text='vernalization effective days with temperatures between').pack(side='left', padx=5, pady=5)

        self.vern_min_temp = IntVar(self.vernparam_root, self.crop_dict.get('vernalization_tmin', 0))
        self.vern_max_temp = IntVar(self.vernparam_root, self.crop_dict.get('vernalization_tmax', 8))

        ent_vern_min_temp = Entry(frm1, textvariable=self.vern_min_temp, width=7, justify='center')
        ent_vern_min_temp.pack(side='left', padx=5, pady=5)
        Label(frm1, text='°C and').pack(side='left', padx=5, pady=5)

        ent_vern_max_temp = Entry(frm1, textvariable=self.vern_max_temp, width=7, justify='center')
        ent_vern_max_temp.pack(side='left', padx=5, pady=5)
        Label(frm1, text='°C').pack(side='left', padx=5, pady=5)

        frm2 = Frame(self.vernparam_root)
        frm2.pack(fill='x', expand=0)

        self.max_days_below = IntVar(self.vernparam_root, self.crop_dict.get('frost_resistance_days', 3))
        self.max_days_below_temp = IntVar(self.vernparam_root, self.crop_dict.get('frost_resistance', -20))
        Label(frm2, text='Maximal').pack(side='left', padx=5, pady=5)
        sbx_frost_dur = Spinbox(frm2, from_=1, to=20, width=7, textvariable=self.max_days_below, justify='center')
        sbx_frost_dur.pack(side='left', padx=5, pady=5)
        Label(frm2, text='days below').pack(side='left', padx=5, pady=5)
        sbx_frost_dur_temp = Spinbox(frm2, from_=-40, to=0, width=7, textvariable=self.max_days_below_temp, justify='center')
        sbx_frost_dur_temp.pack(side='left', padx=5, pady=5)
        Label(frm2, text='°C').pack(side='left', padx=5, pady=5)

        frm3 = Frame(self.vernparam_root)
        frm3.pack(fill='x', expand=0)

        self.day_to_vern = IntVar(self.vernparam_root, self.crop_dict.get('days_to_vernalization', 80))
        Label(frm3, text='Days from sowing to begin of vernalization period: ').pack(side='left', padx=5, pady=5)
        sbx_frost_dur = Spinbox(frm3, from_=1, to=int(self.crop_dict.get('growing_cycle', 1))-1, width=7, textvariable=self.day_to_vern, justify='center')
        sbx_frost_dur.pack(side='left', padx=5, pady=5)
        Label(frm3, text='days').pack(side='left', padx=5, pady=5)

        botton_frame = tk.Frame(self.vernparam_root)
        botton_frame.pack(fill=tk.X, side='bottom')
        cancel_button = tk.Button(botton_frame, text="Exit", command=self.vernparam_root.destroy)
        cancel_button.pack(side=tk.LEFT, padx=5, pady=5)
        save_button = tk.Button(botton_frame, text="Save", command=self.vern_params_to_dict)
        save_button.pack(side=tk.RIGHT, padx=5, pady=5)    

    def vern_params_to_dict(self):
        vernalization_effective_days = self.vern_effect_days.get()
        vernalization_tmin = self.vern_min_temp.get()
        vernalization_tmax = self.vern_max_temp.get()
        frost_resistance_days = self.max_days_below.get()
        frost_resistance = self.max_days_below_temp.get()
        days_to_vernalization = self.day_to_vern.get()

        add_dict = {'wintercrop': 'y', 'vernalization_effective_days': vernalization_effective_days,
                    'vernalization_tmin': vernalization_tmin, 'vernalization_tmax': vernalization_tmax,
                    'frost_resistance_days': frost_resistance_days, 'frost_resistance': frost_resistance,
                    'days_to_vernalization': days_to_vernalization}  
        for key, value in add_dict.items():
            self.crop_dict[key] = value
        self.vernparam_root.destroy()

    def add_conditions_window(self):
        self.addcon_root = Tk()

        crop = str(self.plant).split('.')[0].capitalize()

        x, y = int(550 * self.fact), int(400 * self.fact)
        self.addcon_root.geometry(f'{x}x{y}+{(self.addcon_root.winfo_screenwidth() - x) // 2}+{(self.addcon_root.winfo_screenheight() - y) // 2}')
        self.addcon_root.title(f'Additional Conditions - {crop}')
        self.addcon_root.resizable(0, 0) #type:ignore
        self.addcon_root.focus_force()

        self.cons_preproc_var = IntVar(self.addcon_root, int(self.crop_dict.get('consider_in_preproc', 0) in [1, 'y']))
        cons_preproc_cb = Checkbutton(self.addcon_root, text='Consider in Preprocessing', variable=self.cons_preproc_var)
        cons_preproc_cb.pack(anchor='w', padx=5, pady=5)

        main_frame = tk.Frame(self.addcon_root)
        main_frame.pack(fill=tk.BOTH, expand=1, padx=5, pady=5)

        self.canvas = tk.Canvas(main_frame)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.canvas.bind('<Configure>', lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))) #type:ignore

        self.table_frame = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.table_frame, anchor="nw")
        headers = ["#", "Parameter", "Start Day", "End Day", "Condition", "Value"]

        widths = [3, 12, 5, 5, 5, 8]

        for idx, text in enumerate(headers):
            label = tk.Label(self.table_frame, text=text, padx=10, pady=5, width=widths[idx])
            label.grid(row=0, column=idx)

        self.row_data = []
        self.row_count = 0

        self.button_frame = tk.Frame(self.addcon_root)
        self.button_frame.pack(fill=tk.X)

        cancel_button = tk.Button(self.button_frame, text="Cancel", command=self.addcon_root.destroy)
        cancel_button.pack(side=tk.LEFT, padx=5, pady=5)

        add_button = tk.Button(self.button_frame, text="Add", command=lambda: self.add_row(self.crop_dict))
        add_button.pack(side=tk.LEFT, padx=5, pady=5)

        remove_button = tk.Button(self.button_frame, text="Remove", command=self.remove_row)
        remove_button.pack(side=tk.LEFT, padx=5, pady=5)

        save_button = tk.Button(self.button_frame, text="Save", command=self.save_data)
        save_button.pack(side=tk.RIGHT, padx=5, pady=5)

        for i in range(100):
            if f'AddCon:{i}' in self.crop_dict.keys():
                first = 0 if self.crop_dict.get(f'AddCon:{i}')[0] == 'Temperature' else 1 #type:ignore
                second = int(self.crop_dict.get(f'AddCon:{i}')[1])-1 #type:ignore
                third = int(self.crop_dict.get(f'AddCon:{i}')[2])-1 #type:ignore
                cond = self.crop_dict.get(f'AddCon:{i}')[3] #type:ignore
                fourth = 0 if cond == '>' else 1 if cond == '>=' else 2 if cond == '<=' else 3 
                fifth = float(self.crop_dict.get(f'AddCon:{i}')[4]) #type:ignore
                self.add_row(self.crop_dict, values=[first, second, third, fourth, fifth])

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
        self.add_data_to_crop_dict(data)
        self.addcon_root.destroy()
    
    def add_data_to_crop_dict(self, data):
        for i in range(100):
            try:
                self.crop_dict.pop(f'AddCon:{i}', None)
            except:
                pass
        self.crop_dict.update({f'AddCon:{item[0]}': item[1:] for item in data})
    
    def cancel(self):
        self.addcon_root.destroy()

    def germination_cb_switched(self):
        if self.germ_cb_val.get() == 1:
            self.germ_cb.config(bg='SeaGreen1')
            self.germ_frm.config(bg='SeaGreen1')
            self.germ_but.config(state='normal')
        else:
            self.germ_cb.config(bg=self.root.cget("bg"))
            self.germ_frm.config(bg=self.root.cget("bg"))
            self.germ_but.config(state='disabled')

    def lethal_cb_switched(self):
        if self.lethals_cb_val.get() == 1:
            self.lethals_cb.config(bg='SeaGreen1')
            self.lethals_frm.config(bg='SeaGreen1')
            self.lethals_but.config(state='normal')
        else:
            self.lethals_cb.config(bg=self.root.cget("bg"))
            self.lethals_frm.config(bg=self.root.cget("bg"))
            self.lethals_but.config(state='disabled')

    def photo_cb_switched(self):
        if self.photo_cb_val.get() == 1:
            self.photo_cb.config(bg='SeaGreen1')
            self.photo_frm.config(bg='SeaGreen1')
            self.photo_but.config(state='normal')
        else:
            self.photo_frm.config(bg=self.root.cget("bg"))
            self.photo_cb.config(bg=self.root.cget("bg"))
            self.photo_but.config(state='disabled')

    def winter_cb_switched(self):
        if self.winter_cb_val.get() == 1:
            self.winter_cb.config(bg='SeaGreen1')
            self.winter_frm.config(bg='SeaGreen1')
            self.winter_but.config(state='normal')
        else:
            self.winter_cb.config(bg=self.root.cget("bg"))
            self.winter_frm.config(bg=self.root.cget("bg"))
            self.winter_but.config(state='disabled')
    
    def find_percentiles(self, x, y, p):
        cdf = cumulative_trapezoid(y / np.trapz(y, x), x, initial=0)
        return np.interp(p, cdf, x)

    def find_x_for_y_left(self, x, y, target_y):
        """
        Finds the x-axis values corresponding to a target y-value from the left and right.
        """
        if y[0] >= target_y:
            return x[0]
        x, y = np.array(x), np.array(y)
        left_idx = np.where(y >= target_y)[0][0]
        x_left = np.interp(target_y, [y[left_idx - 1], y[left_idx]], [x[left_idx - 1], x[left_idx]])

        return x_left
    
    def find_right_x_for_y(self, x, y, target_y):
        if y[-1] >= target_y:
            return x[-1]
        x, y = np.array(x), np.array(y)
        for i in range(len(y) - 1, 0, -1):
            if y[i] < target_y <= y[i - 1]:
                return np.interp(target_y, [y[i], y[i - 1]], [x[i], x[i - 1]])

    def plot_graph(self):
        try:
            curr_key = next((k for k, v in self.label_dict.items() if v == self.memship_var.get()), '')
            vals, suit = [float(v) for v in self.crop_dict.get(curr_key+'_vals', 'temp_vals')], [float(s) for s in self.crop_dict.get(curr_key+'_suit', 'temp_suit')]
            if curr_key in ['temp', 'prec']:
                try:
                    lower_perc, upper_perc = self.spinbox_low_var.get(), self.spinbox_up_var.get()
                except:
                    lower_perc, upper_perc = 5, 5
                if curr_key == 'temp':
                    #lower_lim = self.find_percentiles(vals, suit, lower_perc)
                    #upper_lim = self.find_percentiles(vals, suit, upper_perc)

                    #lower_lim = np.percentile(vals, lower_perc)
                    #upper_lim = np.percentile(vals, upper_perc)

                    lower_lim = self.find_x_for_y_left(vals, suit, lower_perc)
                    upper_lim = self.find_right_x_for_y(vals, suit, upper_perc)
                    
                    add_dict = {'lowtemp_lim': lower_lim, 'hightemp_lim': upper_lim}
                    for key, value in add_dict.items():
                        self.crop_dict[key] = value
                    
                elif curr_key == 'prec':
                    #lower_lim = self.find_percentiles(vals, suit, lower_perc)
                    #upper_lim = self.find_percentiles(vals, suit, upper_perc)
                    
                    #lower_lim = int(np.percentile(vals, lower_perc))
                    #upper_lim = int(np.percentile(vals, upper_perc))

                    lower_lim = self.find_x_for_y_left(vals, suit, lower_perc)
                    upper_lim = self.find_right_x_for_y(vals, suit, upper_perc)

                    add_dict = {'lowprec_lim': lower_lim, 'highprec_lim': upper_lim}
                    for key, value in add_dict.items():
                        self.crop_dict[key] = value
                    
                self.varval_left.config(state='normal')
                self.varval_right.config(state='normal')
                self.varval_lower.set(round(float(lower_lim), 2) if curr_key == 'temp' else int(lower_lim))
                self.varval_upper.set(round(float(upper_lim), 2) if curr_key == 'temp' else int(upper_lim)) #type:ignore
                self.varval_left.config(state='disabled')
                self.varval_right.config(state='disabled')
            self.ax.clear()
            self.ax.plot(vals, suit, 'black', marker='o', linestyle='-')
            if curr_key in ['temp', 'prec']:
                self.ax.axvline(x=lower_lim, color='blue', linestyle='--', label='Lower Limit for Variability') #type:ignore
                self.ax.axvline(x=upper_lim, color='red', linestyle='--', label='Upper Limit for Variability') #type:ignore
            self.ax.set_xlabel(self.memship_var.get())
            self.ax.set_ylabel('Suitability')
            self.ax.set_title(f'Membership Function for {self.memship_var.get()}')
            self.ax.axhspan(-0.1, 0, facecolor='lightgray', alpha=0.5)
            self.ax.axhspan(1.0, 1.1, facecolor='lightgray', alpha=0.5)
            self.ax.set_ylim(-0.1, 1.1)
            self.ax.grid()
            self.fig.tight_layout() #type:ignore
            #plt.tight_layout()
            self.canvas.draw() #type:ignore
        except Exception as e:
            print(e)

    def save(self):
        winter_keys = ['wintercrop', 'vernalization_effective_days', 'vernalization_tmax', 'vernalization_tmin',
                       'frost_resistance_days', 'frost_resistance', 'days_to_vernalization']
        germ_keys = ['prec_req_after_sow', 'prec_req_days', 'temp_for_sow_duration', 'temp_for_sow']
        lethal_keys = ['lethal_thresholds', 'lethal_min_temp_duration', 'lethal_min_temp',
                    'lethal_max_temp_duration', 'lethal_max_temp', 'lethal_min_prec_duration', 'lethal_min_prec',
                    'lethal_max_prec', 'lethal_max_prec_duration']  
        photo_keys = ['photoperiod', 'minimum_sunlight_hours', 'maximum_sunlight_hours']

        rem_keys = []
        if self.germ_cb_val.get() == 0:
            rem_keys += germ_keys
        if self.winter_cb_val.get() == 0:
            rem_keys += winter_keys
        if self.lethals_cb_val.get() == 0:
            rem_keys += lethal_keys
        if self.photo_cb_val.get() == 0:
            rem_keys += photo_keys

        for key in rem_keys:
            self.crop_dict.pop(key, None)

        self.crop_dict['growing_cycle'] = self.growing_cycle_var.get()

        with open(os.path.join(self.config_file['files'].get('plant_param_dir'), 'available', self.plant), 'w') as write_file:
            write_file.write(f'name = \t\t{os.path.splitext(self.plant)[0].lower()}\n')
            for key, value in self.crop_dict.items():
                if key == 'name':
                    continue
                if isinstance(value, list):
                    value_str = ','.join(map(str, value))
                elif isinstance(value, bool):
                    value_str = 'y' if value else 'n'
                else:
                    value_str = str(value)
                write_file.write(f'{key} = \t\t{value_str}\n')
            if hasattr(self, 'cons_preproc_var') and self.cons_preproc_var.get() == 1:
                write_file.write(f'consider_in_preproc = \t\t{self.cons_preproc_var.get()}\n')
        # self.root.destroy()

    def add_memship(self):
        new_memship = str(add_name('Abbreviation of new\nMembership Function')).lower()
        self.params.append(new_memship)
        self.memship_cbox['values'] = [self.label_dict.get(k, k.capitalize()) for k in self.param_keys] + [new_memship]
        self.crop_dict[f'{new_memship}_vals'] = [0, 1]
        self.crop_dict[f'{new_memship}_suit'] = [1, 1]
        self.xvals_var.set('0, 1')
        self.yvals_var.set('1, 1')
        self.memship_var.set(new_memship)
        self.plot_graph()

    def membership_changed(self, event):
        new_key = next((k for k, v in self.label_dict.items() if v == self.memship_var.get()), '')
        x_vals = self.crop_dict.get(new_key+'_vals')
        y_vals = self.crop_dict.get(new_key+'_suit')
        self.xvals_var.set(', '.join(str(e) for e in x_vals)) #type:ignore
        self.yvals_var.set(', '.join(str(e) for e in y_vals)) #type:ignore
        self.check_val_lengths(None)
        self.plot_graph()

        if new_key in ['temp', 'prec']:
            self.spinbox_lower.config(state='normal')
            self.spinbox_upper.config(state='normal')
        else:
            self.spinbox_lower.config(state='disabled')
            self.spinbox_upper.config(state='disabled')            

    def check_val_lengths(self, event):
        x_vals = self.xvals_var.get().replace(', ', ',').split(',')
        y_vals = self.yvals_var.get().replace(', ', ',').split(',')

        y_vals = [1 if float(v) > 1 else 0 if float(v) < 0 else float(v) for v in y_vals]

        if len(x_vals) == len(y_vals):
            self.xvals_ent.config(highlightthickness=2, highlightbackground='green')
            self.yvals_ent.config(highlightthickness=2, highlightbackground='green')
            self.plot_graph()
        else:
            self.xvals_ent.config(highlightthickness=2, highlightbackground='red')
            self.yvals_ent.config(highlightthickness=2, highlightbackground='red')

        self.xvals_var.set(', '.join(str(e) for e in x_vals))
        self.yvals_var.set(', '.join(str(e) for e in y_vals))

        ky = next((k for k, v in self.label_dict.items() if v == self.memship_var.get()), '')
        self.crop_dict[f'{ky}_vals'] = self.xvals_var.get().replace(', ', ',').split(',')
        self.crop_dict[f'{ky}_suit'] = self.yvals_var.get().replace(', ', ',').split(',')
        self.plot_graph()

    def crop_perennial(self):
        if self.growing_cycle_perennial_var.get() == 1:
            self.growing_cycle_var.set(365)
            self.growing_cycle_entry.config(state='disabled')
        else:
            self.growing_cycle_entry.config(state='normal')
            self.growing_cycle_var.set(int(self.crop_dict.get('growing_cycle', 0))) 

    def check_lgc_value(self):
        try:
            growing_cycle_value = self.growing_cycle_var.get()
        except Exception as inst:
            d = inst
        if growing_cycle_value >= 365:
            self.growing_cycle_entry.config(state='disabled')
            self.growing_cycle_var.set(365)
            self.growing_cycle_perennial_var.set(1)
        elif growing_cycle_value < 1:
            self.growing_cycle_entry.config(state='normal')
            self.growing_cycle_var.set(1)
            self.growing_cycle_perennial_var.set(0)

    def get_empty_crop_dict(self):
        def_plant = os.listdir(os.path.join(self.config_file['files'].get('plant_param_dir'), 'available'))
        def_plant = [plant for plant in def_plant if plant.endswith('.inf')][1]
        def_plant = parse_file_to_dict(os.path.join(self.config_file['files'].get('plant_param_dir'), 'available', def_plant))

        for key in def_plant.keys():
            if key.endswith('_vals') and key not in ['temp_vals', 'prec_vals', 'freqcropfail_vals']:
                def_plant[key] = [0, 1]
                def_plant[key.replace('_vals', '_suit')] = [1, 1]
        name, copy = self.new_crop_name()
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
            def_plant = parse_file_to_dict(os.path.join(self.config_file['files'].get('plant_param_dir'), 'available', f'{copy}.inf'))
            def_plant['name'] = name
        new_file = f'{name}.inf'
        return def_plant, new_file
    
    def new_crop_name(self):
        name_win = Tk()
        name_win.title('New Crop')
        name_win.resizable(0, 0) #type:ignore
        name_win.focus_force()
        x, y = int(200 * self.fact), int(100 * self.fact)
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

def get_available_crop_list():
    try:
        verified = r'U:\web\verified'
        not_verified = r'U:\web\not_verified'
        os.makedirs(verified, exist_ok=True)
        os.makedirs(not_verified, exist_ok=True)

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

def open_gui(config_ini_path, plant_inf):
    root = Tk()
    root.withdraw()
    plant_window = plant_param_gui(tk.Toplevel(root), config_ini_path, plant=plant_inf)
    root.wait_window(plant_window.root)

if __name__ == '__main__':
    pass
    # config_ini_path = r"U:\Source Code\CropSuite\world_config.ini"
    # root = Tk()
    # root.withdraw()
    # plant_window = plant_param_gui(tk.Toplevel(root), config_ini_path, plant='Maize_plutor.inf')
    # root.wait_window(plant_window.root)