from tkinter import Tk, Label, StringVar, DoubleVar, filedialog, Frame, Entry, Button, OptionMenu, ttk
import os
import read_climate_ini as rci


class ParamGUI:
    def __init__(self, config_file, parent_frame=None):
        super().__init__()
        self.fact = 1 if os.name == 'nt' else 1.25
        self.config_file = config_file
        self.config_dict = rci.read_ini_file(config_file)
        self.param_dict = None
        self.parent_frame = parent_frame
        if parent_frame:
            self.param_window = parent_frame
        else:
            self.param_window = Tk()
            self.param_window.title("Parameter GUI")
            self.setup_window()
        self.create_widgets()
        self.load_initial_params()
        if not parent_frame:
            self.param_window.mainloop()

    def load_initial_params(self):
        """Load initial parameters from the configuration dictionary and set up default values in the GUI."""
        vals = [val.split('.')[1] for val in list(self.config_dict.keys()) if val.startswith('parameters.')]
        first_param = vals[0]
        self.param_dict = self.config_dict.get(f'parameters.{first_param}', {})
        self.dropdown_param.set(first_param)
        #self.update_gui_with_params()
        self.update_param()

    def update_gui_with_params(self):
        """Update the GUI entries and selections based on the current parameter dictionary."""
        # Example of updating GUI entries
        self.dir_entry_val.set(self.param_dict.get('data_directory', '')) #type:ignore
        self.weight_method.set(self.get_key_by_value(self.weight_dict, int(self.param_dict.get('weighting_method', 0)))) #type:ignore
        self.convfact_val.set(float(self.param_dict.get('conversion_factor', 1.0))) #type:ignore
        self.nodata_val.set(float(self.param_dict.get('no_data', 0.0))) #type:ignore
        # Continue for other fields...

    def setup_window(self):
        x, y = 450, int(580 * self.fact)
        self.param_window.geometry(f'{x}x{y}+{(self.param_window.winfo_screenwidth() - x) // 2}+{(self.param_window.winfo_screenheight() - y) // 2}')
        self.param_window.title('CropSuite - Parameter Datasets')
        self.param_window.resizable(0, 0) #type:ignore
        self.param_window.focus_force()

    def create_widgets(self):
        font12 = 'Helvetica 12'
        if not self.parent_frame:
            Label(self.param_window, text='Parameter Datasets\n', font=f'{font12} bold').pack(anchor='w', padx=5, pady=5)
        self.create_param_selector()
        ttk.Separator(self.param_window, orient='horizontal').pack(padx=5, pady=5, fill='x')
        self.create_directory_selector()
        ttk.Separator(self.param_window, orient='horizontal').pack(padx=5, pady=5, fill='x')
        self.create_weighting_dropdown()
        ttk.Separator(self.param_window, orient='horizontal').pack(padx=5, pady=5, fill='x')
        self.create_weighting_factors()
        ttk.Separator(self.param_window, orient='horizontal').pack(padx=5, pady=5, fill='x')
        self.create_conversion_factor()
        self.create_no_data_entry()
        self.create_interpolation_dropdown()
        self.create_function_dropdown()
        ttk.Separator(self.param_window, orient='horizontal').pack(padx=5, pady=5, fill='x')
        self.create_buttons()

    def create_param_selector(self):
        self.vals = [val.split('.')[1] for val in list(self.config_dict.keys()) if val.startswith('parameters.')]
        self.dropdown_param = StringVar(self.param_window)
        self.dropdown_param.set(self.vals[0])
        self.previous_value = self.vals[0]
        self.dropdown_param.trace_add('write', self.on_change)
        self.param_dict = self.config_dict.get(f'parameters.{self.vals[0]}')
        self.param_selector = ttk.Combobox(self.param_window, textvariable=self.dropdown_param, values=self.vals, state="readonly")
        self.param_selector.pack(anchor='w', padx=5, pady=5, fill='x')

    def on_change(self, *args):
        self.update_param(self.previous_value)
        self.previous_value = self.dropdown_param.get()

    def create_directory_selector(self):
        self.dir_frame = Frame(self.param_window)
        self.dir_frame.pack(anchor='w', padx=5, pady=5, fill='x')
        Label(self.dir_frame, text='Path: ').pack(side='left', padx=5)
        self.dir_entry_val = StringVar(self.dir_frame, self.param_dict.get('data_directory')) #type:ignore
        Entry(self.dir_frame, textvariable=self.dir_entry_val).pack(side='left', padx=5, fill='x', expand=True)

        Button(self.dir_frame, text='Select', width=12, command=self.select_param_dir).pack(side='left')

    def create_weighting_dropdown(self):
        self.weight_dict = {'First Layer only': 0, 'Topsoil/First Three Layers': 1, 'Full soil/Six Layers': 2}

        weight_frame = Frame(self.param_window)
        weight_frame.pack(anchor='w', padx=5, pady=5, fill='x')
        Label(weight_frame, text='Weighting Method: ').pack(side='left')

        self.weight_method = StringVar(weight_frame, self.get_key_by_value(self.weight_dict, int(self.param_dict.get('weighting_method')))) #type:ignore
        weight_dropdown = ttk.Combobox(weight_frame, textvariable=self.weight_method, width=40)
        weight_dropdown['values'] = self.get_all_keys(self.weight_dict)
        weight_dropdown.bind('<<ComboboxSelected>>', self.update_weighting)
        weight_dropdown.pack(side='right')

    def create_weighting_factors(self):
        self.factors_value1 = DoubleVar(self.param_window, float(self.param_dict.get('weighting_factors').split(',')[0])) #type:ignore
        self.factors_value2 = DoubleVar(self.param_window, float(self.param_dict.get('weighting_factors').split(',')[1])) #type:ignore
        self.factors_value3 = DoubleVar(self.param_window, float(self.param_dict.get('weighting_factors').split(',')[2])) #type:ignore
        self.factors_value4 = DoubleVar(self.param_window, float(self.param_dict.get('weighting_factors').split(',')[3])) #type:ignore
        self.factors_value5 = DoubleVar(self.param_window, float(self.param_dict.get('weighting_factors').split(',')[4])) #type:ignore
        self.factors_value6 = DoubleVar(self.param_window, float(self.param_dict.get('weighting_factors').split(',')[5])) #type:ignore

        frame1 = Frame(self.param_window)
        frame2 = Frame(self.param_window)
        frame3 = Frame(self.param_window)
        frame4 = Frame(self.param_window)
        frame5 = Frame(self.param_window)
        frame6 = Frame(self.param_window)

        factors_label1 = Label(frame1, text='Weighting Factor   0 -  25 cm: ')
        self.factors_entry1 = Entry(frame1, textvariable=self.factors_value1)
        factors_label2 = Label(frame2, text='Weighting Factor  25 -  50 cm: ')
        self.factors_entry2 = Entry(frame2, textvariable=self.factors_value2)
        factors_label3 = Label(frame3, text='Weighting Factor  50 -  75 cm: ')
        self.factors_entry3 = Entry(frame3, textvariable=self.factors_value3)
        factors_label4 = Label(frame4, text='Weighting Factor  75 - 100 cm: ')
        self.factors_entry4 = Entry(frame4, textvariable=self.factors_value4)
        factors_label5 = Label(frame5, text='Weighting Factor 100 - 125 cm: ')
        self.factors_entry5 = Entry(frame5, textvariable=self.factors_value5)
        factors_label6 = Label(frame6, text='Weighting Factor 125 - 200 cm: ')
        self.factors_entry6 = Entry(frame6, textvariable=self.factors_value6)

        factors_label1.pack(side='left')
        self.factors_entry1.pack(side='right')
        factors_label2.pack(side='left')
        self.factors_entry2.pack(side='right')
        factors_label3.pack(side='left')
        self.factors_entry3.pack(side='right')
        factors_label4.pack(side='left')
        self.factors_entry4.pack(side='right')
        factors_label5.pack(side='left')
        self.factors_entry5.pack(side='right')
        factors_label6.pack(side='left')
        self.factors_entry6.pack(side='right')

        frame1.pack(anchor='w', padx=5, pady=5, fill='x')
        frame2.pack(anchor='w', padx=5, pady=5, fill='x')
        frame3.pack(anchor='w', padx=5, pady=5, fill='x')
        frame4.pack(anchor='w', padx=5, pady=5, fill='x')
        frame5.pack(anchor='w', padx=5, pady=5, fill='x')
        frame6.pack(anchor='w', padx=5, pady=5, fill='x')    


    def create_conversion_factor(self):
        self.convfact_val = DoubleVar(self.param_window, float(self.param_dict.get('conversion_factor'))) #type:ignore
        #self.convfact_val.trace_add('write', self.auto_save)

        frame = Frame(self.param_window)
        frame.pack(anchor='w', padx=5, pady=5, fill='x')
        Label(frame, text='Conversion Factor: ').pack(side='left')
        Entry(frame, textvariable=self.convfact_val, width=40).pack(side='right')

    def create_no_data_entry(self):
        self.nodata_val = DoubleVar(self.param_window, self.param_dict.get('no_data')) #type:ignore
        #self.nodata_val.trace_add('write', self.auto_save)

        frame = Frame(self.param_window)
        frame.pack(anchor='w', padx=5, pady=5, fill='x')
        Label(frame, text='No Data Value: ').pack(side='left')
        Entry(frame, textvariable=self.nodata_val, width=40).pack(side='right')

    def create_interpolation_dropdown(self):
        self.interp_dict = {'Linear': 0, 'Cubic': 1, 'Quadratic': 2, 'Spline': 3, 'Polygonal': 4, 'First Order Spline/sLinear': 5}
        interp_frame = Frame(self.param_window)
        interp_frame.pack(anchor='w', padx=5, pady=5, fill='x')
        Label(interp_frame, text='Interpolation Method of Membership Function: ').pack(side='left')

        self.interp_method = StringVar(interp_frame, self.get_key_by_value(self.interp_dict, int(self.param_dict.get('interpolation_method')))) #type:ignore
        interp_dropdown = ttk.Combobox(interp_frame, textvariable=self.interp_method, width=40)
        interp_dropdown['values'] = self.get_all_keys(self.interp_dict)
        interp_dropdown.pack(side='right')

    def create_function_dropdown(self):
        self.avail_functions = sorted(self.get_all_membershipfunction_names())
        func_frame = Frame(self.param_window)
        func_frame.pack(anchor='w', padx=5, pady=5, fill='x')
        Label(func_frame, text='Corresponding Membership Function: ').pack(side='left')

        self.func_val = StringVar(func_frame, str(self.param_dict.get('rel_member_func'))) #type:ignore
        func_dropdown = ttk.Combobox(func_frame, textvariable=self.func_val, width=40)
        func_dropdown['values'] = self.avail_functions
        func_dropdown.pack(side='right')

    def create_buttons(self):
        but_frame = Frame(self.param_window)
        but_frame.pack(anchor='w', padx=5, pady=5, fill='x')
        if self.parent_frame:
            Button(but_frame, text='Save Parameter Configuration', command=lambda: self.auto_save(close=False)).pack(side='right', padx=5)
            Button(but_frame, text='Add', command=self.add_param).pack(side='left', padx=5)
            Button(but_frame, text='Remove', command=self.rem_param).pack(side='left', padx=5)
        else:
            Button(but_frame, text='Save', command=lambda: self.auto_save(close=True)).pack(side='left')
            Button(but_frame, text='Add', command=self.add_param).pack(side='left', padx=5)
            Button(but_frame, text='Remove', command=self.rem_param).pack(side='left', padx=5)
            Button(but_frame, text='Exit', command=lambda: self.param_window.destroy()).pack(side='right')         

    def select_param_dir(self):
        print('Select Parameter Directory')
        param_dir = filedialog.askdirectory(initialdir=os.getcwd(), title='Select Parameter Directory')
        if param_dir:
            self.dir_entry_val.set(param_dir)
            self.param_window.update()

    def update_param(self, previous_value=None):
        if previous_value != None:
            self.auto_save(previous_value=previous_value)
        param = self.dropdown_param.get()
        self.param_dict = self.config_dict.get(f'parameters.{param}')
        self.dir_entry_val.set(self.param_dict.get('data_directory')) #type:ignore
        self.weight_method.set(self.get_key_by_value(self.weight_dict, int(self.param_dict.get('weighting_method')))) #type:ignore
        self.convfact_val.set(float(self.param_dict.get('conversion_factor', 1.0))) #type:ignore
        self.update_weighting()
        self.nodata_val.set(self.param_dict.get('no_data', -9999)) #type:ignore
        self.interp_method.set(self.get_key_by_value(self.interp_dict, int(self.param_dict.get('interpolation_method', 0)))) #type:ignore
        self.func_val.set(str(self.param_dict.get('rel_member_func'))) #type:ignore
        
    def update_weighting(self, *args):
        weight_method = self.weight_dict.get(self.weight_method.get()) #type:ignore
        try:
            factors_values = [float(val) for val in self.param_dict.get('weighting_factors').split(',')] #type:ignore
        except:
            factors_values = [1, 0, 0, 0, 0, 0]
        if weight_method == 0:
            self.factors_value1.set(1)
            self.factors_value2.set(0)
            self.factors_value3.set(0)
            self.factors_value4.set(0)
            self.factors_value5.set(0)
            self.factors_value6.set(0)

            self.factors_entry1.config(state='normal')
            self.factors_entry2.config(state='disabled')
            self.factors_entry3.config(state='disabled')
            self.factors_entry4.config(state='disabled')
            self.factors_entry5.config(state='disabled')
            self.factors_entry6.config(state='disabled')

        elif weight_method == 1:
            self.factors_value1.set(1)
            self.factors_value2.set(1)
            self.factors_value3.set(1)
            self.factors_value4.set(0)
            self.factors_value5.set(0)
            self.factors_value6.set(0)

            self.factors_entry1.config(state='normal')
            self.factors_entry2.config(state='normal')
            self.factors_entry3.config(state='normal')
            self.factors_entry4.config(state='disabled')
            self.factors_entry5.config(state='disabled')
            self.factors_entry6.config(state='disabled')

        elif weight_method == 2:
            self.factors_value1.set(factors_values[0])
            self.factors_value2.set(factors_values[1])
            self.factors_value3.set(factors_values[2])
            self.factors_value4.set(factors_values[3])
            self.factors_value5.set(factors_values[4])
            self.factors_value6.set(factors_values[5])

            self.factors_entry1.config(state='normal')
            self.factors_entry2.config(state='normal')
            self.factors_entry3.config(state='normal')
            self.factors_entry4.config(state='normal')
            self.factors_entry5.config(state='normal')
            self.factors_entry6.config(state='normal')


    def refresh_gui(self):
        self.update_param()

    def add_param(self):
        new_name = self.add_name('Add Parameter Name')
        self.config_dict[f'parameters.{new_name}'] = {
            'data_directory': '.',
            'weighting_method': '0',
            'conversion_factor': 1.0
        }
        self.auto_save()
        self.vals.append(new_name)
        self.param_selector['values'] = self.vals
        self.dropdown_param.set(new_name)

    def rem_param(self):
        param = self.dropdown_param.get()
        self.config_dict.pop(f'parameters.{param}', None)
        rci.write_config(self.config_dict, self.config_file)
        self.vals.remove(param)
        self.param_selector['values'] = self.vals
        if self.vals:
            self.dropdown_param.set(self.vals[0]) 
            self.update_param()
        else:
            self.dropdown_param.set('')

    def get_key_by_value(self, dict, value):
        return list(dict.keys())[list(dict.values()).index(value)]
    
    def add_name(self, label):
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
    
    def get_all_keys(self, dict):
        return list(dict.keys())

    def get_all_membershipfunction_names(self):
        all_pairs = []
        directory = os.path.join('plant_params', 'available')
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

    def auto_save(self, close=False, previous_value = None):
        if previous_value != None:
            param_key = f'parameters.{previous_value}'
        else:
            param_key = f'parameters.{self.dropdown_param.get()}'
        if param_key not in self.config_dict:
            return
            
        self.config_dict[param_key]['data_directory'] = self.dir_entry_val.get()
        self.config_dict[param_key]['weighting_method'] = self.weight_dict.get(self.weight_method.get(), 0)
        weighting_factors = f'{self.factors_value1.get()},{self.factors_value2.get()},{self.factors_value3.get()},{self.factors_value4.get()},{self.factors_value5.get()},{self.factors_value6.get()}'
        
        self.config_dict[param_key]['weighting_factors'] = weighting_factors
        self.config_dict[param_key]['conversion_factor'] = self.convfact_val.get()

        if self.nodata_val.get() != -9999:
            self.config_dict[param_key]['no_data'] = self.nodata_val.get()

        self.config_dict[param_key]['interpolation_method'] = self.interp_dict.get(self.interp_method.get())
        self.config_dict[param_key]['rel_member_func'] = self.func_val.get()
        rci.write_config(self.config_dict, self.config_file)

        if close:
            self.param_window.destroy()
            

if __name__ == '__main__':
    #ParamGUI(r"U:\Source Code\CropSuite\config.ini")
    pass