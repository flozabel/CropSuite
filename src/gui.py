from tkinter import *



class CropSuiteGui:
    def __init__(self):
        self.window = Tk()
        self.window.geometry("360x800")
        self.window.title("CropSuite")

        # Define colors
        self.gray = '#BFBFBF'
        self.red = '#E40808'
        self.yellow = '#E4C308'
        self.green = '#08E426'
        self.invisible = '#f0f0f0'

        self.setup_ui()

    def setup_ui(self):
        # UI setup
        Label(self.window, text='\nCropSuite\n', font='Helvetica 18 bold').pack()
        Label(self.window, text='Version 1.0\n').pack()
        Label(self.window, text='2024-11-19\n').pack()
        Label(self.window, text='Matthias Knüttel\nFlorian Zabel\n\n').pack()
        Label(self.window, text='2024\n').pack()
        Label(self.window, text='Departement of Environmental Sciences').pack()
        Label(self.window, text='University of Basel\n\n').pack()
        Label(self.window, text='© All rights reserved\n\n').pack()

        frame = Frame(self.window)
        frame.pack(anchor='w')
        cfg = Label(frame, text='Path to config.ini:')
        cfg.pack(side='left')
        self.config_ini = Entry(frame, width=40)
        self.config_ini.insert(0, r'.\config.ini')
        self.config_ini.pack(side='left')

        empty = Label(self.window, text='', fg=self.yellow, font='Helvetica 12 bold', anchor='w')
        empty.pack()

        self.libraries = Label(self.window, text='☐  Checking required Libraries', fg=self.gray, font='Helvetica 12 bold', anchor='w')
        self.libraries.pack(anchor='w')

        self.check_cfg_ini = Label(self.window, text='☐  Checking config.ini', fg=self.gray, font='Helvetica 12 bold', anchor='w')
        self.check_cfg_ini.pack(anchor='w')

        self.check_inputs = Label(self.window, text='☐  Checking all input files', fg=self.gray, font='Helvetica 12 bold', anchor='w')
        self.check_inputs.pack(anchor='w')

        self.all_checked = Label(self.window, text='\nAll requirements successfully checked!', fg=self.invisible, font='Helvetica 12 bold', anchor='w')
        self.all_checked.pack(anchor='w')

        self.downscaling = Label(self.window, text='\n☑  Downscaling Completed!', fg=self.invisible, font='Helvetica 12 bold', anchor='w')
        self.downscaling.pack(anchor='w')

        self.clim_suit = Label(self.window, text='\n☑  Climate Suitability Calculation Completed!', fg=self.invisible, font='Helvetica 12 bold', anchor='w')
        self.clim_suit.pack(anchor='w')

        self.setup_buttons()
        self.window.mainloop()

    def setup_buttons(self):
        self.but_downscaling = Button(self.window, text="Set downscaling", command=self.set_downscaling)
        self.but_downscaling.pack()

    def set_config_ini(self, ini_path):
        self.config_ini.delete(0, 'end')
        self.config_ini.insert(0, ini_path)

    def set_libraries_true(self):
        self.libraries.config(text='☑  Required Libraries checked', fg=self.green, font='Helvetica 12 bold', anchor='w')
        self.check_cfg_ini.config(text='☐  Checking config.ini', fg=self.yellow, font='Helvetica 12 bold', anchor='w')

    def set_libraries_false(self):
        self.libraries.config(text='✗  Missing Libraries', fg=self.red, font='Helvetica 12 bold', anchor='w')

    def check_cfg_ini_true(self):
        self.check_cfg_ini.config(text='☑  config.ini checked', fg=self.green, font='Helvetica 12 bold', anchor='w')
        self.check_inputs.config(text='☐  Checking all input files', fg=self.yellow, font='Helvetica 12 bold', anchor='w')

    def check_cfg_ini_false(self):
        self.check_cfg_ini.config(text='✗  config.ini faulty', fg=self.red, font='Helvetica 12 bold', anchor='w')

    def check_inpts_true(self):
        self.check_inputs.config(text='☑  All input files checked', fg=self.green, font='Helvetica 12 bold', anchor='w')
        self.all_checked.config(text='\nAll requirements successfully checked!', fg=self.green, font='Helvetica 12 bold', anchor='w')

    def check_inpts_false(self):
        self.check_inputs.config(text='✗  One or more input files faulty', fg=self.red, font='Helvetica 12 bold', anchor='w')

    def set_downscaling(self):
        completed = False
        extent = '- '+''
        no = 5
        out_of = 12

        oof = f'- {no} out of {out_of}'
        text = '\n☑  Downscaling Completed!' if completed else f'\n☐  Downscaling in Progress {extent} {oof}'
        fg = self.green if completed else self.yellow
        self.downscaling.config(text=text, fg=fg)

    
if __name__ == '__main__':
    CropSuiteGui()