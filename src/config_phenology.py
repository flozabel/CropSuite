import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import os
import numpy as np
import re
from PIL import Image, ImageTk
try:
    from src import membership_editor as me
except:
    import membership_editor as me

class PhenologyWindow:
    def __init__(self, parent_self):
        super().__init__()
        self.parent = parent_self
        self.crop_dict = parent_self.crop_dict
        self.growing_cycle = int(float(self.crop_dict.get('growing_cycle')))
        self.rows = []
        self.row_index = 0
        self.build_window()

    def build_window(self):
        fact = 1 if os.name == 'nt' else 1.25
        self.window = tk.Toplevel()
        self.window.title("Phenology Window")
        self.window.resizable(False, False)

        width, height = int(480 * fact), int(600 * fact)
        screen_w = self.window.winfo_screenwidth()
        screen_h = self.window.winfo_screenheight()
        x = (screen_w - width) // 2
        y = (screen_h - height) // 2
        self.window.geometry(f"{width}x{height}+{x}+{y}")
        self.window.title(f'CropSuite - {self.crop_dict.get("name", "").capitalize()} Parameterization')
        self.window.focus_force()

        self.notebook = ttk.Notebook(self.window)
        self.notebook.pack(fill="both", expand=True)

        self.temp_tab = ttk.Frame(self.notebook)
        self.prec_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.temp_tab, text="Temperature")
        self.notebook.add(self.prec_tab, text="Precipitation")

        self.table_frames = {}
        self.row_lists = {}
        self.row_indices = {}

        for mode, tab in [("temp", self.temp_tab), ("prec", self.prec_tab)]:
            self.row_lists[mode] = []
            self.row_indices[mode] = 0
            self.add_plot(tab, mode=mode)
            self.add_table(tab, mode=mode)
            self.add_controls(tab, mode=mode)


    def add_plot(self, parent, mode):
        if hasattr(self, 'canvases') and mode in self.canvases:
            old_canvas = self.canvases[mode]
            old_canvas.get_tk_widget().pack_forget()
            old_canvas.get_tk_widget().destroy()
            del self.canvases[mode]
            del self.figures[mode]
            if hasattr(self, 'axes') and mode in self.axes:
                del self.axes[mode]
        fig = Figure(figsize=(4, 2), dpi=100)
        bg_col = tk.Tk()
        bg_col.withdraw()
        self.bcolor = bg_col.winfo_rgb(bg_col.cget("bg"))
        self.bcolor = "#{:02x}{:02x}{:02x}".format(self.bcolor[0] // 256, self.bcolor[1] // 256, self.bcolor[2] // 256)
        bg_col.destroy()
        fig.patch.set_facecolor(self.bcolor)
        
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="x", padx=10, pady=(10, 0), ipady=5)
        if not hasattr(self, 'figures'):
            self.figures = {}
        if not hasattr(self, 'canvases'):
            self.canvases = {}
        if not hasattr(self, 'axes'):
            self.axes = {}

        self.figures[mode] = fig
        self.canvases[mode] = canvas
        self.axes[mode] = fig.add_subplot(111)  

    def smooth_curve(self, x, lower, upper, smoothness):
        x = np.asarray(x)
        half = smoothness / 2
        k = 2 * np.log(99) / smoothness
        rise = np.zeros_like(x, dtype=float)
        rise[x <= lower - half] = 0
        rise[x >= lower + half] = 100
        in_rise = (x > lower - half) & (x < lower + half)
        rise[in_rise] = 100 / (1 + np.exp(-k * (x[in_rise] - lower)))
        fall = np.zeros_like(x, dtype=float)
        fall[x <= upper - half] = 100
        fall[x >= upper + half] = 0
        in_fall = (x > upper - half) & (x < upper + half)
        fall[in_fall] = 100 / (1 + np.exp(k * (x[in_fall] - upper)))
        return np.minimum(rise, fall)
    
    def smooth_curve_rounded(self, x, lower, upper, smoothness):
        x = np.asarray(x)
        half = smoothness / 2
        k = 2 * np.log(99) / smoothness
        rise = np.zeros_like(x, dtype=float)
        rise[x <= lower - half] = 0
        rise[x >= lower + half] = 100
        in_rise = (x > lower - half) & (x < lower + half)
        rise[in_rise] = 100 / (1 + np.exp(-k * (x[in_rise] - lower)))
        fall = np.zeros_like(x, dtype=float)
        fall[x <= upper - half] = 100
        fall[x >= upper + half] = 0
        in_fall = (x > upper - half) & (x < upper + half)
        fall[in_fall] = 100 / (1 + np.exp(k * (x[in_fall] - upper)))
        y = np.minimum(rise, fall)
        result = list(zip(x.astype(int), np.rint(y).astype(int)))
        if any(x < 0 for x, _ in result):
            result = [(xi, yi) for xi, yi in result if xi >= 0]
            result.insert(0, (0, 0))
        return result

    def create_raster(self, mode='temp'):
        growing_cycle = int(self.crop_dict.get('growing_cycle', 366))
        data = []

        _min, _max = 1000, 0
        for i in range(len(self.row_lists[mode])):
            _min = float(self.row_lists[mode][i][3].get()) if float(self.row_lists[mode][i][3].get()) < _min else _min
            _max = float(self.row_lists[mode][i][4].get()) if float(self.row_lists[mode][i][4].get()) > _max else _max
        _min, _max = np.min([_min, -5 if mode == 'temp' else 0]), np.max([_max, 45 if mode == 'temp' else 500])

        x_rng = np.arange(_min, _max, .1)
        y_rng = np.arange(1, growing_cycle+1, 1)
        for row in self.row_lists[mode]:
            if len(row) == 6:
                data.append([row[1].get(), row[2].get(), row[3].get(), row[4].get(), row[5].get()])
            else:
                data.append([row[1].get(), row[2].get(), row[3].get(), row[4].get(), row[5].get(), row[6]])
        raster = np.zeros((len(y_rng), len(x_rng)))
        for lst in data:
            y_start, y_end = int(float(lst[0])), int(float(lst[1]))
            if len(lst) == 6 and lst[5]:
                points = sorted(lst[5])
                x_vals, y_vals = np.array(points).T
                def interp(x):
                    return np.interp(x, x_vals, y_vals)
                for y in range(y_start-1, y_end):
                    raster[y] = interp(x_rng)
            else:
                x_start, x_end = int(float(lst[2])), int(float(lst[3]))
                smoothness = float(lst[4])
                for y in range(y_start-1, y_end):
                    raster[y] = self.smooth_curve(x_rng, x_start, x_end, smoothness=smoothness)
        return raster
    
    def get_optimum(self, mode='temp'):
        temp_vals, temp_suit = self.crop_dict.get(f'{mode}_vals'), self.crop_dict.get(f'{mode}_suit')
        temp_vals_f, temp_suit_f = list(map(float, temp_vals)), list(map(float, temp_suit))
        vals_with_suit_1 = [val for val, suit in zip(temp_vals_f, temp_suit_f) if suit == 1.0]
        if vals_with_suit_1:
            return sum(vals_with_suit_1) / len(vals_with_suit_1)
        else:
            return None

    def plot(self, data, mode='temp'):
        fig = self.figures[mode]
        canvas = self.canvases[mode]
        ax = self.axes[mode]
        ax.clear()
        h, _ = data.shape

        _min, _max = 1000, 0
        for i in range(len(self.row_lists[mode])):
            _min = float(self.row_lists[mode][i][3].get()) if float(self.row_lists[mode][i][3].get()) < _min else _min
            _max = float(self.row_lists[mode][i][4].get()) if float(self.row_lists[mode][i][4].get()) > _max else _max
        opt_val = self.get_optimum(mode=mode)
        if mode == 'prec':
            sm = 0
            for i in range(len(self.row_lists[mode])):
                if len(self.row_lists[mode][i]) > 6 and self.row_lists[mode][i][6] != []:
                    points = self.row_lists[mode][i][6]
                    sm += sum(x for x, y in points if y == max(y for _, y in points)) / sum(1 for x, y in points if y == max(y for _, y in points))
                else:
                    minv, maxv = self.row_lists[mode][i][3].get(), self.row_lists[mode][i][4].get()
                    sm += (int(float(minv)) + int(float(maxv))) / 2
        else:
            mn = 0
            for i in range(len(self.row_lists[mode])):
                if len(self.row_lists[mode][i]) > 6 and self.row_lists[mode][i][6] != []:
                    dur = float(self.row_lists[mode][i][2].get()) - float(self.row_lists[mode][i][1].get()) 
                    points = self.row_lists[mode][i][6]
                    mn += int(dur * sum(x for x, y in self.row_lists[mode][i][6] if y == max(y for _, y in self.row_lists[mode][i][6])) / len([1 for _, y in self.row_lists[mode][i][6] if y == max(y for _, y in self.row_lists[mode][i][6])]))
                else:
                    dur = float(self.row_lists[mode][i][2].get()) - float(self.row_lists[mode][i][1].get()) 
                    minv, maxv = float(self.row_lists[mode][i][3].get()), float(self.row_lists[mode][i][4].get())
                    mn += ((maxv + minv) / 2) * dur
            mn /= self.growing_cycle

        _min, _max = np.min([_min, -5 if mode == 'temp' else 0]), np.max([int(_max*1.3), int(sm*1.3), int(opt_val*1.3), 500]) if mode == 'prec' else np.max([int(_max*1.3), 45, int(mn*1.3)])#type:ignore
        extent = [_min, _max, 0, h]
        im = ax.imshow(data, cmap='RdYlBu', vmin=0, vmax=100, extent=extent, aspect='auto', alpha=.7)

        if mode == 'temp':
            ax.set_xticks(np.arange(_min, _max+1, 5))
            ax.axvline(0, color='darkblue', linestyle=':')
            ax.axvline(5, color='blue', linestyle=':')
            ax.axvline(40, color='darkred', linestyle=':')
            opt = self.get_optimum(mode=mode)
            color = 'green' if opt is not None and abs(opt - mn) <= 1 else 'black'
            ax.axvline(mn, color=color)
            if opt is not None:
                ax.axvline(opt, color=color, linestyle='--')
        else:
            ax.set_xticks(np.arange(_min, _max+1, 50))
            opt = self.get_optimum(mode=mode)
            color = 'green' if opt is not None and abs(opt - sm) <= 25 else 'black'
            if opt is not None:
                ax.axvline(opt, color=color, linestyle='--')
            ax.axvline(sm, color=color)

            if opt is not None:
                missing = round(opt - sm)
                if missing > 25:
                    ax.text(sm + 5, 0.5 * ax.get_ylim()[1], f"{missing} mm missing", fontsize=8, color='black', va='center', ha='left', rotation=90)
                if missing < -25:
                    ax.text(sm - 20, 0.5 * ax.get_ylim()[1], f"{abs(missing)} mm too much", fontsize=8, color='black', va='center', ha='left', rotation=90)

        yticks = np.linspace(0, self.growing_cycle, 15)
        ax.set_yticks(yticks)
        ax.invert_yaxis()
        ax.set_yticklabels([str(int(y)) for y in yticks[::-1]])
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.set_xlabel("Temperature [°C]" if mode == 'temp' else "Precipitation [mm]", fontsize=8)
        ax.set_ylabel("Day of Growing Cycle", fontsize=8)
        ax.set_ylim(0, h)
        fig.subplots_adjust(left=.18, right=.98, top=.9, bottom=.18)
        canvas.draw()

    def add_table(self, parent, mode):
        table_frame = tk.Frame(parent)
        table_frame.pack(fill="both", expand=True, padx=10, pady=(10, 0))
        self.table_frames[mode] = table_frame

        headers = ["#", "Start Day", "End Day", "Min", "Max", "Smoothness"]
        widths = [3, 7, 7, 7, 7, 7]

        for idx, text in enumerate(headers):
            label = tk.Label(table_frame, text=text, padx=10, pady=5, width=widths[idx])
            label.grid(row=0, column=idx, sticky="nsew")
        self.process_crop_dict(self.crop_dict, mode)

    def add_controls(self, parent, mode):
        btn_frame = tk.Frame(parent)
        btn_frame.pack(pady=10)
        cnc_btn = tk.Button(btn_frame, text="Cancel", width=10, command=self.window.destroy).pack(side="left", padx=5)
        add_btn = tk.Button(btn_frame, text="Add", width=10, command=lambda: self.add_row(mode=mode)).pack(side="left", padx=5)
        remove_btn = tk.Button(btn_frame, text="Remove", width=10, command=lambda: self.remove_row(mode=mode)).pack(side="left", padx=5)
        ok_btn = tk.Button(btn_frame, text='Ok', width=10, command=self.ok).pack(side="left", padx=5)
    

    def _bind_spinbox(self, widget, mode, row):
        widget.bind("<FocusOut>", lambda event: self.spbx_changed(mode, row))
        widget.bind("<Return>", lambda event: self.spbx_changed(mode, row))
        widget.bind("<KeyRelease-Up>", lambda event: self.spbx_changed(mode, row))
        widget.bind("<KeyRelease-Down>", lambda event: self.spbx_changed(mode, row))
        widget.bind("<ButtonRelease-1>", lambda event: self.spbx_changed(mode, row))

    def process_crop_dict(self, crop_dict, mode):
        for key, value in crop_dict.items():
            match = re.match(r"phen\d+_(\d+)-(\d+)_(\w+)", key)
            if not match:
                continue
            from_day = int(match.group(1))
            to_day = int(match.group(2))
            curr_mode = match.group(3)
            if curr_mode != mode:
                continue
            try:
                if isinstance(value, str):
                    parts = list(map(float, value.split(",")))
                else:
                    parts = (value[:3] + [[] if value[3] == '[]' else [(int(value[i].strip(' []()')), int(value[i+1].strip(' []()'))) for i in range(3, len(value), 2)]] if len(value) > 3 else [[]])
                if len(parts) == 3:
                    min_val, max_val, smooth_val = parts
                elif len(parts) == 4:
                    min_val, max_val, smooth_val, points = parts
                else:
                    continue
            except ValueError:
                continue
            self.add_row(param="Phase", from_day=from_day, to_day=to_day, min_val=min_val, max_val=max_val, smooth_val=smooth_val, mode=curr_mode, points=points) #type:ignore

    def add_row(self, param="Phase", from_day=-1, to_day=-1, min_val=-0, max_val=-0, smooth_val = -1, mode='temp', points = []):
        r = self.row_indices[mode] + 1
        table_frame = self.table_frames[mode]
        lbl = tk.Label(table_frame, text=str(r), width=3)
        lbl.grid(row=r, column=0, padx=2)

        growing_cycle = int(self.crop_dict.get('growing_cycle', 366))
        min_range, max_range = (-10, 45) if mode == 'temp' else (0, 500)
        if to_day == -1:
            to_day = growing_cycle
        if min_val == -0 and max_val == -0:
            min_val, max_val = (5, 30) if mode == 'temp' else (50, 150)
        if smooth_val == -1:
            smooth_val = 3 if mode == 'temp' else 25
        if from_day == -1:
            if self.row_lists[mode]:
                lst = [int(spinbox[1].get()) for spinbox in self.row_lists[mode]] + \
                    [int(spinbox[2].get()) for spinbox in self.row_lists[mode]]
                from_day = max(lst) + 1
            else:
                from_day = 1

        # From
        spin_from = tk.Spinbox(table_frame, from_=from_day, to=growing_cycle, increment=1, width=5)
        spin_from.delete(0, tk.END) 
        spin_from.insert(0, str(from_day))
        spin_from.grid(row=r, column=1, padx=2)
        self._bind_spinbox(spin_from, mode, r)

        # To
        spin_to = tk.Spinbox(table_frame, from_=from_day, to=growing_cycle, increment=1, width=5)
        spin_to.delete(0, tk.END) 
        spin_to.insert(0, str(to_day))
        spin_to.grid(row=r, column=2, padx=2)
        self._bind_spinbox(spin_to, mode, r)

        # Min
        spin_min = tk.Spinbox(table_frame, from_=min_range, to=max_range, width=5, increment=.5 if mode == 'temp' else 5)
        spin_min.delete(0, tk.END) 
        spin_min.insert(0, str(min_val))
        spin_min.grid(row=r, column=3, padx=2)
        self._bind_spinbox(spin_min, mode, r)

        # Max
        spin_max = tk.Spinbox(table_frame, from_=min_range, to=max_range, width=5, increment=.5 if mode == 'temp' else 5)
        spin_max.delete(0, tk.END) 
        spin_max.insert(0, str(max_val))
        spin_max.grid(row=r, column=4, padx=2)
        self._bind_spinbox(spin_max, mode, r)

        # Smooth
        smooth_to = 10 if mode == 'temp' else 200
        spin_smooth = tk.Spinbox(table_frame, from_=0, to=smooth_to, width=5, increment=.1)
        spin_smooth.delete(0, tk.END) 
        spin_smooth.insert(0, str(smooth_val))
        spin_smooth.grid(row=r, column=5, padx=2)
        self._bind_spinbox(spin_smooth, mode, r)

        img_path = os.path.join('src', 'memship.png') if os.path.exists(os.path.join('src', 'memship.png')) else 'memship.png'
        img = Image.open(img_path).resize((24, 24), Image.LANCZOS) #type:ignore
        image = ImageTk.PhotoImage(img)
        if not hasattr(self, '_img_refs'):
            self._img_refs = []
        self._img_refs.append(image)
        btn = tk.Button(table_frame, image=image, width=24, height=24, command=lambda m=mode, row=r: self.open_memship_editor(m, row))
        btn.grid(row=r, column=6, padx=2)

        if points != []:
            spin_min.config(state='disabled')
            spin_max.config(state='disabled')
            spin_smooth.config(state='disabled')

        self.row_lists[mode].append((lbl, spin_from, spin_to, spin_min, spin_max, spin_smooth, points))
        self.row_indices[mode] += 1
        self.spbx_changed(mode=mode, row=r)

    def open_memship_editor(self, mode, row):
        row -= 1
        if len(self.row_lists[mode][row]) == 7 and self.row_lists[mode][row][6] != []:
            points = self.row_lists[mode][row][6]
        else:
            _min = float(self.row_lists[mode][row][3].get())
            _max = float(self.row_lists[mode][row][4].get())
            _smooth = float(self.row_lists[mode][row][5].get())
            points = self.smooth_curve_rounded(np.linspace(_min - _smooth, _max + _smooth + 1, 11), _min, _max, _smooth)
        editor = me.MembershipFunctionEditor(self.window, mode=mode, points=points)
        self.window.wait_window(editor)
        if editor.result is not None and editor.result != []:
            for i in range(3, 6):
                self.row_lists[mode][row][i].config(state='disabled')
            if len(self.row_lists[mode][row]) == 7:
                lst = list(self.row_lists[mode][row])
                if len(lst) > 6:
                    lst[6] = editor.result
                else:
                    lst.append(editor.result)
                self.row_lists[mode][row] = tuple(lst)
            else:
                self.row_lists[mode][row] += (editor.result,)
        else:
            for i in range(3, 6):
                self.row_lists[mode][row][i].config(state='normal')
            if len(self.row_lists[mode][row]) >= 7:
                values = list(self.row_lists[mode][row])
                del values[-1]
                self.row_lists[mode][row] = values
        raster = self.create_raster(mode=mode)
        self.plot(raster, mode=mode)

    def spbx_changed(self, mode, row):
        raster = self.create_raster(mode=mode)
        self.plot(raster, mode=mode)

    def remove_row(self, mode='temp'):
        if self.row_lists[mode]:
            widgets = self.row_lists[mode].pop()
            for w in widgets:
                w.destroy()
            self.row_indices[mode] -= 1

    def ok(self):
        for k in [k for k in self.crop_dict if k.startswith('phen')]:
            del self.crop_dict[k]
        for mode in ['temp', 'prec']:
            for idx, state in enumerate(self.row_lists[mode]):
                _from = int(state[1].get())
                _to = int(state[2].get())
                _min = float(state[3].get())
                _max = float(state[4].get())
                _smooth = float(state[5].get())
                try:
                    _custom = state[6]
                except:
                    _custom = []
                self.crop_dict[f'phen{idx}_{_from}-{_to}_{mode}'] = f'{_min},{_max},{_smooth},{_custom}'
        self.window.destroy()