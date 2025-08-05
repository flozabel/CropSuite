import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np


class MembershipFunctionEditor(tk.Toplevel):
    def __init__(self, parent, mode, points = []):
        super().__init__(parent)
        self.mode = mode
        self.title("Custom Membership Function")
        self.geometry("600x600")

        bg_col = tk.Tk()
        bg_col.withdraw()
        self.bcolor = bg_col.winfo_rgb(bg_col.cget("bg"))
        self.bcolor = "#{:02x}{:02x}{:02x}".format(self.bcolor[0] // 256, self.bcolor[1] // 256, self.bcolor[2] // 256)
        bg_col.destroy()

        self.fig = Figure(figsize=(4, 4))
        self.fig.patch.set_facecolor(self.bcolor)
        self.ax = self.fig.add_subplot(111)

        x_vals = [p[0] for p in points] if points else []
        if mode == 'temp':
            x_min, x_max = (-5, 45) if not points else (np.min([min(x_vals) - 2, -5]), np.max([max(x_vals) + 2, 45]))
        else:
            x_min, x_max = (-20, 500) if not points else (np.min([min(x_vals) - 2, -20]), np.max([max(x_vals) + 2, 500]))
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(-5, 105)
        self.ax.axhline(0, color='black')
        self.ax.axhline(100, color='black')
        if mode == 'prec':
            self.ax.axvline(0, color='black')
        ticks = np.arange(x_min, x_max+1, 5) if mode == 'temp' else np.arange(x_min, x_max+1, 20)
        self.ax.set_xticks(ticks)
        self.ax.set_yticks(np.arange(0, 101, 10))
        if len(ticks) > 20:
            self.ax.set_xticklabels(ticks, rotation=90)
        self.ax.set_xlabel("Temperature [°C]" if mode == 'temp' else "Precipitation [mm]", fontsize=10)
        self.ax.set_ylabel("Suitability", fontsize=10)
        self.ax.grid(linestyle='--', alpha=.3, linewidth=1)

        self.points = points
        self.selected_idx = None
        self.line, = self.ax.plot([], [], linewidth=2, marker='o', picker=5)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.canvas.mpl_connect('motion_notify_event', self.on_drag)
        self.canvas.mpl_connect('button_release_event', self.on_release)

        label_frame = tk.Frame(self)
        label_frame.pack(pady=10, fill='x')
        tk.Label(label_frame, text='Left click: Add new data point at cursor position', font=('TkDefaultFont', 8)).pack(anchor='w', padx=5)
        tk.Label(label_frame, text='Left click and drag: Move data point', font=('TkDefaultFont', 8)).pack(anchor='w', padx=5)
        tk.Label(label_frame, text='Right click: Delete data point at cursor position', font=('TkDefaultFont', 8)).pack(anchor='w', padx=5)

        btn_frame = tk.Frame(self)
        btn_frame.pack(pady=10, fill='x')
        ttk.Button(btn_frame, text="Clear custom membership function", command=self.delete_all_points).pack(side='left', pady=5, padx=5)
        ttk.Button(btn_frame, text="OK", command=self.on_ok).pack(side='right', pady=5, padx=5)

        self.result = None
        self.update_plot()

    def delete_all_points(self):
        self.points = []
        self.update_plot()

    def snap(self, x, y):
        return np.clip(round(x), -5 if self.mode == 'temp' else 0, 45 if self.mode == 'temp' else 500), np.clip(round(y), 0, 100)

    def on_click(self, event):
        if not event.inaxes:
            return
        x, y = self.snap(event.xdata, event.ydata)
        if event.button == 1:
            for i, (px, py) in enumerate(self.points):
                if px == x:
                    self.points[i] = (x, y)
                    self.selected_idx = i
                    self.update_plot()
                    return
            self.points.append((x, y))
            self.points.sort()
            self.update_plot()
        elif event.button == 3:
            for i, (px, py) in enumerate(self.points):
                if abs(px - x) <= 0.5 and abs(py - y) <= 2:
                    del self.points[i]
                    self.selected_idx = None
                    self.update_plot()
                    return

    def on_drag(self, event):
        if self.selected_idx is None or not event.inaxes or event.button != 1:
            return
        x, y = self.snap(event.xdata, event.ydata)
        if any(px == x and i != self.selected_idx for i, (px, _) in enumerate(self.points)):
            return
        self.points[self.selected_idx] = (x, y)
        self.update_plot()

    def on_release(self, event):
        self.selected_idx = None

    def update_plot(self):
        x_vals, y_vals = zip(*self.points) if self.points else ([], [])
        self.line.set_data(x_vals, y_vals)
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw() #type:ignore

    def on_ok(self):
        self.result = self.points.copy()      
        self.destroy()

    def destroy(self):
        self.canvas.get_tk_widget().destroy() #type:ignore
        self.canvas.figure = None #type:ignore
        self.canvas = None
        plt.close('all')
        super().destroy()

    def get_result(self):
        return self.result