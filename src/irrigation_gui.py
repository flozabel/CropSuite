import tkinter as tk

class IrrigationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Irrigation")

        self.months = [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ]
        self.vars = {}
        self.checkbuttons = {}

        # Nur eine Checkbox pro Zeile mit Monatsname und Status
        for i, month in enumerate(self.months):
            var = tk.BooleanVar(value=False)  # False = Rainfed, True = Irrigated
            cb = tk.Checkbutton(
                root,
                text=f"{month}: Rainfed",
                variable=var,
                anchor="w",
                width=25,
                command=lambda m=month: self.update_label(m)
            )
            cb.grid(row=i, column=0, padx=10, pady=2, sticky="w")
            self.vars[month] = var
            self.checkbuttons[month] = cb

        # Select All / None Buttons
        tk.Button(root, text="Select All", command=self.select_all).grid(
            row=12, column=0, padx=10, pady=(10, 2), sticky="ew"
        )
        tk.Button(root, text="Select None", command=self.select_none).grid(
            row=13, column=0, padx=10, pady=(0, 2), sticky="ew"
        )

        # OK Button
        tk.Button(root, text="OK", command=self.ok_pressed).grid(
            row=14, column=0, padx=10, pady=(5, 10), sticky="ew"
        )

    def update_label(self, month):
        var = self.vars[month]
        cb = self.checkbuttons[month]
        status = "Irrigated" if var.get() else "Rainfed"
        cb.config(text=f"{month}: {status}")

    def select_all(self):
        for month in self.months:
            self.vars[month].set(True)
            self.update_label(month)

    def select_none(self):
        for month in self.months:
            self.vars[month].set(False)
            self.update_label(month)

    def ok_pressed(self):
        selection = {month: self.vars[month].get() for month in self.months}
        print("Selection:", selection)
        self.root.destroy()

# Start
if __name__ == "__main__":
    root = tk.Tk()
    app = IrrigationApp(root)
    root.mainloop()