import os
from pathlib import Path
from typing import Optional
import base64 as b
import pyzipper
from importlib.metadata import version, PackageNotFoundError
import platform
import psutil
try:
    from src import read_climate_ini as rci
except:
    import read_climate_ini as rci
shared = 'c3VpdGFiaWxpdHk='

def write_debug_package(config: Optional[Path] = None):
    write_list = []
    with open('debug_file.txt', 'w') as wf:
        wf.write(f'config: {config}\n')
        wf.write(f'config_exists: {config is not None and os.path.exists(config)}\n')
        wf.write('current directory')
        wf.write('\nContents of current directory:\n')
        for idx, name in enumerate(os.listdir('.')):
            path = Path(name)
            type_info = 'file' if path.is_file() else 'directory' if path.is_dir() else 'unknown'
            wf.write(f'{idx} - {type_info} - {name} \n')
        write_list.append('debug_file.txt')

    if config and os.path.exists(config):
        write_list.append(config)
        config_dict = rci.read_ini_file(config)

        if os.path.exists(config_dict['files'].get('plant_param_dir', 'plant_params')):
            for i in os.listdir(config_dict['files'].get('plant_param_dir', 'plant_params')):
                write_list.append(os.path.join(config_dict['files'].get('plant_param_dir', 'plant_params'), i))

        if os.path.exists(config_dict['files'].get('climate_data_dir', False)):
            with open('clf.txt', 'w') as wf:
                wf.write('climate_files')
                for i in os.listdir(config_dict['files'].get('climate_data_dir')):
                     wf.write(f'{i}\n')
            write_list.append('clf.txt')

        write_list.append(check_all_paths_and_write_report(config_dict))

    if os.path.exists('preproc.ini'):
        write_list.append('preproc.ini')
    write_list.append(write_system_info())

    write_zip(write_list)

def write_system_info(filename="sys.txt"):
    pkgs = ["cartopy", "numpy", "scipy", "dask", "distributed", "matplotlib",
            "netCDF4", "rasterio", "psutil", "pyproj", "rio-cogeo", "numba",
            "scikit-image", "tkinter", "Pillow", "xarray"]
    with open(filename, "w") as f:
        f.write(f"OS: {platform.system()} {platform.release()}\n")
        f.write(f"Python: {platform.python_version()}\n")
        f.write(f"CPU cores: {psutil.cpu_count(logical=True)}\n")
        mem = psutil.virtual_memory()
        f.write(f"RAM total (GB): {mem.total / 1e9:.2f}\n")
        disk = psutil.disk_usage('.')
        f.write(f"Disk total (GB): {disk.total / 1e9:.2f}\n")
        f.write(f"Disk free (GB): {disk.free / 1e9:.2f}\n")
        f.write("\nPackage versions:\n")
        for p in pkgs:
            if p == "tkinter":
                try:
                    import tkinter
                    f.write("tk: installed\n")
                except ImportError:
                    f.write("tk: not installed\n")
                continue
            try:
                f.write(f"{p}: {version(p)}\n")
            except PackageNotFoundError:
                f.write(f"{p}: not installed\n")
    return filename

def check_all_paths_and_write_report(config_dict, output_file="file_report.txt"):
    with open(output_file, "w") as f:
        for section, keys in config_dict.items():
            f.write(f"--- Sektion [{section}] ---\n\n")
            for key, path in keys.items():
                # Nur gültige, nicht-leere Strings prüfen
                if not isinstance(path, str) or not path.strip():
                    continue

                exists = os.path.exists(path)
                filename = os.path.basename(path)
                size_str = "–"

                if exists:
                    if os.path.isfile(path):
                        size_bytes = os.path.getsize(path)
                        size_str = f"{size_bytes} Bytes"
                    elif os.path.isdir(path):
                        size_str = "<Directory>"
                else:
                    filename = "<Path not existing>"

                f.write(f"{key}:\n")
                f.write(f"  Path: {path}\n")
                f.write(f"  Existing: {exists}\n")
                f.write(f"  Name: {filename}\n")
                f.write(f"  Size: {size_str}\n\n")
    return output_file

def write_zip(filenames = []):
    try:
        os.path.exists('debug_file.zip') and os.remove('debug_file.zip') #type:ignore
        with pyzipper.AESZipFile('debug_file.zip', 'w', compression=pyzipper.ZIP_DEFLATED) as zf:
            #zf.setpassword(b.b64decode(shared).decode('utf-8').encode())
            for f in filenames:
                zf.write(f)
        os.path.exists('DumpFile.dmp') and os.remove('DumpFile.dmp') #type:ignore
        os.rename('debug_file.zip', 'DumpFile.dmp')
        for f in ['debug_file.txt', 'clf.txt', 'file_report.txt', 'sys.txt']:
            try:
                os.remove(f)
            except:
                pass
    except:
        input('General Error.')