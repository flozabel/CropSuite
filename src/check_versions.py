import sys
import importlib


def check_versions():
    compatible_pythons = ['3.8', '3.9', '3.10', '3.11', '3.12', '3.13']
    current_version = f"{sys.version_info.major}.{sys.version_info.minor}"

    if current_version in compatible_pythons:
        print(f"Python version {current_version} is supported.")
    else:
        print(f"Python {current_version} is NOT compatible.\nExit")
        input()
        sys.exit(1)

    compatible_packages = {
        'numpy': ['1.24.3', '1.25.0', '1.26.4', '2.2.4'],
        'scipy': ['1.9.3', '1.11.2', '1.11.4', '1.14.1', '1.15.1', '1.15.2'],
        'rasterio': ['1.3.9', '1.3.10', '1.3.11', '1.4.1', '1.4.3'],
        'matplotlib': ['3.6.3', '3.9.2', '3.10.1'],
        'xarray': ['2023.6.0', '2024.10.0', '2025.1.2', '2025.3.1'],
        'numba': ['0.60.0', '0.61.0', '0.61.2'],
        'rio_cogeo': ['5.3.3', '5.3.6', '5.4.1'],
        'cartopy': ['0.22.0', '0.23.0', '0.24.1'],
        'dask': ['2023.3.2', '2023.12.1', '2023.12.1+dfsg', '2024.10.0', '2025.3.0'],
        'netCDF4': ['1.6.4', '1.7.2'],
        'PIL': ['10.2.0', '10.4.0', '11.0.0', '11.1.0'],
        'pyproj': ['3.6.1', '3.7.0', '3.7.1'],
        'skimage': ['0.25.1', '0.25.2']
    }

    notcomp = []

    for pkg, allowed_versions in compatible_packages.items():
        try:
            module = importlib.import_module(pkg)
            version = getattr(module, '__version__', 'unknown')
            if version not in allowed_versions:
                notcomp.append(f'{module.__name__} may not be compatible with {version}.\n    Please use one of the following versions: {", ".join(allowed_versions)}\n    >> pip install {module.__name__}=={allowed_versions[-1]}')
        except ModuleNotFoundError:
            notcomp.append(f"{pkg} is NOT installed.")


    if len(notcomp) > 0:
        print("\nSome packages are not compatible with the current version:\n")
        for pkg in notcomp:
            print(f' -> {pkg}\n')

        print('''The list of compatible package versions provided is not exhaustive.
The versions listed here have been tested and confirmed to be compatible.
While other versions may work as well, they could potentially cause issues.
Therefore, it is recommended to use the tested and compatible versions.''')
        if input('\nEnter "y" to continue anyway, press any other key to exit: ').strip().lower() != 'y':
            print("Exiting...")
            exit()

    return True