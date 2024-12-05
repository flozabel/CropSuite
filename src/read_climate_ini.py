import configparser


def read_ini_file(file_path, comment_char='!'):
    """
    Reads an INI configuration file and returns its contents as a nested dictionary.
    Args:
        file_path (str): The path to the INI file to be read.
        comment_char (str, optional): The character used to mark comments in the INI file. Defaults to '!'.
    Returns:
        dict: A nested dictionary representing the contents of the INI file. The outer keys are the sections
        in the file, and the inner keys and values correspond to the key-value pairs in each section.
    """
    config = configparser.ConfigParser()
    config.read(file_path)
    result = {}
    for section in config.sections():
        result[section] = {}
        for key, value in config.items(section):
            result[section][key] = value.split(comment_char)[0].strip()
    return replace_yn_with_boolean(result)


def replace_yn_with_boolean(data):
    """
    Replace all values of 'y' and 'n' in a dictionary with True and False, respectively.
    Args:
    - dict_in: A dictionary to be modified.
    Returns:
    - A dictionary with modified values.
    Example:
    >>> my_dict = {'a': 'y', 'b': 'n', 'c': 'y', 'd': 'n'}
    >>> new_dict = replace_yn_with_boolean(my_dict)
    >>> print(new_dict)
    {'a': True, 'b': False, 'c': True, 'd': False}
    """
    
    for section in data:
        for key in data[section]:
            value = str(data[section][key]).lower()
            if value == 'y':
                data[section][key] = True
            elif value == 'n':
                data[section][key] = False
    return data


def print_dict(d, indent=0):
    """
    Recursively prints the key-value pairs of a nested dictionary.
    Args:
        d (dict): The dictionary to be printed.
        indent (int, optional): The number of spaces used to indent each level of the dictionary. Defaults to 0.
    Returns:
        None
    """
    for key, value in d.items():
        if isinstance(value, dict):
            print(' ' * indent + str(key) + ':')
            print_dict(value, indent + 2)
        else:
            print(' ' * indent + str(key) + ': ' + str(value))


def write_config(config_dict, ini_path='config.ini'):
    with open(ini_path, 'w') as wf:
        for main_key in list(config_dict.keys()):
            wf.write(f'[{main_key}]')
            for sub_key in list(config_dict.get(main_key).keys()):
                value = config_dict.get(main_key).get(sub_key)
                if not str(main_key).startswith('parameters.'):
                    if value == True:
                        value = 'y'
                    elif value == False:
                        value = 'n'
                wf.write('\n'+f'{sub_key} = {value}')
            wf.write('\n\n')

