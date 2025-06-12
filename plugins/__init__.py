import os
import importlib

def load_plugins():
    plugins = []
    base = os.path.dirname(__file__)
    for fname in os.listdir(base):
        if fname.endswith('.py') and fname != '__init__.py':
            name = fname[:-3]
            mod = importlib.import_module(f'plugins.{name}')
            plugins.append(mod)
    return plugins

