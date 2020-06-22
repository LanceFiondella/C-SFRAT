import pkgutil
import importlib
from core.model import Model

for (module_loader, name, ispkg) in pkgutil.iter_modules(['./models']):
    importlib.import_module('.'+name, __package__)

modelList = {cls.__name__: cls for cls in Model.__subclasses__()}
