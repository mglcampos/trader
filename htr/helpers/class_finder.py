
from importlib.machinery import SourceFileLoader
import os

def get_class(name, path):
	"""Returns a class by searching the file name."""

	path = os.path.join(os.getcwd().split('/tests')[0], path)
	_class = None
	name_list = [name.replace(' ', '_').upper(), name.replace(' ', '').upper(), name.upper()]

	for root, dirs, files in os.walk(path):
		if files:
			for file in files:
				if file.endswith('.py') and not file.startswith('__init__') and not file.startswith('manager'):
					if file.replace('.py', '').upper() in name_list:
						file_path = os.path.join(root, file)
						module_name = file_path.split('\\')[-1].split('.')[0].title().replace('_', '')
						my_module = SourceFileLoader(module_name, file_path).load_module()
						_class = getattr(my_module, module_name)

	return _class

class FilePaths:
	"""."""

	STRATEGY = 'htr\\core\\strategy'
	RISK_HANDLER = 'htr\\core\\portfolio\\risk'
	DATA_HANDLER = 'htr\\core\\data/handler'
	BROKER_HANDLER = 'htr\\core\\brokerage'