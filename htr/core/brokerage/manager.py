
from importlib.machinery import SourceFileLoader
import os


class DataManager:
	"""."""

	def __init__(self):

		pass

	@staticmethod
	def get_data_handler(name):
		"""."""

		path = os.path.join(os.getcwd().split('/htr')[0], 'htr/core/brokerage')
		_class = None
		name_list = [name.replace(' ', '_').upper(), name.replace(' ', '').upper(), name.upper()]

		for root, dirs, files in os.walk(path):
			if files:
				for file in files:
					if file.endswith('.py') and not file.startswith('__init__') and not file.startswith('manager'):
						if file.replace('.py', '').upper() in name_list:
							file_path = os.path.join(root, file)
							module_name = file_path.split('/')[-1].split('.')[0].title().replace('_', '')
							my_module = SourceFileLoader(module_name, file_path).load_module()
							_class = getattr(my_module, module_name)

		return _class
