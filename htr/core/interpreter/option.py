
class Option:
	"""Object which will be handled to an Interpreter to be parsed and mapped to a specific action.

	Attributes:
		name (str): Option name (e.g.: 'version').
		alias (list<str>): List of possible tokens to invoke the action (e.g.: ['-v', '--version']).
		description (str): Short description (e.g.: 'Prints version number').
		handler (func): Function to be called (e.g.: 'Core.version').
	"""

	def __init__(self, name, alias, description, handler = None):
		"""Class constructor, initializes instance.

		Args:
			name (str): Name of the option, also used as attribute passed to an interpreter.
			alias (list<str>): List of possible alias for the option.
			description (str): Textual description used for generating the syntax.
			handler (func, optional): Function to invoke when the option is matched, defaults to 'None'.
		"""

		self.name = name
		self.alias = alias
		self.description = description
		self.handler = handler

	def __str__(self):
		"""Converts object to a human-readable string.

		Returns:
			str: Containing most relevant attributes.
		"""

		return '[name="{}", alias="{}", description="{}", handler="{}"]'.format(
			self.name,
			self.alias,
			self.description,
			self.handler)
