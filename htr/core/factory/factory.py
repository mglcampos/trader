
from htr.core.configuration import *
from htr.core.data.handler import *
from htr.core.engines import Backtest as BacktestEngine
from htr.core.execution import *
from htr.core.factory.exceptions import InvalidFactoryTypeError
from htr.core.portfolio import *
from htr.core.portfolio.risk import *
from htr.helpers.class_finder import FilePaths, get_class
from htr.core.strategy import StrategyManager


class NodeFactory():
	"""Receives factory_type and simulation from the interpreter and instantiates nodes."""

	def __init__(self, factory_type, mode):
		"""Init variables and read configuration.

		Args:
			factory_type (str): Should be CONCURRENT OR SEQUENTIAL.
			mode (str): Should be Simulation or Live
		"""

		self.type = factory_type
		if mode.upper() == RuntimeMode.LIVE.upper():
			self.simulation = False
		else:
			self.simulation = True
		# Get running context
		self.context = ConfigManager(ConfigFiles.CONFIG).get_context()

		if self.type.upper() == FactoryType.SEQUENTIAL.upper():
			self._initialize_sequential()

		elif self.type.upper() == FactoryType.CONCURRENT.upper():
			self._initialize_concurrent()

		else:
			raise InvalidFactoryTypeError('Invalid Factory type.')

	def _initialize_sequential(self):
		"""Initializes a backtest node for each node in context."""

		if self.simulation is True:
			for node in self.context.backtest_nodes:
				if node['enabled'] == True:
					risk_class = get_class(node['risk_handler'], FilePaths.RISK_HANDLER)
					data_handler_class = get_class(node['data_handler'], FilePaths.DATA_HANDLER)
					# Merges backtest nodes dict attribute with specs dict.
					merged_dict = dict(list(node.items()) + list(getattr(self.context, Context.SPECS).items()))
					node_context = Context()
					node_context.__dict__ = merged_dict
					BacktestEngine(data_handler_class, SimulatedExecutionHandler, BacktestPortfolio, StrategyManager, risk_class, node_context).simulate_trading()

		elif self.simulation is False:
			for node in self.context.live_nodes:
				if node['enabled'] == True:
					risk_class = get_class(node['risk_handler'], FilePaths.RISK_HANDLER)
					data_handler_class = get_class(node['data_handler'], FilePaths.DATA_HANDLER)
					broker_handler_class = get_class(node['brokerage', FilePaths.BROKER_HANDLER])
					# Merges backtest nodes dict attribute with specs dict.
					merged_dict = dict(list(node.items()) + list(getattr(self.context, Context.SPECS).items()))
					node_context = Context()
					node_context.__dict__ = merged_dict
					## todo init data gatherer
					# BacktestEngine(data_handler_class, SimulatedExecutionHandler, BacktestPortfolio, StrategyManager,
					#                risk_class, node_context).simulate_trading()
					print(node)
				# launch live nods


	def _initialize_concurrent(self):

		if self.simulation is True:
			for node in self.context.backtest_nodes:
				print(node, '\n', node['data_sources'], '\n', node['portfolio'])
			# launch backtest nodes

		elif self.simulation is False:
			for node in self.context.live_nodes:
				print(node)
			# launch live nods

class FactoryType:

	SEQUENTIAL = 'Sequential'
	CONCURRENT = 'Concurrent'

class RuntimeMode:

	SIMULATION = 'Simulation'
	LIVE = 'Live'
