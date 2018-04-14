
import threading

from htr.core.configuration import *
from htr.core.data.handler import *
from htr.core.data.gatherer import CryptoGatherer
from htr.core.engines import Backtest as BacktestEngine
from htr.core.engines import LiveTrader as LiveTradeEngine
from htr.core.execution import *
from htr.core.factory.exceptions import InvalidFactoryTypeError
from htr.core.portfolio import *
from htr.core.portfolio.risk import *
from htr.helpers.class_finder import FilePaths, get_class
from htr.core.strategy import StrategyManager
from htr.core.brokerage import KrakenHandler


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

					if 'specs' in node.keys() and node['specs'] != '':
						reader = CsvReader(node['specs'])
						for specification in reader.read():
							node_context = Context()
							node_context.__dict__ = dict(list(merged_dict.items()) + list(specification.items()))
							node_context = NodeFactory.__set_name(node_context, specification)
							BacktestEngine(data_handler_class, SimulatedExecutionHandler, BacktestPortfolio, StrategyManager, risk_class, node_context).simulate_trading()

					else:
						node_context = Context()
						node_context.__dict__ = merged_dict
						BacktestEngine(data_handler_class, SimulatedExecutionHandler, BacktestPortfolio, StrategyManager, risk_class, node_context).simulate_trading()

		elif self.simulation is False:
			for node in self.context.live_nodes:
				if node['enabled'] == True:
					risk_class = get_class(node['risk_handler'], FilePaths.RISK_HANDLER)
					data_handler_class = get_class(node['data_handler'], FilePaths.DATA_HANDLER)

					## todo select broker handler from config file, unpack brokerage dict
					# broker_handler_class = get_class(node['brokerage'], FilePaths.BROKER_HANDLER)
					broker_handler_class = KrakenHandler

					## todo select portfolio from config file, unpack brokerage dict
					# portfolio_class = get_class(node['portfolio'], FilePaths.PORTFOLIO)
					portfolio_class = CryptoPortfolio

					# Merges backtest nodes dict attribute with specs dict.
					merged_dict = dict(list(node.items()) + list(getattr(self.context, Context.SPECS).items()))
					node_context = Context()
					node_context.__dict__ = merged_dict

					# Start Data Gatherer server.
					gatherer = CryptoGatherer(broker_handler_class)
					gatherer_thread = threading.Thread(target=gatherer.start_server)
					gatherer_thread.start()

					LiveTradeEngine(data_handler_class, MetatraderExecutionHandler, MetatraderPortfolio, StrategyManager,
					               risk_class, broker_handler_class, node_context).trade()

	def _initialize_concurrent(self):

		if self.simulation is True:
			for node in self.context.backtest_nodes:
				print(node, '\n', node['data_sources'], '\n', node['portfolio'])
			# launch backtest nodes

		elif self.simulation is False:
			for node in self.context.live_nodes:
				print(node)
			# launch live nods


	@classmethod
	def __set_name(cls, context, specification):
		"""."""

		name = ''
		for strategy in context.strategies:
			name += strategy + ' '

		for k, v in specification.items():
			name += k + v

		setattr(context, 'name', name)
		return context

class FactoryType:

	SEQUENTIAL = 'Sequential'
	CONCURRENT = 'Concurrent'

class RuntimeMode:

	SIMULATION = 'Simulation'
	LIVE = 'Live'
