
class Context:

	BACKTEST_NODES = 'backtest_nodes'
	LIVE_NODES = 'live_nodes'
	SPECS = 'specifications'

	def __str__(self):
		return str(self.__dict__)