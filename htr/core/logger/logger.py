
from htr.core.strategy.strategy import Strategy
from htr.core.engines.backtest import Backtest

##todo plan what to log, divide into backtest and live trade log?
# class Logger:
# 	# Singleton
# 	_instance = None
#
# 	def Instance():
# 		if Logger._instance == None:
# 			Logger._instance = Logger()
#
# 		return Logger._instance
#
# 	def __init__(self):
# 		# Define writer.
# 		# self.writer = Printer()
#
# 		# TestCase Events
# 		Calculator.sum = Logger.Aspect.before(Calculator.sum, 'Starting test case execution "{}"', 'name')
# 		Calculator.sum = Logger.Aspect.after(Calculator.sum, 'Finished running test case in "{}ms"', 'execution_time')
#
# 		# Deluge Events
# 		# Deluge.add = logger.Aspect.after(Deluge.add, 'Added new torrent to the list')
# 		Calculator.sumstr = Logger.Aspect.before(Calculator.sumstr, 'Stating write test results in "{}" file({})',
# 		                                         'type', 'path')
# 		Calculator.sumstr = Logger.Aspect.after(Calculator.sumstr, 'Finished write test results in "{}" file ({})',
# 		                                        'type', 'path')
#
# 	def i(self, event):
# 		print('Info', event)
#
# 	def w(self, event):
# 		print('Warning', event)
#
# 	def e(self, event):
# 		print('Error', event)
#
# 	class Aspect:
# 		def before(func, msg, *attrs):
# 			def advice(*args, **kwargs):
# 				# Get attributes.
# 				a = map(lambda x: getattr(args[0], x), attrs)
#
# 				# Log event.
# 				Logger.Instance().i(msg.format(*a))
#
# 				# Run function.
# 				return func(*args, **kwargs)
#
# 			return advice
#
# 		def after(func, msg, *attrs):
# 			def advice(*args, **kwargs):
# 				# Run function.
# 				result = func(*args, **kwargs)
#
# 				# Get attributes.
# 				a = map(lambda x: getattr(args[0], x), attrs)
#
# 				# Log event.
# 				Logger.Instance().i(msg.format(*a))
#
# 				return result
#
# 			return advice