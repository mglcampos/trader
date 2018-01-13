
from __future__ import print_function
import datetime
import time

try:
    import Queue as queue
except ImportError:
    import queue
from htr.core.events.event import FillEvent, OrderEvent
from htr.core.execution.execution import ExecutionHandler

class SimulatedExecutionHandler(ExecutionHandler):
    """
    The simulated execution handler simply converts all order
    objects into their equivalent fill objects automatically
    without latency, slippage or fill-ratio issues.
    This allows a straightforward "first go" test of any strategy,
    before implementation with a more sophisticated execution
    handler.
    """
    def __init__(self, context, events):
        """
        Initialises the handler, setting the event queues
        up internally.
        Parameters:
        events - The Queue of Event objects.
        """

        self.events = events
        self.commission = context.commission
        
    def execute_order(self, event):
        """
        Simply converts Order objects into Fill objects.
        Parameters:
        event - Contains an Event object with order information.
        """
        if event.type == 'ORDER':
            fill_event = FillEvent(
               event.timestamp, event.symbol,
                None, event.quantity, event.direction, None , commission = self.commission, position=event.order_type  ##TODO hardcoded exchange
            )
            self.events.put(fill_event)


