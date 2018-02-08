
import os

from htr.core.configuration import JsonReader, Context


class ConfigManager:
    """Manages configuration, imports it from config file and transforms it into Context object."""

    def __init__(self, config):

        self.context = Context()
        print('{}{}{}'.format(os.getcwd().split('/tests')[0], '/config/', config))
        self.config = JsonReader('{}{}{}'.format(os.getcwd().split('/tests')[0], '/config/', config)).read()
        self.backtest_nodes = []
        self.live_nodes = []
        self._load_backtest_nodes()
        self._load_live_nodes()
        self._load_specs()

    def _load_backtest_nodes(self):
        """Adds the Backtest Nodes dict list to the context object as an attribute."""

        for c in self.config[Context.BACKTEST_NODES]:
            self.backtest_nodes.append(c)

        setattr(self.context, Context.BACKTEST_NODES, self.backtest_nodes)

    def _load_live_nodes(self):
        """Adds the Live Nodes dict list to the context object as an attribute."""

        for c in self.config[Context.LIVE_NODES]:
            self.live_nodes.append(c)

        setattr(self.context, Context.LIVE_NODES, self.live_nodes)

    def _load_specs(self):
        """Adds the Specs dict to the context object as an attribute."""

        for s in self.config[Context.SPECS]:
            setattr(self.context, Context.SPECS, s)

    def get_context(self):
        """."""

        return self.context


class ConfigFiles:

    CONFIG = 'config.json'