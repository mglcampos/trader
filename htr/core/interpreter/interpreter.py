
from htr.core.interpreter import Option
from htr.core.interpreter import OptionParser
from htr.core.factory import NodeFactory, FactoryType, RuntimeMode


class Interpreter:
    """Basic Command Line Interface (CLI which parses user commands and maps into actions.

    Attributes:
        self.parser (OptionParser): Option handler used for parsing the command.
    """

    VERSION = 'version 0.2'

    def __init__(self):
        """Class constructor, initializes instance along with parsable options."""

        self.parser = OptionParser("htr", Interpreter._run_factory)

        # Default Options
        self.parser.add(Option('factory_type',
                                      ['--factory_type', '-f'],
                                      'Sequential or Concurrent.'))

        self.parser.add(Option('mode',
                                      ['--mode', '-m'],
                                      'Backtest or live trade.'))


        # Support options.
        self.parser.add(Option('help',
                               ['-h', '--help'],
                               'Prints syntax usage and available options.',
                               handler = self._print_usage))

        self.parser.add(Option('version',
                               ['-v', '--version'],
                               'Prints build version.',
                               handler = self._print_version))

    def parse(self, command):
        """Parses a user command and maps into a function or action.

        Args:
            command (list<str>): User command containing none of various options.
        """

        try:
            # Clean exit.
            self.parser.parse(command)
            return 0

        except Exception as e:
            # Exit with errors.
            print('(Error): {}'.format(e))
            self._print_usage()
            return 1

    def _print_usage(self):
        """Prints help and syntax usage."""

        print(self.parser.syntax())

    def _print_version(cls):
        """Prints build and release version."""

        print(cls.VERSION)

    def _run_factory(factory_type = None, mode = None):
        """Launches the test manager.

        Args:
            factory_type (str): Factory type, Concurrent or Sequential.
            mode (str): Simulation or Live.
        """

        if factory_type == '' or factory_type is None:
            factory_type = FactoryType.SEQUENTIAL

        if mode == '' or mode is None:
            mode = RuntimeMode.LIVE

        NodeFactory(factory_type, mode)

        # try:
        #     NodeFactory(type, simulation)
        #
        # except InvalidTestAreaError:
        #     print('[Error]: No Test-Cases have been found for the specified "{}" test area. '
        #           'Please make sure that these are available on "tests/book/".'.format(area))
        #
        # except InvalidTestTypeError:
        #     print('[Error]: No Test-Cases have been found for the specified "{}" test type.'.format(type))
        #
        # except EmptyRepositoryError:
        #     print(
        #         '[Error]: No Test-Cases have been found for the specified "{}" test area and "{}" test type. '
        #         'Please make sure that these are available on "tests/book/".'.format(area, type))
        #
        # except NoRootAccessError:
        #     print('[Error]: User has no root privileges, please run htr with the sudo command.')
