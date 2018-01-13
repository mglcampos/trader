
from htr.core.interpreter.exceptions import InvalidSyntaxError
from htr.core.interpreter.exceptions import UnknownOptionError

class OptionParser:
    """Parses Options and maps user commands into actions.

    Attributes:
        alias (str): Short string which represents the alias to be used on the CLI or OS Shell when starting the tool.
        handler (func): Default handler, function to be called when no handler is specified on the matched options.
        options (list<Option>): List of options which are parsable.
        options_alias_max_len (int): Dictates the maximum length of the options alias, used for syntax pretty print.
    """

    def __init__(self, alias, handler):
        """Class constructor, initializes instance.

        	Args:
        		alias (str): Keyword used for starting the script on App used for generating the syntax.
        		handler (func): Function to be called by default when no option with handler is matched.
        """

        self.alias = alias
        self.handler = handler
        self.options = []

        self.options_alias_max_len = 0

    def add(self, option):
        """Adds option to the parser and sorts all options according to the shorted alias.

        Args:
            option (Option): Option to be added to the list.
        """

        # Sort alias by size.
        option.alias = sorted(option.alias, key = lambda x: len(x))

        # Add option to list.
        self.options.append(option)

        # Determine padding space for syntax.
        l = len(', '.join(option.alias))
        if l > self.options_alias_max_len:
            self.options_alias_max_len = l

        # Sorts options by first alias.
        self.options = sorted(self.options, key = lambda x: x.alias[0])

    def parse(self, command):
        """Parses a command and maps into an action.

        If the matched option has a valid handler, then it will be invoked otherwise it will fall-back to the
        OptionParser default handler.

        Args:
            command (list<str>): List of string tokens with options (e.g.: ['--area', 'Wi-Fi', '--type', 'Functional']).

        Raises:
            UnknownOptionError: A match was not found for the provided option.
        """

        parsed_options = {}

        # Iterate through the command options.
        idx = 0
        while idx < len(command):
            # Assume no match was found.
            found = False

            # Iterate through the registered options.
            for parser_opt in self.options:

                # Compare the command with the valid option alias.
                if command[idx] in parser_opt.alias:

                    # Call the matching handler and exit.
                    if parser_opt.handler is not None:
                        return parser_opt.handler()

                    # Parse the second option as an argument.
                    else:
                        try:
                            # Duplicate option.
                            if parser_opt.name in parsed_options.keys():
                                raise InvalidSyntaxError('Option "{}" is duplicate.'.format(command[idx]))

                            # Parse option and argument.
                            parsed_options[parser_opt.name] = command[idx + 1]
                            idx += 2
                            found = True
                            break

                        # Raise exception if no argument was provided.
                        except:
                            raise InvalidSyntaxError('Option "{}" requires an argument.'.format(command[idx]))

            # If a match was found continue parsing the remaining options.
            if found:
                continue

            # Raise exception if no match was found on the registered option list.
            else:
                raise UnknownOptionError('Invalid option "{}".'.format(command[idx]))

        # Run Default Handler.
        return self.handler(**parsed_options)

    def syntax(self):
        """Generate a syntax usage string.

        It contains an auto-generated list of sorted (by first alias) options, which also includes are alias variants
        and brief description.

        Returns:
            str: Syntax usage.
        """

        # If there are no registered options return basic syntax with no option details.
        if len(self.options) == 0:
            return 'Usage: {}'.format(self.alias)

        # Add basic usage syntax.
        s = ['Usage: {} [options]'.format(self.alias), 'Where:']

        # Add a tab space before description.
        max_len = self.options_alias_max_len + 4

        # Pretty print options.
        for o in self.options:
            alias = ', '.join(o.alias)
            padding = ' ' * (max_len - len(alias))
            s.append('    {}{}{}'.format(alias, padding, o.description))

        return '\n'.join(s)
