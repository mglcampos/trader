
import sys

#from htr.core.logger import Logger
from htr.core.interpreter import Interpreter


if __name__ =='__main__':
    # Init Logger.
    #Logger.Instance()

    # Delegate User Commands to CLI Interpreter.
    Interpreter().parser.parse(sys.argv[1:])