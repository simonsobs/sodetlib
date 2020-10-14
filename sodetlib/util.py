"""

Module for miscellaneous functions and classes that are useful in many sodetlib
scripts.

"""
class TermColors:
    HEADER = '\n\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def cprint(msg, style=TermColors.OKBLUE):
    if style == True:
        style = TermColors.OKGREEN
    elif style == False:
        style = TermColors.FAIL
    print(style + str(msg) + TermColors.ENDC)

