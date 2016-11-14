Logging
=======

pySOT logs all important events that occur during the optimization process. The user can
specify what level of logging he wants to do. The five levels are:

- critical
- error
- warning
- info
- debug

Function evaluations are recorded on the info level, so this is the recommended level for pySOT.
There is currently nothing that is being logged on the debug level, but better logging for
debugging will likely be added in the future. Crashed evaluations are recorded on the warning
level.

More information about logging in Python 2.7 is available at:
`https://docs.python.org/2/library/logging.html <https://docs.python.org/2/library/logging.html>`_.
