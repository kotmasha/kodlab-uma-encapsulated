1 to build the project on linux run "sh make.sh", on windows, need to the environment on yourself
2 to run the project, run "source setEnv.sh" to set the correct PYTHONPATH
3 there is a test wrapper, when including it, just do:
    from test wrapper import *
    test = wrapper()
    test.....
4 when doing more development, you can set add log to your code, and check the log info easily, usually do _log->LOG_LEVEL_FUN("string")