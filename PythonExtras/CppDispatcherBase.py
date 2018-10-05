import ctypes
import h5py
import os
import platform
import time
import math
from typing import Tuple, Union, List, Dict, Callable

import numpy as np


__all__ = ['CppFunc', 'CppDispatcherBase']


def CppFunc(func):

    def _decorator(*args, **kwargs):
        dispatcher = args[0]  # type: CppDispatcherBase
        assert(isinstance(dispatcher, CppDispatcherBase))

        funcName = func.__name__
        func(*args, **kwargs)

        assert(funcName in dispatcher._signatures)
        signature = dispatcher._signatures[funcName]
        signatureArgs = signature[0]  # type: List[Callable[[object], object]]

        dll = dispatcher._get_c_dll()
        cppFunc = getattr(dll, funcName)
        cppFunc.argtypes = signatureArgs
        if signature[1] is not None:
            cppFunc.restype = signature[1]

        # The first argument is the CppDispatcher 'self'.
        args = args[1:]

        argsConverted = []
        # kwargsConverted = {}
        for i in range(len(args)):
            arg = args[i]
            argType = signatureArgs[i]
            if isinstance(arg, np.ndarray):
                argsConverted.append(args[i].ctypes.data_as(signatureArgs[i]))
            else:
                argsConverted.append(signatureArgs[i](args[i]))

        if kwargs.__len__() > 0:
            raise RuntimeError("Named arguments aren't yet supported.")

        return cppFunc(*argsConverted, **kwargs)

    _decorator.__origName = func.__name__

    return _decorator


class CppDispatcherBase:
    """
    An attempt to create a more flexible way of defining the C++ methods in Python.
    Methods aren't explicitly wrapped, but rather their signatures are described,
    performing the calls semi-automatically.
    """
    
    def __init__(self, dllDir: str, dllName: str):
        self._dllDir = dllDir
        self._dllName = dllName
        
        self._dll = None
        self._signatures = {}  # type: Dict[str, Tuple[List[object], object]]

    def _get_c_dll(self):
        if self._dll is None:
            platformName = platform.system()
            if platformName == 'Windows':
                dllPath = os.path.join(self._dllDir, self._dllName + '.dll')
                self._dll = ctypes.CDLL(dllPath)
            elif platformName == 'Linux':
                soPath = os.path.join(self._dllDir, self._dllName + '.so')
                self._dll = ctypes.CDLL(soPath)
            else:
                raise RuntimeError("Unknown platform: {}".format(platformName))

        return self._dll

    def _define_signature(self, funcName, argTypes, returnType):
        self._signatures[funcName] = (argTypes, returnType)