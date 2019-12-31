from abc import ABC
from typing import Any

import numpy as np

from .definedTypes import Indice, Shape, IndexMap
from .utils import pp, intDot, initParams


class IndexConverter(ABC):
    def __call__(self, outIndice: Indice) -> Indice:
        return ()

    def __repr__(self, out: str = "", suf: str = "") -> str:
        return self.reprPrefix(out) + self.reprMain() + self.reprSuffix(suf)

    @staticmethod
    def reprPrefix(out: str = "") -> str:
        return "{} = ".format(out) if out else ""

    @staticmethod
    def reprSuffix(suf: str = "") -> str:
        return " = {}".format(suf) if suf else ""

    @staticmethod
    def reprMain() -> str:
        return ""

    def setModValues(self, modValues: Shape) -> Any:
        return self

    def nthBound(self, i: int) -> bool:
        return False


class LinearIndiceConverter(IndexConverter):
    def __init__(self, linearCoeff, b, modValue=-1, indexModCoeff=0, indexModValue=1):
        super().__init__()
        self.linearCoeff = linearCoeff
        self.b = b
        self.modValue = initParams(modValue, b)
        self.indexModCoeff = initParams(indexModCoeff, linearCoeff)
        self.indexModValue = initParams(indexModValue, linearCoeff)

    def __call__(self, outIndice: Indice) -> Indice:
        if self.modValue == 1:
            return 0,
        linearSum = intDot(outIndice, self.linearCoeff)
        modSum = intDot([x % m for (x, m) in zip(outIndice, self.indexModValue)], self.indexModCoeff)
        res = linearSum + modSum + self.b
        return (int(res % self.modValue),) if self.modValue > 1 else (int(res),)

    def setModValues(self, modValue: int) -> Any:
        self.modValue = modValue
        return self

    def nthBound(self, i: int) -> bool:
        return self.linearCoeff[i] != 0

    def reprMain(self) -> str:
        linearItem = " + ".join(
            [
                pp("|_{1:flag}*x{0:.0f}_|", ai).format(i, ai)
                for (i, ai) in enumerate(self.linearCoeff)
                if ai != 0
            ]
        )
        modItem = " + ".join(
            [
                pp("|_{1:flag}*mod(x{0:d}, {2:.0f})_|", bi).format(i, bi, ci)
                for (i, (bi, ci)) in enumerate(zip(self.indexModCoeff, self.indexModValue))
                if bi != 0
            ]
        )
        bItem = pp("{:flag}", self.b).format(self.b) if self.b != 0 else ""
        repr0 = (
            " + ".join([x for x in (linearItem, modItem, bItem) if x != ""])
                 .replace("+-", "-")
                 .replace("+ -", "-")
                 .replace("+ |_-", "- |_")
                 .replace("|_-", "-|_")
                 .replace("1*", "")
        )
        if repr0 == "":
            return "0"
        else:
            return "mod({}, {:.0f})".format(repr0, self.modValue) if self.modValue > 1 else repr0


class LinearIndexConverter(IndexConverter):
    # mod(floor(linearCoeffs@xi) + bs + floor(indexModCoeffs@mod(xi, indexModValues)), modValues)
    def __init__(self, linearCoeffs, bs, modValues=-1, indexModCoeffs=0, indexModValues=1):
        super().__init__()
        self.linearCoeffs = linearCoeffs
        self.bs = bs
        self.modValues = initParams(modValues, bs)
        self.indexModCoeffs = initParams(indexModCoeffs, linearCoeffs)
        self.indexModValues = initParams(indexModValues, linearCoeffs)
        self.converters = [
            LinearIndiceConverter(linearCoeff, b, modValue, indexModCoeff, indexModValue)
            for (linearCoeff, b, modValue, indexModCoeff, indexModValue) in zip(
                self.linearCoeffs, self.bs, self.modValues, self.indexModCoeffs, self.indexModValues,
            )
        ]

    def __call__(self, outIndice: Indice) -> Indice:
        return tuple([item(outIndice)[0] for item in self.converters])

    def setModValues(self, modValues: Shape) -> Any:
        self.modValues = (
            modValues * np.ones_like(np.array(self.bs))
            if isinstance(modValues, (int, float))
            else modValues
        )
        self.converters = [item.setModValues(modValue) for (item, modValue) in zip(self.converters, self.modValues)]
        return self

    def __repr__(self, out: str = "", suf: str = ""):
        return "\n".join([
            item.__repr__("{}{:d}".format(out, i), suf) for (i, item) in enumerate(self.converters)
        ])

    def nthBound(self, i: int) -> bool:
        return np.any(np.array(self.linearCoeffs)[:, i] != 0)


class UnitIndexConverter(LinearIndiceConverter):
    def __init__(self, outLen: int, inLen: int):
        super().__init__(np.zeros((inLen, outLen)).tolist(), [0] * inLen)
        self.inLen = inLen
        self.outLen = outLen

    def __call__(self, outIndice: Indice) -> Indice:
        return tuple([0] * self.inLen)

    def reprMain(self) -> str:
        return "0"


class ZeroIndexConverter(IndexConverter):
    def __init__(self):
        super().__init__()

    def __call__(self, outIndice: Indice) -> Indice:
        return ()

    def reprMain(self) -> str:
        return "()"


class NullIndexConverter(IndexConverter):
    def __init__(self):
        super().__init__()

    def __call__(self, outIndex: Indice) -> Indice:
        return ()

    def reprMain(self) -> str:
        return "_"


class FixIndexConverter(IndexConverter):
    def __init__(self, indexMap: IndexMap):
        super().__init__()
        self.indexMap = indexMap

    def __call__(self, outIndice: Indice) -> Indice:
        return self.indexMap[outIndice]

    def reprMain(self) -> str:
        return "Fixed: " + repr(self.indexMap)

    def nthBound(self, i: int) -> bool:
        return True
