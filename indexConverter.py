from typing import Any

import numpy as np

from definedTypes import Indice, Shape
from utils import pp, intDot


class IndexConverter: # mod(floor(linearCoeffs@xi) + bs + floor(indexModCoeffs@mod(xi, indexModValues)), modValues)
    def __init__(
            self, linearCoeffs, bs, modValues=-1, indexModCoeffs=0, indexModValues=1
    ):
        self.linearCoeffs = linearCoeffs
        self.bs = bs
        self.modValues = (
            modValues * np.ones_like(np.array(bs))
            if isinstance(modValues, (int, float))
            else modValues
        )
        self.indexModCoeffs = (
            indexModCoeffs * np.zeros_like(linearCoeffs)
            if isinstance(indexModCoeffs, (int, float))
            else indexModCoeffs
        )
        self.indexModValues = (
            indexModValues * np.ones_like(linearCoeffs)
            if isinstance(indexModValues, (int, float))
            else indexModValues
        )

    def __call__(self, outIndice: Indice) -> Indice:
        return [
            self.oneIndiceConverter(
                outIndice, linearCoeff, b, modValue, indexModCoeff, indexModValue
            )
            for (linearCoeff, b, modValue, indexModCoeff, indexModValue) in zip(
                self.linearCoeffs, self.bs, self.modValues, self.indexModCoeffs, self.indexModValues,
            )
        ]

    def setModValues(self, modValues: Shape) -> Any:
        self.modValues = (
            modValues * np.ones_like(np.array(self.bs))
            if isinstance(modValues, (int, float))
            else modValues
        )
        return self

    def __repr__(self, out: str = "y"):
        reprs = []
        for (linearCoeff, b, modValue, indexModCoeff, indexModValue) in zip(
                self.linearCoeffs, self.bs, self.modValues, self.indexModCoeffs, self.indexModValues,
        ):
            linearItem = " + ".join(
                [
                    pp("|_{1:flag}*x{0:.0f}_|", ai).format(i, ai)
                    for (i, ai) in enumerate(linearCoeff)
                    if ai != 0
                ]
            )
            modItem = " + ".join(
                [
                    pp("|_{1:flag}*mod(x{0:d}, {2:.0f})_|", bi).format(i, bi, ci)
                    for (i, (bi, ci)) in enumerate(zip(indexModCoeff, indexModValue))
                    if bi != 0
                ]
            )
            bItem = pp("{:flag}", b).format(b) if b != 0 else ""
            repr0 = (
                " + ".join([x for x in (linearItem, modItem, bItem) if x != ""])
                    .replace("+-", "-")
                    .replace("+ |_-", "- |_")
                    .replace("|_-", "-|_")
                    .replace("1*", "")
            )
            if repr0 == "":
                repr0 = "0"
            else:
                repr0 = (
                    "mod({}, {:.0f})".format(repr0, modValue) if modValue > 1 else repr0
                )
            reprs.append(repr0)
        return "\n".join(
            ["{}{:d} = {}".format(out, i, reprs[i]) for i in range(len(reprs))]
        )

    @staticmethod
    def oneIndiceConverter(
            outIndice: Indice, linearCoeff, b, modValue, indexModCoeff, indexModValue
    ) -> int:
        if modValue == 1:
            return 0
        linearSum = intDot(outIndice, linearCoeff)
        modSum = intDot(
            [x % m for (x, m) in zip(outIndice, indexModValue)], indexModCoeff
        )
        res = linearSum + modSum + b
        return int(res % modValue) if modValue > 1 else int(res)

    def hasLinearCoeffs(self, i: int) -> bool:
        return np.any(np.array(self.linearCoeffs)[:, i] != 0)


class UnitIndexConverter(IndexConverter):
    def __init__(self, outLen: int, inLen: int):
        super().__init__(np.zeros((inLen, outLen)).tolist(), [0] * inLen)
        self.inLen = inLen
        self.outLen = outLen

    def __call__(self, outIndex: Indice) -> Indice:
        return tuple([0] * self.inLen)


class ZeroIndexConverter(IndexConverter):
    def __init__(self):
        super().__init__([[]], [])

    def __call__(self, outIndex: Indice) -> Indice:
        return ()

    def __repr__(self, out: str = "y") -> str:
        return "{} = _".format(out)

    def hasLinearCoeffs(self, i: int) -> bool:
        return False
