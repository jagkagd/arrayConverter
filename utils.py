from collections import deque
from math import floor
import numpy as np
from definedTypes import *
from typing import List, Any


def intDot(a, b) -> float:
    return sum([floor(a0 * b0) for (a0, b0) in zip(a, b)])


def allSame(items: Sequence) -> bool:
    return all([x == items[0] for x in items])


def isint(val: float) -> str:
    if isinstance(val, int):
        return "d"
    if isinstance(val, float) and val.is_integer():
        return "d"
    return "f"


def pp(fstr: str, val: float) -> str:
    return fstr.replace("flag", {"d": ".0f", "f": ".2f"}[isint(val)])


class InnerPtsDeque(deque):
    def __init__(self):
        super().__init__()

    def __contains__(self, item: Indice) -> bool:
        return super().__contains__(item) or any([i < 0 for i in item])


class InnerPts(list):
    def __init__(self, pts: Indice):
        super().__init__(pts)
        self.pts = pts
        self.len = len(self.pts)

    def previousPts(self) -> List[Any]:
        return [
            InnerPts(tuple(a))
            for a in np.tile(np.array(self.pts), (self.len, 1)) - np.eye(self.len)
        ]

    def nextPts(self) -> List[Any]:
        return [
            InnerPts(tuple(a))
            for a in np.tile(np.array(self.pts), (self.len, 1)) + np.eye(self.len)
        ]

    def toShape(self) -> Shape:
        return tuple([int(i+1) for i in self.pts])
