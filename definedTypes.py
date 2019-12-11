from typing import Tuple, Dict, Sequence, Callable

from numpy import ndarray

Indice = Sequence[int]
Shape = Sequence[int]
LabelInIndice = Tuple[int, Indice]
InIndex = Dict[str, Indice]
OutIndex = Dict[Indice, str]
IndexMap = Dict[Indice, Indice]
ArrayConverter = Callable[
    [Sequence[ndarray], Shape, Shape, Dict[str, Callable]], ndarray
]

