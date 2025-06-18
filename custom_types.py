from typing import Callable, List, Any, Dict, Optional, TypeVar, Tuple

import numpy as np

FuncType = Callable[[float, float, Dict[str, Any]], float]
HistoryType = List[np.ndarray] 