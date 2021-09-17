import numpy as np
from typing import *

CV2Image = NewType('CV2_Image', np.ndarray)
OCRData = NewType('OCR_Data', Dict[str, Dict[int, List[Union[int, str]]]])
PipelineFunction = Callable[[CV2Image], CV2Image]