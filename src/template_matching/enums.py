from enum import Enum

import cv2


class TemplateMatchingMethod(Enum):
    CCOEF = cv2.TM_CCOEFF
    CCOEF_NORM = cv2.TM_CCOEFF_NORMED
    CCORR = cv2.TM_CCORR
    CCORR_NORM = cv2.TM_CCORR_NORMED
    SQDIFF = cv2.TM_SQDIFF
    SQDIFF_NORM = cv2.TM_SQDIFF_NORMED
