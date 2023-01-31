########################################################################################
#
#
# This file implements a generic function to compute the equal error rate.
#
# Author(s): Nik Vaessen
########################################################################################

from typing import List

import numpy as np

from pyllr.pav_rocch import PAV, ROCCH

########################################################################################
# helper methods for both measures


def _verify_correct_scores(
    groundtruth_scores: List[int], predicted_scores: List[float]
):
    if len(groundtruth_scores) != len(predicted_scores):
        raise ValueError(
            f"length of input lists should match, while"
            f" groundtruth_scores={len(groundtruth_scores)} and"
            f" predicted_scores={len(predicted_scores)}"
        )
    if not all(np.isin(groundtruth_scores, [0, 1])):
        raise ValueError(
            f"groundtruth values should be either 0 and 1, while "
            f"they are actually one of {np.unique(groundtruth_scores)}"
        )

########################################################################################
# EER (equal-error-rate)


def calculate_eer(
    groundtruth_scores: List[int], predicted_scores: List[float]
) -> float:
    """
    Calculate the equal error rate between a list of groundtruth pos/neg scores
    and a list of predicted pos/neg scores.

    Positive ground truth scores should be 1, and negative scores should be 0.

    :param groundtruth_scores: a list of groundtruth integer values (either 0 or 1)
    :param predicted_scores: a list of prediction float values
    :return: a float value containing the equal error rate
    """
    _verify_correct_scores(groundtruth_scores, predicted_scores)

    scores = np.asarray(predicted_scores, dtype=float)
    labels = np.asarray(groundtruth_scores, dtype=float)
    pav = PAV(scores, labels)
    rocch = ROCCH(pav)

    eer = rocch.EER()

    return float(eer)
