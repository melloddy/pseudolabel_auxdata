import numpy as np

# TODO add typings


def prob_ncm(scores, labels):
    """
    Converts Neural Network scores into Nonconformity Measures for CP.
    Assumes that scores are directly related to the probability of being active
    """
    return np.where(labels > 0, -scores, scores)


# p-Values calculation
def p_values(calibration_alphas, test_alphas, randomized=False):
    sorted_cal_alphas = sorted(calibration_alphas)
    if randomized:
        # for each test alpha, tieBreaker is the (number of calibration alphas with the same value)*(uniform RV
        # between 0 and 1)
        tie_counts = np.searchsorted(
            sorted_cal_alphas, test_alphas, side="right"
        ) - np.searchsorted(sorted_cal_alphas, test_alphas)
        tie_breaker = (
            np.random.uniform(size=len(np.atleast_1d(test_alphas))) * tie_counts
        )
        return (
            len(calibration_alphas)
            - (
                np.searchsorted(sorted_cal_alphas, test_alphas, side="right")
                - tie_breaker
            )
            + 1
        ) / (len(calibration_alphas) + 1)
    else:
        return (
            len(calibration_alphas)
            - np.searchsorted(sorted_cal_alphas, test_alphas)
            + 1
        ) / (len(calibration_alphas) + 1)


# Mondrian Inductive Conformal Predictor
def micp(
    calibration_alphas,
    calibration_labels,
    test_alphas_0,
    test_alphas_1,
    randomized=False,
):
    """
    Mondrian Inductive Conformal Predictor
    Parameters:
    calibration_alphas: 1d array of Nonconformity Measures for the calibration examples
    calibration_labels: 1d array of labels for the calibration examples - ideally 0/1 or -1/+1,
                        but negative/positive values also accepted
    test_alpha_0: 1d array of NCMs for the test examples, assuming 0 as label
    test_alpha_1: 1d array of NCMs for the test examples, assuming 1 as label
    Returns:
    p0,p1 : pair of arrays containing the p-values for label 0 and label 1
    """
    if not len(calibration_labels) == len(calibration_alphas):
        raise ValueError(
            "calibration_labels and calibration alphas must have the same size"
        )

    if not len(np.atleast_1d(test_alphas_0)) == len(np.atleast_1d(test_alphas_1)):
        raise ValueError("test_alphas_0 and test_alphas_1 must have the same size")

    p_0 = p_values(
        calibration_alphas[calibration_labels <= 0], test_alphas_0, randomized
    )
    p_1 = p_values(
        calibration_alphas[calibration_labels > 0], test_alphas_1, randomized
    )
    return p_0, p_1


# function to predict label from p0 and p1
def cp_label_predictor(p0, p1, eps):
    # Active: p1 > ϵ and p0 ≤ ϵ
    # Inactive: p0 > ϵ and p1 ≤ ϵ
    # Uncertain (Both): p1 > ϵ and p0 > ϵ
    # Empty (None): p1 ≤ ϵ and p0 ≤ ϵ
    if p1 > eps >= p0:
        return 1
    elif p0 > eps >= p1:
        return 0
    elif p0 > eps and p1 > eps:
        return "uncertain both"
    elif p0 <= eps and p1 <= eps:
        # return 'empty'
        # it should actually return 'empty', but to avoid a confusion for people
        return "uncertain none"
