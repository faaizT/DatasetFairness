import numpy as np
from scipy.stats import norm
import scipy.optimize as opt


def compute_diff_random_permutation(a1, a2):
    # Make a copy of a1 and a2 because np.random.shuffle operates in-place
    a1_copy = np.copy(a1)
    a2_copy = np.copy(a2)

    # Shuffle both arrays
    np.random.shuffle(a1_copy)
    np.random.shuffle(a2_copy)

    # Truncate the longer array to the size of the shorter one
    if len(a1) > len(a2):
        a1_copy = a1_copy[: len(a2)]
    elif len(a1) < len(a2):
        a2_copy = a2_copy[: len(a1)]

    # Compute and return the difference between the two arrays
    return a1_copy - a2_copy


def get_bootstrap_bounds(data, alpha, num_bootstrap_samples=100):
    """
    Compute bootstrap confidence bounds for given data and coverage level.
    
    Args:
        data (list or numpy array): The data for which to compute the bounds.
        alpha (float): Desired coverage level. Must be between 0 and 1.
        num_bootstrap_samples (int): Number of bootstrap samples to generate.
    
    Returns:
        tuple: lower bound, mean and upper bound of the data according to bootstrap method.
    """
    data = np.array(data)
    n = len(data)
    
    # Initialize an array to store bootstrap means
    bootstrap_means = np.zeros(num_bootstrap_samples)
    
    # Generate bootstrap samples and compute their means
    for i in range(num_bootstrap_samples):
        bootstrap_sample = np.random.choice(data, size=n, replace=True)
        bootstrap_means[i] = np.mean(bootstrap_sample)
    
    # Compute empirical mean of the data
    empirical_mean = np.mean(data)
    
    # Compute lower and upper percentiles for confidence bounds
    lower_bound = np.percentile(bootstrap_means, (alpha / 2) * 100)
    upper_bound = np.percentile(bootstrap_means, (1 - alpha / 2) * 100)
    
    return lower_bound, empirical_mean, upper_bound

def get_hoeffding_bounds(data, alpha):
    """
    Compute Hoeffding's bounds for given data and coverage level.
    Args:
        data (list or numpy array): The data for which to compute the bounds.
        alpha (float): Desired coverage level. Must be between 0 and 1.

    Returns:
        tuple: lower bound, mean and upper bound of the data according to Hoeffding's inequality.
    """
    data = np.array(data)
    mean = np.mean(data)
    n = len(data)

    bound = np.sqrt(np.log(2 / alpha) / (2 * n))
    lb = mean - bound
    ub = mean + bound

    return lb, mean, ub


def get_bernstein_bounds(data, alpha, variance=1 / 4):
    """
    Compute Bernstein's bounds for given data and coverage level.
    Args:
        data (list or numpy array): The data for which to compute the bounds.
        alpha (float): Desired coverage level. Must be between 0 and 1.

    Returns:
        tuple: lower bound, mean and upper bound of the data according to Bernstein's inequality.
    """
    data = np.array(data)
    mean = np.mean(data)
    n = len(data)
    variance = np.var(data, ddof=0)  # population variance
    B = 1  # as data lies in [0, 1]

    t = np.sqrt((2 * variance * np.log(2 / alpha)) / n) + (B * np.log(2 / alpha)) / (
        3 * n
    )
    lb = mean - t
    ub = mean + t

    return lb, mean, ub


def get_bentkus_bounds(data, alpha):
    """
    Compute Bentkus bound for given data and confidence level.

    Args:
        data (list or numpy array): The data for which to compute the bounds.
        alpha (float): Desired confidence level. Must be between 0 and 1.

    Returns:
        tuple: lower and upper bound according to Bentkus inequality.
    """
    data = np.array(data)
    mean = np.mean(data)
    n = len(data)

    # solve for t using scipy's optimization
    def f(t):
        return t - np.log(1 + np.exp(t)) - np.log(alpha) / n

    t_opt = opt.root_scalar(f, method="brentq", bracket=[0, 50]).root

    # calculate bound and confidence intervals
    bound = np.log(1 + np.exp(t_opt)) / n
    lb = mean - bound
    ub = mean + bound

    return lb, mean, ub


def get_asymptotic_bounds(data, alpha):
    """
    Compute Asymptotic CIs for given data and coverage level.
    Args:
        data (list or numpy array): The data for which to compute the bounds.
        alpha (float): Desired coverage level. Must be between 0 and 1.

    Returns:
        tuple: lower bound, mean and upper bound of the data according to CLT.
    """
    data = np.array(data)
    mean = np.mean(data)
    n = len(data)
    std_error = np.std(data, ddof=1) / np.sqrt(n)

    z_score = norm.ppf(1 - alpha / 2)  # two-tailed test

    lower_bound = mean - z_score * std_error
    upper_bound = mean + z_score * std_error

    return lower_bound, mean, upper_bound


def get_bootstrap_bounds_dp(dp_males_arr, dp_females_arr, alpha, num_bootstrap_samples=100):
    # Initialize an array to store bootstrap means
    bootstrap_means = np.zeros(num_bootstrap_samples)
    
    # Generate bootstrap samples and compute their means
    for i in range(num_bootstrap_samples):
        bootstrap_sample_dp_males = np.random.choice(dp_males_arr, size=len(dp_males_arr), replace=True)
        bootstrap_sample_dp_females = np.random.choice(dp_females_arr, size=len(dp_females_arr), replace=True)
        bootstrap_means[i] = np.mean(bootstrap_sample_dp_males) - np.mean(bootstrap_sample_dp_females)
    
    # Compute empirical mean of the data
    empirical_mean = np.abs(np.mean(bootstrap_means))
    
    # Compute lower and upper percentiles for confidence bounds
    lower_bound = np.percentile(np.abs(bootstrap_means), (alpha / 2) * 100)
    upper_bound = np.percentile(np.abs(bootstrap_means), (1 - alpha / 2) * 100)
    
    return lower_bound, empirical_mean, upper_bound


def get_fairness_bounds(
    accuracies, dp_males_arr, dp_females_arr, alpha, bound_type="hoeffdings"
):
    if bound_type.startswith("hoeffdings"):
        get_bounds = get_hoeffding_bounds
    elif bound_type.startswith("asymptotic"):
        get_bounds = get_asymptotic_bounds
    elif bound_type.startswith("bernstein"):
        get_bounds = get_bernstein_bounds
    elif bound_type.startswith("bentkus"):
        get_bounds = get_bentkus_bounds 
    elif bound_type.startswith("bootstrap"):
        get_bounds = get_bootstrap_bounds
    else:
        raise ValueError(
            "bound_type must be one of ['hoeffdings', 'asymptotic', 'bernstein', 'bentkus']"
        )
    # Compute Hoeffding's bounds
    if bound_type.startswith("bentkus"):
        accuracy_lb, accuracy, accuracy_ub = get_hoeffding_bounds(accuracies, alpha / 2)
    else:
        accuracy_lb, accuracy, accuracy_ub = get_bounds(accuracies, alpha / 2)
    accuracy_lb = np.clip(accuracy_lb, 0, 1)
    accuracy_ub = np.clip(accuracy_ub, 0, 1)

    if bound_type.startswith("bootstrap"):
        dp_lb, dp, dp_ub = get_bootstrap_bounds_dp(dp_males_arr, dp_females_arr, alpha)
    elif "diff" in bound_type:
        dp_male_minus_female_arr = compute_diff_random_permutation(dp_males_arr, dp_females_arr)
        dp_lb, dp, dp_ub = get_bounds(dp_male_minus_female_arr, alpha / 2)
    else:
        dp_lb_males, dp_males, dp_ub_males = get_bounds(dp_males_arr, alpha / 4)
        dp_lb_females, dp_females, dp_ub_females = get_bounds(dp_females_arr, alpha / 4)
        dp = abs(dp_males - dp_females)

        dp_lb = dp_lb_males - dp_ub_females
        dp_ub = dp_ub_males - dp_lb_females

    # if same sign, [min(|l|, |u|), max(|l|,|u|)] is a valid interval for | dp |
    if dp_lb * dp_ub > 0:
        dp_lb, dp_ub = min(abs(dp_lb), abs(dp_ub)), max(abs(dp_lb), abs(dp_ub))
    else:
        # otherwise, [0, max(|l|,|u|)] is a valid interval for | dp |
        dp_lb, dp_ub = 0, max(abs(dp_lb), abs(dp_ub))
    dp_ub = np.clip(dp_ub, 0, 1)
    dp_lb = np.clip(dp_lb, 0, 1)

    return accuracy, accuracy_lb, accuracy_ub, dp, dp_lb, dp_ub


def get_ci1_over_ci2_series(ci1, ci2):
    """
    Compute the division of two confidence intervals represented as tuples of pd.Series.

    Args:
        ci1, ci2: Two confidence intervals of the form (a, b) and (c, d) where a, b, c, d are pd.Series.

    Returns:
        ci: Interval resulting from the division of ci1 and ci2.
    """

    a, b = ci1
    c, d = ci2

    # Return the new interval
    return (a / d, b / c)  # flip the denominator to account for inversion


def get_ci1_plus_ci2_series(ci1, ci2):
    """
    Compute the addition of two confidence intervals represented as tuples of pd.Series.

    Args:
        ci1, ci2: Two confidence intervals of the form (a, b) and (c, d) where a, b, c, d are pd.Series.

    Returns:
        ci: Interval resulting from the addition of ci1 and ci2.
    """

    a, b = ci1
    c, d = ci2

    # Return the new interval
    lb = np.clip(a + c, 0, 1)
    ub = np.clip(b + d, 0, 1)
    return (lb, ub)
