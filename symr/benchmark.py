import argparse

import numpy as np

from symr.fake_sr import RandomSR, PolySR, FourierSR

from symr.metrics import compute_r2, compute_isclose_accuracy,\
        compute_r2_over_threshold, compute_relative_error,\
        compute_r2_truncated, compute_tree_distance,\
        compute_exact_equivalence, get_r2_threshold_function

from symr.helpers import load_benchmark

def evaluate(**kwargs):
    """
    function for evaluating SR methods
    """
    method_dict = {\
            "RandomSR": RandomSR,\
            "FourierSR": FourierSR,\
            "PolySR":   PolySR\
            }
    metric_dict = {\
            "r2": compute_r2,\
            "tree_distance": compute_tree_distance,\
            "exact": compute_exact_equivalence,\
            "r2_over_95": get_r2_threshold_function(threshold=0.95),\
            "r2_over_99": get_r2_threshold_function(threshold=0.99),\
            "r2_over_999": get_r2_threshold_function(threshold=0.999),\
            "r2_cutoff": compute_r2_truncated,\
            "isclose": compute_isclose_accuracy\
            }

    if "metrics" in kwargs.keys():
        metrics = kwargs["metrics"]
    else: 
        print("warning, no metrics specified, using default r^2")
        metrics = ["r2"]

    if "sr_methods" in kwargs.keys():
        sr_methods = kwargs["sr_methods"]
    else:
        print("warning, no method specified, using default RandomSR")
        sr_methods = ["RandomSR"]

    if "trials" in kwargs.keys():
        trials = kwargs["trials"]
    else:
        print("warning, number of trials not specified using default 1")
        trials = 1

    # TODO: sample size should be user-selectable
    sample_size = 200

    log_lines = []
    msg = "method, expression, trial, r2, tree_distance, "\
            "exact, r2_over_95, r2_over_99, r2_over_999, "\
            "isclose\n"
    print(msg)
    log_lines.append(msg)

    # load benchmark with default filepath
    benchmark = load_benchmark()

    expressions = [elem.split(",")[0] for elem in benchmark[1:]]
    supports = [elem.split(",")[1] for elem in benchmark[1:]]
    variables = [elem.split(",")[3] for elem in benchmark[1:]]

    for method in sr_methods:
        for expr_index, expression in enumerate(expressions):
            for metric in metrics:
                for trial in range(trials):
                    # implement k-fold validation here, TODO
                    my_inputs = {}
                    for v_index, variable in \
                            enumerate(variables[expr_index][1:].split(" ")):
                        
                        low = float(\
                                supports[expr_index][1:].split(" ")[v_index+0])
                        high = float(\
                                supports[expr_index][1:].split(" ")[v_index+1])
                        my_stretch = high - low

                        my_inputs[variable] = np.random.rand(sample_size,1)
                        my_inputs[variable] = my_stretch \
                                * my_inputs[variable] - low

                    pass

                
                
                
                
    
    return 0

if __name__ == "__main__": #pragma: no cover
    # this scope will be tested, but in a way that doesn't
    # get tracked by coverage (hence to no cover pragma)
    # i.e. it's called by subprocess.check_output

    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--metrics", type=str, nargs="+",\
        default=["exact", "tree_distance", "r2",\
            "r2_over_95", "r2_over_99", "r2_over_999",\
            "r2_cutoff", "isclose"],\
        help="metrics to include in benchmarks,"
            " can specify multiple metrics"\
            "options: "\
            "   r2, "\
            "   tree_distance, "\
            "   exact, "\
            "   r2_over_95, "\
            "   r2_over_99, "\
            "   r2_over_999, "\
            "   r2_cutoff, "\
            "   isclose"\
            "default is to assess all metrics"
        )
    parser.add_argument("-k", "--k-folds", type=int, default=4, \
            help="number of cross-validation splits to use")
    parser.add_argument("-s", "--sr-methods", type=str, nargs="+",\
            default=["RandomSR"],\
            help="which SR methods to benchmark. "
                "Default is RandomSR, other options include: "\
                " FourierSR, PolySR")
    parser.add_argument("-t", "--trials", type=int, default=1,\
            help="number of trials per expression during testing")

    args = parser.parse_args()

    kwargs = dict(args._get_kwargs())


    print(kwargs)
    print("all ok")
