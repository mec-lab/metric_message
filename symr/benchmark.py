import argparse
import subprocess
import sys
import os

import numpy as np
import sympy as sp
import torch

from symr.fake_sr import RandomSR, PolySR, FourierSR
from symr.nsrts_wrapper import NSRTSWrapper
from symr.symgpt_wrapper import SymGPTWrapper
from symr.symformer_wrapper import SymformerWrapper
from symr.pysr_wrapper import PySRWrapper

from symr.metrics import compute_r2, compute_isclose_accuracy,\
        compute_r2_over_threshold, compute_relative_error,\
        compute_r2_truncated, compute_tree_distance,\
        compute_relative_error, compute_relative_squared_error,\
        compute_exact_equivalence, get_r2_threshold_function

from symr.helpers import load_benchmark

def evaluate(**kwargs):
    """
    function for evaluating SR methods
    """
    method_dict = {\
            "RandomSR": RandomSR,\
            "FourierSR": FourierSR,\
            "PolySR":   PolySR,\
            "PySR":   PySRWrapper,\
            "NSRTS":   NSRTSWrapper,\
            "SymGPT":   SymGPTWrapper,\
            "Symformer":   SymformerWrapper\
            }
    metric_dict = {\
            "tree_distance": compute_tree_distance,\
            "exact": compute_exact_equivalence,\
            "nmae": compute_relative_error,\
            "nmse": compute_relative_squared_error,\
            "r2": compute_r2,\
            "r2_cutoff": compute_r2_truncated,\
            "r2_over_95": get_r2_threshold_function(threshold=0.95),\
            "r2_over_99": get_r2_threshold_function(threshold=0.99),\
            "r2_over_999": get_r2_threshold_function(threshold=0.999),\
            "isclose": compute_isclose_accuracy\
            }

    if "use_bfgs" in kwargs.keys():
        use_bfgs = kwargs["use_bfgs"]
    else: 
        use_bfgs = 0 

    if "degree" in kwargs.keys():
        degree = kwargs["degree"]
    else:
        degree = 10 

    if "input_dataset" in kwargs.keys():
        input_dataset = kwargs["input_dataset"]
    else:
        input_dataset = None

    if "k_folds" in kwargs.keys():
        k_folds = kwargs["k_folds"]
    else:
        k_folds = 1

    if "metrics" in kwargs.keys():
        metrics = kwargs["metrics"]
    else: 
        metrics = ["r2"]

    if "sr_methods" in kwargs.keys():
        sr_methods = kwargs["sr_methods"]
    else:
        sr_methods = ["PolySR"]

    if "trials" in kwargs.keys():
        trials = kwargs["trials"]
    else:
        trials = 1

    if "write_csv" in kwargs.keys():
        write_csv = kwargs["write_csv"]
    else:
        write_csv = False 

    if "output_filename" in kwargs.keys() and write_csv:
        output_filename = kwargs["output_filename"]
        temp = os.path.splitext(output_filename)
        partial_filename = temp[0] + "_partial" + temp[1]
    else:
        write_csv = 0 

    if "sample_size" in kwargs.keys():
        sample_size = kwargs["sample_size"]
    else:
        sample_size = 20

        

    log_lines = []
    msg = "method, use_bfgs, expression, predicted, trial, k_fold, tree_distance, exact,"\
            "in_nmae, in_nmse,"\
            "in_r2, in_r2_cuttoff, in_r2_over_95, in_r2_over_99, in_r2_over_999, in_isclose,"\
            "ex_nmae, ex_nmse,"\
            "ex_r2, ex_r2_cuttoff, ex_r2_over_95, ex_r2_over_99, ex_r2_over_999, ex_isclose,"\
            "failed, time_elapsed, git_hash, entry_point\n"

    msg += "meta "
    msg += " , " * (msg.count(",")-2) 
    msg += ", " + kwargs["git_hash"] if "git_hash" in kwargs.keys() else "none"
    msg += ", " + kwargs["entry_point"] if "entry_point" in kwargs.keys() else "none"
    msg += "\n"

    if write_csv:
        with open(partial_filename, "w") as f:
            f.writelines(msg)

    log_lines.append(msg)

    # load benchmark with default filepath
    benchmark = load_benchmark(input_dataset)

    expressions = [elem.split(",")[0] for elem in benchmark[1:]]
    supports = [elem.split(",")[1] for elem in benchmark[1:]]
    variables = [elem.split(",")[3] for elem in benchmark[1:]]

    if type(sr_methods) is str:
        sr_methods = [sr_methods]

    for method in sr_methods:
        for trial in range(trials):
            if "random_seed" in kwargs.keys():
                np.random.seed(kwargs["random_seed"] * trial )
                torch.manual_seed(kwargs["random_seed"] * trial)
            else:
                # safety fallback
                np.random.seed(trial)
                torch.manual_seed(trial)
            for expr_index, expression in enumerate(expressions):
                for fold in range(k_folds):


                    # implement k-fold validation here, TODO

                    if "ex_proportion" in kwargs.keys():
                        # proportion of range to use for validation (extrapolation)
                        ex_proportion = kwargs["ex_proportion"]
                        # proportion of range to use for examples
                        in_proportion = 1.0 - ex_proportion
                    else:
                        ex_proportion = 1 / (k_folds+1)
                        in_proportion = 1.0 - ex_proportion

                    # example inputs
                    my_inputs = {}
                    # in-distribution validation inputs (interpolation)
                    id_val_inputs = {}
                    # ex-distribution validation inputs (extrapolation)
                    ed_val_inputs = {}


                    
                    for v_index, variable in \
                            enumerate(variables[expr_index][1:].split(" ")):
                        # generate random (uniform) samples for each variable
                        # within support range
                        
                        low = float(\
                                supports[expr_index][1:].split(" ")[v_index+0])
                        high = float(\
                                supports[expr_index][1:].split(" ")[v_index+1])

                        support_range = high - low

                        in_support_range = in_proportion * support_range
                        id_low = low + np.random.rand() * (support_range - in_support_range)
                        id_high = id_low + in_support_range

                        ex_support_range_0 = id_low - low
                        ex_support_range_1 = high - id_low
                        sample_size_0 = int(ex_support_range_0 / \
                                (ex_support_range_0+ex_support_range_1))
                        sample_size_1 = sample_size-sample_size_0

                        # example data, used for SR inference
                        my_inputs[variable] = np.random.uniform(low=id_low, high=id_high,\
                                size=(sample_size,1))

                        # in-distribution evaluation data
                        # different samples taken from the same range
                        id_val_inputs[variable] = np.random.uniform(low=id_low, high=id_high,\
                                size=(sample_size,1))
                        
                        # ex-distribution evaluation data
                        # different samples taken from outside example range
                        ed_val_0 = np.random.uniform(low=low, high=id_low,\
                                size=(sample_size_0,1))
                        ed_val_1 = np.random.uniform(low=id_high, high=high,\
                                size=(sample_size_1,1))
                        ed_val_inputs[variable] = np.append(ed_val_0, ed_val_1, \
                                axis=0)

                    lambda_variables = ",".join(variables[expr_index][1:].split(" "))

                    # sp.lambdify does not currently recognize ln
                    # but default base for log is e, 
                    expression = expression.replace("ln","log")
                    target_function = sp.lambdify(\
                            lambda_variables, \
                            expr=expression)

                    y_target = target_function(**my_inputs)
                    id_y_target = target_function(**id_val_inputs)
                    ed_y_target = target_function(**ed_val_inputs)

                    model = method_dict[method](use_bfgs=use_bfgs, \
                            input_variables=variables[expr_index], \
                            degree=degree)

                    predicted_expression, info = model( \
                            target=y_target, \
                            **my_inputs)

                    if "failed" in info.keys():
                        failed = info["failed"]
                    else:
                        failed = "n/a"

                    if "time_elapsed" in info.keys():
                        time_elapsed = info["time_elapsed"]
                    else:
                        time_elapsed = "n/a"

                    predicted_function = sp.lambdify(\
                            lambda_variables, \
                            expr=predicted_expression)

                    id_y_predicted = predicted_function(\
                            **id_val_inputs)
                    ed_y_predicted = predicted_function(\
                            **ed_val_inputs)

                    id_scores = []
                    ed_scores = []
                    partial_msg = ""
                    for metric in metric_dict.keys():
                        
                        if metric in ["tree_distance", "exact"]:
                            id_scores.append(metric_dict[metric](expression, predicted_expression))
                        else:
                            if metric in metrics:
                                metric_function = metric_dict[metric]
                            else:
                                metric_function = lambda **kwargs: "None"

                            try:
                                id_scores.append(metric_function(targets=id_y_target, predictions=id_y_predicted))
                                ed_scores.append(metric_function(targets=ed_y_target, predictions=ed_y_predicted))
                            except:
                                import pdb; pdb.set_trace()

                    partial_msg += f"{method}, {use_bfgs}, {expression}, {predicted_expression}, {trial}, {fold}"

                    for metric, score in zip(metric_dict.keys(), id_scores):

                        partial_msg += f", {score}"

                    for metric, score in zip(list(metric_dict.keys())[2:], ed_scores):

                        partial_msg += f", {score}"

                    partial_msg += f", {failed}, {time_elapsed}" 
                    
                    partial_msg += "\n"

                    if write_csv:
                        with open(partial_filename, "a") as f:
                            f.writelines(partial_msg)
                    else: 
                        print(partial_msg)

                    msg += partial_msg

    if write_csv:
        with open(output_filename, "w") as f:
            f.writelines(msg)
    else: 
        print(msg)

    return 0

if __name__ == "__main__": #pragma: no cover
    # this scope will be tested, but in a way that doesn't
    # get tracked by coverage (hence to no cover pragma)
    # i.e. it's called by subprocess.check_output

    parser = argparse.ArgumentParser()


    parser.add_argument("-b", "--use_bfgs", type=int, default=1,\
            help="use BFGS for post-inference optimization")
    parser.add_argument("-d", "--degree", type=int, default=10,\
            help="number of terms to use for fake sr methods PolySR, FourierSR, and RandomSR")
    parser.add_argument("-e", "--ex_proportion", type=float, default=0.5,\
            help="proportion of support range to use for extrapolation")
    parser.add_argument("-i", "--input_dataset", type=str, default="data/nguyen.csv",\
            help="benchmark csv")
    parser.add_argument("-k", "--k-folds", type=int, default=4, \
            help="number of cross-validation splits to use")
    parser.add_argument("-m", "--metrics", type=str, nargs="+",\
        default=["exact", "tree_distance", "nmae", "nmse","r2",\
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
    parser.add_argument("-o", "--output_filename", type=str, default="results/temp.csv",\
            help="filename to save csv")
    parser.add_argument("-r", "--random_seed", type=int, default=42,\
            help="seed for pseudorandom number generators, default 42")
    parser.add_argument("-s", "--sr-methods", type=str, nargs="+",\
            default=["RandomSR"],\
            help="which SR methods to benchmark. "\
                "Default is RandomSR, other options include: "\
                " FourierSR, PolySR, RandomSR, NSRTS, SymGPT")
    parser.add_argument("-t", "--trials", type=int, default=1,\
            help="number of trials per expression during testing")
    parser.add_argument("-w", "--write_csv", type=int, default=0,\
            help="1 - write csv, 2 - do not write csv, default 0")
    parser.add_argument("-z", "--sample-size", type=int, default=100, \
            help="number of samples per dataset")

    args = parser.parse_args()

    kwargs = dict(args._get_kwargs())

    # use subprocess to get the current git hash, store
    hash_command = ["git", "rev-parse", "--verify", "HEAD"]
    git_hash = subprocess.check_output(hash_command)

    # store the command-line call for this experiment
    entry_point = []
    entry_point.append(os.path.split(sys.argv[0])[1])
    args_list = sys.argv[1:]

    sorted_args = []
    for aa in range(0, len(args_list)):

        if "-" in args_list[aa]:
            sorted_args.append([args_list[aa]])
        else: 
            sorted_args[-1].append(args_list[aa])

    sorted_args.sort()
    entry_point = "python -m symr.benchmark "

    for elem in sorted_args:
        entry_point += " " + " ".join(elem)

    kwargs["entry_point"] = entry_point 
    kwargs["git_hash"] = git_hash.decode("utf8")[:-1]
    
    evaluate(**kwargs)
