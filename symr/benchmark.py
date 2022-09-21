import argparse

import numpy as np
import sympy as sp

from symr.fake_sr import RandomSR, PolySR, FourierSR
from symr.nsrts_wrapper import NSRTSWrapper
from symr.symgpt_wrapper import SymGPTWrapper

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
            "PolySR":   PolySR,\
            "NSRTS":   NSRTSWrapper,\
            "SymGPT":   SymGPTWrapper\
            }
    metric_dict = {\
            "r2": compute_r2,\
            "tree_distance": compute_tree_distance,\
            "exact": compute_exact_equivalence,\
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
    else:
        write_csv = 0 

    if "sample_size" in kwargs.keys():
        sample_size = kwargs["sample_size"]
    else:
        sample_size = 20

    if "input_dataset" in kwargs.keys():
        input_dataset = kwargs["input_dataset"]
    else:
        input_dataset = None
        

    log_lines = []
    msg = "method, use_bfgs, expression, predicted, trial, r2, tree_distance, "\
            "exact, r2_cuttoff, r2_over_95, r2_over_99, r2_over_999, "\
            "isclose\n"
    log_lines.append(msg)

    # load benchmark with default filepath
    benchmark = load_benchmark(input_dataset)

    expressions = [elem.split(",")[0] for elem in benchmark[1:]]
    supports = [elem.split(",")[1] for elem in benchmark[1:]]
    variables = [elem.split(",")[3] for elem in benchmark[1:]]

    if type(sr_methods) is str:
        sr_methods = [sr_methods]

    for method in sr_methods:
        for expr_index, expression in enumerate(expressions):
            for trial in range(trials):
                # implement k-fold validation here, TODO
                my_inputs = {}
                model = method_dict[method](use_bfgs=use_bfgs, \
                        input_variables=variables[expr_index])
                for v_index, variable in \
                        enumerate(variables[expr_index][1:].split(" ")):
                    # generate random (uniform) samples for each variable
                    # within support range
                    
                    low = float(\
                            supports[expr_index][1:].split(" ")[v_index+0])
                    high = float(\
                            supports[expr_index][1:].split(" ")[v_index+1])
                    my_stretch = high - low

                    my_inputs[variable] = np.random.rand(sample_size,1)
                    my_inputs[variable] = my_stretch \
                            * my_inputs[variable] - low
                        
                lambda_variables = ",".join(variables[expr_index][1:].split(" "))

                # sp.lambdify does not currently recognize ln
                # but default base for log is e, 
                expression = expression.replace("ln","log")
                target_function = sp.lambdify(\
                        lambda_variables, \
                        expr=expression)

                y_target = target_function(**my_inputs)

                predicted_expression = model( \
                        target=y_target, \
                        **my_inputs)

                predicted_function = sp.lambdify(\
                        lambda_variables, \
                        expr=predicted_expression)

                y_predicted = predicted_function(\
                        **my_inputs)

                scores = []
                for metric in metric_dict.keys():
                    
                    if metric in ["tree_distance", "exact"]:
                        scores.append(metric_dict[metric](expression, predicted_expression))
                    else:
                        if metric in metrics:
                            metric_function = metric_dict[metric]
                        else:
                            metric_function = lambda **kwargs: "None"

                        scores.append(metric_function(targets=y_target, predictions=y_predicted))

                msg += f"{method}, {use_bfgs}, {expression}, {predicted_expression}, {trial}"

                for metric, score in zip(metric_dict.keys(), scores):


                    msg += f", {score}"

                msg += "\n"

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
    parser.add_argument("-i", "--input_dataset", type=str, default="data/nguyen.csv",\
            help="benchmark csv")
    parser.add_argument("-k", "--k-folds", type=int, default=4, \
            help="number of cross-validation splits to use")
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
    parser.add_argument("-o", "--output_filename", type=str, default="results/temp.csv",\
            help="filename to save csv")
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


    
    evaluate(**kwargs)
