import argparse

import os 
import numpy as np
import pandas as pd


def main(**kwargs):

    """
    Do a grid search to optimize context for a SR method in kwargs['sr_methods']
    """
    if "use_clip" in kwargs.keys():
        use_clip = kwargs["use_clip"]
    else:
        use_clip = 0

    # set parameters
    if "input_dataset" in kwargs.keys():
        input_dataset = kwargs["input_dataset"]
    else:
        input_dataset = os.path.join("data", "nguyen_univariate.csv")

    # focus on a single metric
    if "metrics" in kwargs.keys():
        metrics = kwargs["metrics"]
    else:
        metrics = ["in_r2"]
    metric_of_choice = metrics[0]

    # focus on a single method
    if "sr_methods" in kwargs.keys():
        method = kwargs["sr_methods"][0]
    else:
        method = ["NSRTS"][0]

    # static context parameters
    parameters = {\
                 "-e": [0.25],\
                 "-k": [3],
                 "-r": [42],\
                 "-t": [6],\
                  "-w": [1]\
                  }

    # variable context parameters
    parameters["-b"] = kwargs["use_bfgs"] 
    parameters["-d"] = kwargs["degree"]
    parameters["-z"] = kwargs["sample_size"]
    parameters["-n"] = kwargs["beam_width"]

    if type(parameters["-b"]) is not list:
        parameters["-b"] = [parameters["-b"]] 
    if type(parameters["-d"]) is not list:
        parameters["-d"] = [parameters["-d"]] 
    if type(parameters["-z"]) is not list:
        parameters["-z"] = [parameters["-z"]]
    if type(parameters["-n"]) is not list:
        parameters["-n"] = [parameters["-n"]]

    results_start = f"results/{method}"

    np.random.seed(42)

    results_files = []
    run_parameters = []

    cmd_start = f"python -m symr.benchmark -s {method} -i {input_dataset} "

    total_options = np.sum([len(parameters[key]) for key in parameters.keys()])

    number_runs = 1 #total_options * 2


    for attempt in range(number_runs):
        results_filename = ""
        cmd = ""
        for key in parameters:
            my_parameter = np.random.choice(parameters[key],\
                    p=[1/len(parameters[key]) for elem in parameters[key]])

            results_filename += f"_{key[1:]}{str(my_parameter).replace('.','')}"
            cmd += f"{key} {str(my_parameter)} "

        results_filename = results_start + results_filename + ".csv"
        run_parameters.append(cmd + "")
        cmd = cmd_start + cmd + f" -o {results_filename}"
        print(cmd)
        results_files.append(results_filename)
        os.system(cmd)

    my_scores = []
    boolean_metrics = [\
            " in_isclose", " in_r2_over_95", \
            " in_r2_over_99", " in_r2_over_999",\
            " ex_isclose", " ex_r2_over_95", \
            " ex_r2_over_99", " ex_r2_over_999",\
            ]

    for filepath in results_files:

        df = pd.read_csv(filepath)

        my_metric_raw = df.loc[df["method"] == method][metric_of_choice]
        my_success = (" False" == df.loc[df["method"] == method]["failed"]).to_numpy(dtype=float)

        if metric_of_choice in boolean_metrics:
            my_metric = (" True" == my_metric_raw[0 < my_success]).to_numpy(dtype=float)
        else:
            if use_clip:
                my_metric = np.clip(my_metric_raw[0 < my_success].to_numpy(dtype=float), -1., 1.0)
            else:
                my_metric = my_metric_raw[0. < my_success].to_numpy(dtype=float)

        my_scores.append(np.mean(my_metric))

    import pdb; pdb.set_trace()
    sort_indices = list(np.argsort(my_scores))
    sort_indices.reverse()
    print(f"best run was {np.array(results_files)[sort_indices[0]]}")
    print(f"best parameters were {run_parameters[sort_indices[0]]}")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser("do a grid search (or a barrel roll)")

    parser.add_argument("-b", "--use_bfgs", type=int, default=1, nargs="+",\
            help="use BFGS for post-inference optimization")
    parser.add_argument("-c", "--use_clip", type=int, default=0, nargs="+",\
            help="use np.clip(..., -1. 1.) before calculating stats from r2")
    parser.add_argument("-d", "--degree", type=int, default=10, nargs="+",\
            help="number of terms to use for fake sr methods PolySR, FourierSR, and RandomSR")
    parser.add_argument("-i", "--input_dataset", type=str, default="data/nguyen_univariate.csv",\
            help="benchmark csv")
    parser.add_argument("-m", "--metrics", type=str, nargs="+",\
        default=["exact", "tree_distance", "nmae", "nmse","r2",\
            "r2_over_95", "r2_over_99", "r2_over_999",\
            "r2_cutoff", "isclose"],\
        help="metrics to include in benchmarks,"
            " can specify multiple metrics"\
            "options: "\
            "   nmse, "\
            "   nmae, "\
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
    parser.add_argument("-n", "--beam_width", type=int, default=1, nargs="+",\
            help="number of equations, aka beam_width")
    parser.add_argument("-s", "--sr_methods", type=str, default="NSRTS", nargs="+",\
            help="method to perform grid search and optimize for")
    parser.add_argument("-z", "--sample_size", type=int, default=100, nargs="+",\
            help="number of samples per dataset")

    args = parser.parse_args()

    kwargs = dict(args._get_kwargs())

    main(**kwargs)
