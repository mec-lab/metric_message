import argparse

def evaluate(**kwargs):

    return 0

if __name__ == "__main__":

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
    parser.add_argument("-s", "--sr-method", type=str, nargs="+",\
            default=["RandomSR"],\
            help="which SR methods to benchmark. "
                "Default is RandomSR, other options include: "\
                " FourierSR, PolySR")

    args = parser.parse_args()

    kwargs = dict(args._get_kwargs())

    print(kwargs)
    print("all ok")
