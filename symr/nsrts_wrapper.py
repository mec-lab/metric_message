import os

import numpy as np

from sympy import lambdify
import sympy as sp

from symr.wrappers import BaseWrapper

import nesymres
from nesymres.architectures.model import Model
from nesymres.utils import load_metadata_hdf5
from nesymres.dclasses import FitParams, NNEquation, BFGSParams
import omegaconf

from functools import partial

import json

if "NSRTS_DIR" in os.environ.keys():
    NSRTS_DIR = os.environ["NSRTS_DIR"]
else:
    NSRTS_DIR = "nsrts" 

class NSRTSWrapper(BaseWrapper):

    def __init__(self, **kwargs):

        if "use_bfgs" in kwargs.keys():
            self.use_bfgs = kwargs["use_bfgs"]
        else:
            self.use_bfgs = False

        self.my_beam_width = 1
        self.initialize_model()
        self.load_parameters()


    def parse_filter(self, expression, variables=["x"]):
        """
        Sometimes cleaning the string isn't enough to yield an actual expression.

        If the expression string is malformed, return f(x) = 0.0 instead
        """
        try: 
            # SymGPT currently only handles one variable: x1
            my_fn = sp.lambdify(variables, expr=expression)
            my_inputs = {key:np.random.rand(3,1) for key in variables}
            _ = my_fn(**my_inputs)

            # return expression if it was successfuly parsed into a function 
            return expression

        except:
            return "+".join([f"0.0 * {my_var}" \
                    for my_var in variables])

    def __call__(self, target, **kwargs):
        
            
        fitfunc = partial(self.model.fitfunc, cfg_params=self.params_fit)

        x = None

        for key in kwargs.keys():
            if x is None:
                x = kwargs[key].reshape(-1,1)
            elif x.shape[-1] < 3:
                my_x = kwargs[key]
                if len(my_x.shape) == 2:
                    x = np.append(x, my_x, axis=-1) 
                else:
                    x = np.append(x, my_x[:,None], axis=-1) 
        
        if self.use_bfgs:
            output = fitfunc(x, target.squeeze())
            try: 
                expression = output["best_bfgs_preds"][0]
            except:
                expression = "+".join([f"0.0 * {my_var}" \
                        for my_var in kwargs.keys()])
        else:
            assert False, "NSRTS requires BFGS"

        for idx, key in enumerate(kwargs.keys()):
            
            expression = expression.replace(f"x_{idx+1}", key)

        expression = self.parse_filter(expression, kwargs.keys())
        
        return expression

    def initialize_model(self):

        json_filepath = os.path.join( NSRTS_DIR, "jupyter", "100M", "eq_setting.json")
        with open(json_filepath, 'r') as json_file:
            eq_setting = json.load(json_file)

        self.config_filepath = os.path.join( NSRTS_DIR, "jupyter", "100M", "config.yaml")
        self.config = omegaconf.OmegaConf.load(self.config_filepath)
        self.weights_path = os.path.join(NSRTS_DIR, "weights", "100M.ckpt")
            
        ## Set up BFGS load rom the hydra config yaml
        bfgs = BFGSParams(
                activated= self.config.inference.bfgs.activated,
                n_restarts= self.config.inference.bfgs.n_restarts,
                add_coefficients_if_not_existing=self.config.inference.bfgs.add_coefficients_if_not_existing,
                normalization_o=self.config.inference.bfgs.normalization_o,
                idx_remove=self.config.inference.bfgs.idx_remove,
                normalization_type=self.config.inference.bfgs.normalization_type,
                stop_time=self.config.inference.bfgs.stop_time,
            )

        # adjust this parameter up for greater accuracy and longer runtime
        self.config.inference.beam_size = self.my_beam_width

        self.params_fit = FitParams(word2id=eq_setting["word2id"], 
                                    id2word={int(k): v for k,v in eq_setting["id2word"].items()}, 
                                    una_ops=eq_setting["una_ops"], 
                                    bin_ops=eq_setting["bin_ops"], 
                                    total_variables=list(eq_setting["total_variables"]),  
                                    total_coefficients=list(eq_setting["total_coefficients"]),
                                    rewrite_functions=list(eq_setting["rewrite_functions"]),
                                    bfgs=bfgs,
                                    beam_size=self.config.inference.beam_size 
                                    )
        self.model = Model(cfg=self.config.architecture)

    def load_parameters(self, filepath=None):

        if filepath is not None:
            self.model =  Model.load_from_checkpoint(filepath, cfg=self.config.architecture)
        else:
            self.model =  Model.load_from_checkpoint(self.weights_path, cfg=self.config.architecture)


