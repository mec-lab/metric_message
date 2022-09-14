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

import json

if "NSRTS_DIR" in os.environ.keys():
    NSRTS_DIR = os.environ["NSRTS_DIR"]
else:
    NSRTS_DIR = "nsrts" 

class NSRTSWrapper(BaseWrapper):

    def __init__(self, **kwargs):

        self.my_beam_width = 1
        self.initialize_model()
        self.load_parameters()

    def __call__(self, target, **kwargs):
        
        expression = "0.0 " + "".join([f"+ {key} " for key in kwargs.keys()])

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

        params_fit = FitParams(word2id=eq_setting["word2id"], 
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


