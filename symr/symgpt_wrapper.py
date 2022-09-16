import os

import numpy as np
import torch

from sympy import lambdify
import sympy as sp

from symr.wrappers import BaseWrapper

from symgpt.models import GPT, GPTConfig, PointNetConfig
from symgpt.utils import processDataFiles, CharDataset,\
        sample_from_model, lossFunc

from scipy.optimize import minimize, least_squares

import glob
from pathlib import Path
from functools import partial

#

import json

if "SYMGPT_DIR" in os.environ.keys():
    SYMGPT_DIR = os.environ["SYMGPT_DIR"]
else:
    SYMGPT_DIR = "symgpt" 


class SymGPTWrapper(BaseWrapper):

    def __init__(self, **kwargs):

        if "use_bfgs" in kwargs.keys():
            self.use_bfgs = kwargs["use_bfgs"]
        else:
            self.use_bfgs = False

        self.my_device = "cpu"

        self.initialize_model()
        self.load_parameters()


    def __call__(self, target, **kwargs):
        
        expression = " + ".join([key for key in kwargs.keys()])

        return expression

    def initialize_model(self):

        embeddingSize = 512
        numPoints = [20,21]
        numVars = 1
        numYs = 1
        method = "EMB_SUM"
        variableEmbedding = "NOT_VAR"

        # create the model
        pconf = PointNetConfig(embeddingSize=embeddingSize,
                               numberofPoints=numPoints[1]-1,
                               numberofVars=numVars,
                               numberofYs=numYs,
                               method=method,
                               variableEmbedding=variableEmbedding)    

        blockSize = 64

        maxNumFiles = 100
        const_range = [-2.1, 2.1]
        decimals = 8
        trainRange = [-3.0,3.0]

        target = "Skeleton"
        addVars = True if variableEmbedding == 'STR_VAR' else False
        path = os.path.join("./symbolicgpt", "datasets", "exp_test_temp", "Train", "*.json")
        my_device = torch.device("cpu")


        files = glob.glob(path)[:maxNumFiles]
        text = processDataFiles(files)
        chars = sorted(list(set(text))+['_','T','<','>',':']) # extract unique characters from the text before converting the text to a list, # T is for the test data
        text = text.split('\n') # convert the raw text to a set of examples
        trainText = text[:-1] if len(text[-1]) == 0 else text
        vocab_size = 49

        train_dataset = CharDataset(text, blockSize, chars, numVars=numVars,
                        numYs=numYs, numPoints=numPoints, target=target, addVars=addVars,
                        const_range=const_range, xRange=trainRange, decimals=decimals, augment=False)


        mconf = GPTConfig(vocab_size, blockSize,
                          n_layer=8, n_head=8, n_embd=embeddingSize,
                          padding_idx=train_dataset.paddingID)

        self.model = GPT(mconf, pconf)   
        model_name = "XYE_1Var_30-31Points_512EmbeddingSize_SymbolicGPT_GPT_PT_EMB_SUM_Skeleton_Padding_NOT_VAR_MINIMIZE.pt"
        self.weights_path = os.path.join(SYMGPT_DIR, "Models", model_name)

        self.model = self.model.eval().to(self.my_device)

    def load_parameters(self, filepath=None):

        if filepath is not None:
            self.model.load_state_dict(torch.load(filepath, map_location=self.my_device))
        else:
            self.model.load_state_dict(torch.load(self.weights_path, map_location=self.my_device))


