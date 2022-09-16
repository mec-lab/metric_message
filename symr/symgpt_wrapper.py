import os

import numpy as np
import torch

from sympy import lambdify
import sympy as sp

from symr.wrappers import BaseWrapper

from symgpt.models import GPT, GPTConfig, PointNetConfig
from symgpt.utils import processDataFiles, CharDataset,\
        sample_from_model, lossFunc

from scipy.optimize import minimize

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
        self.verbose = 1


    def clean_expression(self, expression):

        cleaned_expression = "".join(expression).split(">")[0][1:].replace("s","x").replace("q","s").replace("***","**")
        cleaned_expression = cleaned_expression.replace("xin", "sin").replace("cox","cos")

        not_allowed = "+-*/"

        if cleaned_expression[-1] in not_allowed:
            cleaned_expression = cleaned_expression[:-1]

        if cleaned_expression[0] in not_allowed:
            cleaned_expression = cleaned_expression[1:]
        
        return cleaned_expression

    def __call__(self, target, **kwargs):
        
        if len(kwargs.keys()) > 1: 
            if self.verbose:
                print("SymGPT not implemented for multiple input variables")
            expression = " + ".join([key for key in kwargs.keys()])
            return expression

        else:
            my_key = [key for key in kwargs.keys()][0]
            

        x = torch.tensor(kwargs[my_key])
        y = torch.tensor(target)

        if len(x.shape) == 1:
            x = x[:,None]

        if len(y.shape) == 1:
            y = y[:,None]

        x = torch.tensor(x.transpose(1,0)[None,:,:])
        y = torch.tensor(y.transpose(1,0)[None,:,:])

        points = torch.cat([x, y], dim=1).float()
        
        variables = torch.tensor([1])
        temperature = 0.01
        top_k = 0.0
        top_p = 0.7
        blockSize = 64
        do_sample = False
        inputs = torch.tensor([[23]]) # assume 23 is start token '<'

        pred_outputs = sample_from_model(self.model, inputs,
            blockSize, points=points,\
            variables=variables, temperature=temperature,\
            sample=do_sample, top_k=top_k, top_p=top_p)

        string_output = [self.char_dict[elem.item()] for elem in pred_outputs[0]]
        pred_skeleton = "".join(string_output).split(">")[0][1:].replace("s","x").replace("q","s").replace("***","**")

        expression = self.clean_expression(pred_skeleton)

        constants = [1.0 for i,x in enumerate(pred_skeleton) if x=='C'] # initialize coefficients as 1

        if self.use_bfgs:
            optimized = minimize(lossFunc, constants, args=(pred_skeleton, x.numpy(), y.numpy()), method="BFGS") 

        constants_placed = 0

        pred_expression = ""


        for my_char in expression:

            if my_char == "C" and self.use_bfgs:
                pred_expression += f"{optimized.x[constants_placed]}"
                constants_placed += 1

            elif my_char == "C":
                pred_expression += f"{1.0}"
                constants_placed += 1
            else:
                pred_expression += my_char
            
        return pred_expression


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
        path = os.path.join(SYMGPT_DIR, "datasets", "exp_test_temp", "Train", "*.json")
        my_device = torch.device("cpu")


        files = glob.glob(path)[:maxNumFiles]
        text = processDataFiles(files)
        chars = sorted(list(set(text))+['_','T','<','>',':']) # extract unique characters from the text before converting the text to a list, # T is for the test data
        self.chars =  chars
        self.char_dict =  {index:elem for index, elem in enumerate(chars[:])}
        
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


