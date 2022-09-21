import os

import numpy as np
import torch
import torch.nn.functional as F

from sympy import lambdify
import sympy as sp

from symr.wrappers import BaseWrapper

import nesymres
from nesymres.architectures.model import Model
from nesymres.architectures.beam_search import BeamHypotheses
from nesymres.dataset.generator import Generator
from nesymres.architectures.data import de_tokenize
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

        self.my_device = torch.device("cpu")
        self.my_beam_width = kwargs["beam_width"] \
                if "beam_width" in kwargs.keys() else 4
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
            return expression, False

        except:
            return "+".join([f"0.0 * {my_var}" \
                    for my_var in variables]), True

    def no_bfgs_inference(self, x, target):
        """
        parts of this function are cribbed from the MIT licensed
        https://github.com/SymposiumOrganization/NeuralSymbolicRegressionThatScales
        """
        
        x = torch.tensor(x[None,:])
        y = torch.tensor(target[None,:])

        if x.shape[2] < self.model.cfg.dim_input - 1:
            pad = torch.zeros(1, x.shape[1], self.model.cfg.dim_input - x.shape[2]-1)
            x = torch.cat((x, pad), dim=2)

        if len(y.shape) < 3:
            y = y.unsqueeze(-1)

        with torch.no_grad():
            # 
            encoder_input = torch.cat((x,y), dim=2)

            enc_src = self.model.enc(encoder_input)
            src_enc = enc_src

            enc_src_shape = (self.my_beam_width,) + src_enc.shape[1:]

            enc_src = src_enc.unsqueeze(1).expand((\
                    1, self.my_beam_width) + src_enc.shape[1:])\
                    .contiguous().view(enc_src_shape)

            #assert enc_src.size(0) == self.my_beam_width
            generated = torch.zeros(\
                    [self.my_beam_width, \
                    self.model.cfg.length_eq],\
                    dtype=torch.long, \
                    device=self.my_device)
            generated[:,0] = 1

            cache = {"slen": 0}

            generated_hyps = BeamHypotheses(self.my_beam_width,\
                    self.model.cfg.length_eq, 1.0, 1)
            done = False

            beam_scores = torch.zeros(\
                    self.my_beam_width, device=self.my_device,\
                    dtype=torch.long)
            beam_scores[1:] = -1e9

            cur_len = torch.tensor(1, device=self.my_device,\
                    dtype=torch.int64)

            self.model.eval()

            while cur_len < self.model.cfg.length_eq:

                generated_mask1, generated_mask2 = self.model.make_trg_mask(
                    generated[:, :cur_len]
                )

                pos = self.model.pos_embedding(
                    torch.arange(0, cur_len)  #### attention here
                    .unsqueeze(0)
                    .repeat(generated.shape[0], 1)
                    .type_as(generated)
                )
                te = self.model.tok_embedding(generated[:, :cur_len])
                trg_ = self.model.dropout(te + pos)

                output = self.model.decoder_transfomer(
                    trg_.permute(1, 0, 2),
                    enc_src.permute(1, 0, 2),
                    generated_mask2.float(),
                    tgt_key_padding_mask=generated_mask1.bool(),
                )
                output = self.model.fc_out(output)
                output = output.permute(1, 0, 2).contiguous()
                scores = F.log_softmax(output[:, -1:, :], dim=-1).squeeze(1)
                
                assert output[:, -1:, :].shape == (self.my_beam_width, 1, self.model.cfg.length_eq,)

                n_words = scores.shape[-1]
                # select next words with scores
                _scores = scores + beam_scores[:, None].expand_as(scores)
                _scores = _scores.view(self.my_beam_width* n_words) 

                next_scores, next_words = torch.topk(_scores, 2 * self.my_beam_width, dim=0, largest=True, sorted=True)
                assert len(next_scores) == len(next_words) == 2 * self.my_beam_width
                done = done or generated_hyps.is_done(next_scores.max().item())
                next_sent_beam = []

                # next words for this sentence
                for idx, value in zip(next_words, next_scores):

                    # get beam and word IDs
                    beam_id = idx // n_words
                    word_id = idx % n_words

                    # end of sentence, or next word
                    if (
                        word_id == self.params_fit.word2id["F"]
                        or cur_len + 1 == self.model.cfg.length_eq
                    ):
                        generated_hyps.add(
                            generated[
                                 beam_id,
                                :cur_len,
                            ]
                            .clone()
                            .cpu(),
                            value.item(),
                        )
                    else:
                        next_sent_beam.append(
                            (value, word_id, beam_id)
                        )

                    # the beam for next step is full
                    if len(next_sent_beam) == self.my_beam_width:
                        break

                # update next beam content
                assert (
                    len(next_sent_beam) == 0
                    if cur_len + 1 == self.model.cfg.length_eq
                    else self.my_beam_width
                )
                if len(next_sent_beam) == 0:
                    next_sent_beam = [
                        (0, self.model.trg_pad_idx, 0)
                    ] * self.params_fit.beam_size  # pad the batch


                #next_batch_beam.extend(next_sent_beam)
                assert len(next_sent_beam) == self.my_beam_width

                beam_scores = torch.tensor(
                    [x[0] for x in next_sent_beam], device=self.my_device
                )  # .type(torch.int64) Maybe #beam_scores.new_tensor([x[0] for x in next_batch_beam])
                beam_words = torch.tensor(
                    [x[1] for x in next_sent_beam], device=self.my_device
                )  # generated.new([x[1] for x in next_batch_beam])
                beam_idx = torch.tensor(
                    [x[2] for x in next_sent_beam], device=self.my_device
                )
                generated = generated[beam_idx, :]
                generated[:, cur_len] = beam_words
                for k in cache.keys():
                    if k != "slen":
                        cache[k] = (cache[k][0][beam_idx], cache[k][1][beam_idx])

                # update current length
                cur_len = cur_len + torch.tensor(
                    1, device=self.my_device, dtype=torch.int64
                )


        my_scores = []
        my_expressions = []

        self.params_fit.id2word[3] = "constant"
        for my_beam in generated_hyps.hyp:
            
            raw = de_tokenize(my_beam[1][1:].tolist(), self.params_fit.id2word)

            candidate = Generator.prefix_to_infix(raw, 
                                    coefficients=["constant"], 
                                    variables=self.params_fit.total_variables)

            expression = candidate.replace("{constant}", "1.0")
            expression = expression.replace("ln", "log")

            expr_function = sp.lambdify(self.params_fit.total_variables, expr=expression)
            
            my_inputs = {}
            for ii in range(x.shape[-1]):
                my_inputs[f"x_{ii+1}"] = x.cpu().numpy()[0,:,ii]
            
            predicted = expr_function(**my_inputs)

            my_scores.append(((predicted-target)**2).mean())
            my_expressions.append(expression)

        best_expression = my_expressions[np.argmax(my_scores)]
        self.expressions = my_expressions
        self.expression_scores = my_scores
        self.best_expression = best_expression

        return best_expression

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
                return expression = "+".join([f"0.0 * {my_var}" \
                        for my_var in kwargs.keys()]), {"failed": True}
        else:
            try: 
                expression = self.no_bfgs_inference(x, target)
            except:
                return expression = "+".join([f"0.0 * {my_var}" \
                        for my_var in kwargs.keys()]), {"failed": True}

        for idx, key in enumerate(kwargs.keys()):
            
            expression = expression.replace(f"x_{idx+1}", key)

        expression, failed = self.parse_filter(expression, kwargs.keys())

        info = {"failed": failed}
        
        return expression, info

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


