# -*- coding: utf-8 -*-
"""
Created on Sat May 17 01:25:46 2025

the model...

@author: faith
"""

import torch
import torch.nn as nn
import numpy as np
from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
__logger__ = logging.getLogger(__name__)


def __model__size(model):
    params = filter(lambda p: p.requires_grad, model.parameters())
    size = sum([np.prod(p.size()) for p in params])
    return '{}M'.format(round(size / 1e+6))
def load_or_build_model(config_name, tokenizer_name, model_name_or_path, loaded_model_path):
    """
    Parameters
    ----------
    config_name : str
        name of model config.
    tokenizer_name : TYPE
        name of tokenizer.
    model_name_or_path : TYPE
        name of foundational model we're transferling.
    loaded_model_path : TYPE
        name of our model propenster/text2bn.

    Returns
    -------
    config : TYPE
        pretrained model config.
    tokenizer : TYPE
        pretrained model tokenizer.
    model : TYPE
        pretrained model....

    """
    config = T5Config.from_pretrained(config_name if config_name else model_name_or_path)
    tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name_or_path)
    
    __logger__.info("Finished loading model [%s] from %s", __model__size(model), model_name_or_path)
    if loaded_model_path is not None:
        __logger__.info("Reloading model from {}".format(loaded_model_path))
        model.load_state_dict(torch.load(loaded_model_path))
        
    return config, tokenizer, model



class Seq2Seq(nn.Module):
    """
        Our model relies on T5-ForConditoinalGeneration so we use a Seq2Seq head
        Build Seqence-to-Sequence.

        Parameters:

        * `encoder`- encoder of seq2seq model. e.g. roberta
        * `decoder`- decoder of seq2seq model. e.g. transformer
        * `config`- configuration of encoder model.
        * `beam_size`- beam size for beam search.
        * `max_length`- max length of target for beam search.
        * `sos_id`- start of symbol ids in target for beam search.
        * `eos_id`- end of symbol ids in target for beam search.
    """
    
    def __int__(self, encoder, decoder, config, beam_size=None, max_length=None, sos_id=None, eos_id=None):
        super(Seq2Seq, self).__init__()
        #encoder...t5-small's encoder
        #input...
        self.encoder = encoder
        self.decoder = decoder
        #t5-small's config...
        self.config = config
        self.register_buffer("bias", torch.tril(torch.ones(2048, 2048)))
        #dense layer...
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lsm = nn.LogSoftmax(dim=-1)
        #ensure you clone weights...transfer this weights...pls...
        self.__tie_weights__()
        self.beam_size = beam_size
        self.max_length = max_length
        self.sos_id = sos_id
        self.eos_id=eos_id
        
        
    def tie_weights(self):
        """make sure to tie or clone weights... try to use TorchScript"""
        if self.config.torchscript:
            self.lm_head.weight = nn.Parameter(self.encoder.embeddings.word_embeddings.weight.clone())
        else:
            self.lm_head.weight = self.encoder.embeddings.word_embeddings.weight
    
    def forward(self, input_ids=None, input_mask=None, target_ids=None, target_mask=None, args=None):
        outputs = self.encoder(input_ids, attention_mask=input_mask)
        encoder_output = outputs[0].permute([1, 0, 2]).contiguous()
        if target_ids is not None:
            attention_mask = -1e4 * (1 - self.bias[:target_ids.shape[1], :target_ids.shape[1]])
            target_embeddings = self.encoder.embeddings(target_ids).permute([1, 0, 2]).contiguous()
            output = self.decoder(target_embeddings, encoder_output, target_mask=attention_mask, memory_key_padding_mask=~input_mask)
            #now we activate the hiddenstates
            hidden_states = torch.tanh(self.dense(output)).permute([1, 0, 2]).contiguous()
            lm_logits = self.lm_head(hidden_states)
            
            #shift so we can predict tokens < n
            active_loss = target_mask[..., 1:].ne(0).view(-1) == 1
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = target_ids[..., 1:].contiguous()
            #flatten the tokens...CE
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[active_loss], shift_labels.view(-1)[active_loss])
            outputs = loss, loss * active_loss.sum(), active_loss.sum()
            return outputs
        
        else:
            #do seq senti preidction
            predictions = []
            zeros = torch.cuda.LongTensor(1).fill_(0)
            for k in range(input_ids.shape[0]):
                ctx = encoder_output[:, k:k+1]
                ctx_mask = input_mask[k:k+1, :]
                beam = Beam(self.beam_size, self.sos_id, self.eos_id)
                input_ids = beam.get_current_state()
                ctx = ctx.repeat(1, self.beam_size, 1)
                ctx_mask = ctx_mask.repeat(self.beam_size, 1)
                for _ in range(self.max_length):
                    if beam.done():
                        break
                    attention_mask = -1e4 * (1 - self.bias[:input_ids.shape[1], :input_ids.shape[1]])
                    target_embeddings = self.encoder.embeddings(input_ids).permute([1, 0, 2]).contiguous()
                    out = self.decoder(target_embeddings, ctx, tgt_mask=attention_mask,
                                       memory_key_padding_mask=~ctx_mask)
                    # memory_key_padding_mask=(1 - context_mask).bool())
                    out = torch.tanh(self.dense(out))
                    hidden_states = out.permute([1, 0, 2]).contiguous()[:, -1, :]
                    out = self.lsm(self.lm_head(hidden_states)).data
                    beam.advance(out)
                    input_ids.data.copy_(input_ids.data.index_select(0, beam.get_current_origin()))
                    input_ids = torch.cat((input_ids, beam.get_current_state()), -1)
                hyp = beam.get_hyp(beam.get_final())
                prediction = beam.build_target_tokens(hyp)[:self.beam_size]
                prediction = [torch.cat([x.view(-1) for x in p] + [zeros] * (self.max_length - len(p))).view(1, -1) for p in
                        prediction]
                predictions.append(torch.cat(prediction, 0).unsqueeze(0))

            predictions = torch.cat(predictions, 0)
            return predictions
                
                
                
                
                
                
class Beam(object):
    """
        RAG-sheize...
        Beam search process during sequence generation,
        keeping track of scores and hypotheses to select the most probable output sequences.
    
    
    """
    def __init__(self, size, sos, eos):
        self.size = size
        self.tt = torch.cuda
        #keep track of scores for each transl
        self.scores = self.tt.FloatTensor(size).zero_()
        #the backpointers at each time-step
        self.prev_ks = []
        #the outputs at teach time-step
        self.next_ys = [self.tt.LongTensor(size).fill_(0)]
        self.next_ys[0][0] = sos
        #gas eos topped the beam yet?
        self._eos = eos
        self.eos_top = False
        #time and k pair for finished
        self.finished = []
        
    def get_current_state(self):
        """Get the output state ykth for the current timestep...."""
        return self.tt.LongTensor(self.next_ys[-1]).view(-1, 1)

    def get_current_origin(self):
        """Where did this current times-tep start from?"""
        return self.prev_ks[-1]
    
    def advance(self, wordLk):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attnOut`: Compute and update the beam search.

        Parameters:

        * `wordLk`- probs of advancing from the last step (K x words)
        * `attnOut`- attention at the last step

        Returns: True if beam search is complete.
        """
        numWords = wordLk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)

            # Don't let EOS have children.
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] == self._eos:
                    beamLk[i] = -1e20
        else:
            beamLk = wordLk[0]
        flatBeamLk = beamLk.view(-1)
        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)

        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prevK = bestScoresId // numWords
        self.prevKs.append(prevK)
        self.nextYs.append((bestScoresId - prevK * numWords))

        for i in range(self.nextYs[-1].size(0)):
            if self.nextYs[-1][i] == self._eos:
                s = self.scores[i]
                self.finished.append((s, len(self.nextYs) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.nextYs[-1][0] == self._eos:
            self.eosTop = True

    def done(self):
        return self.eosTop and len(self.finished) >= self.size

    def get_final(self):
        if len(self.finished) == 0:
            self.finished.append((self.scores[0], len(self.nextYs) - 1, 0))
        self.finished.sort(key=lambda a: -a[0])
        if len(self.finished) != self.size:
            unfinished = []
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] != self._eos:
                    s = self.scores[i]
                    unfinished.append((s, len(self.nextYs) - 1, i))
            unfinished.sort(key=lambda a: -a[0])
            self.finished += unfinished[:self.size - len(self.finished)]
        return self.finished[:self.size]

    def get_hyp(self, beam_res):
        """
        Walk back to construct the full hypothesis.
        """
        hyps = []
        for _, timestep, k in beam_res:
            hyp = []
            for j in range(len(self.prevKs[:timestep]) - 1, -1, -1):
                hyp.append(self.nextYs[j + 1][k])
                k = self.prevKs[j][k]
            hyps.append(hyp[::-1])
        return hyps

    def build_target_tokens(self, preds):
        sentence = []
        for pred in preds:
            tokens = []
            for tok in pred:
                if tok == self._eos:
                    break
                tokens.append(tok)
            sentence.append(tokens)
        return sentence
            
    

