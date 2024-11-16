from threading import local
from typing import List, Optional, Tuple, Union

from transformers import BartForConditionalGeneration
import transformers
from transformers.models.bart.modeling_bart import BartLearnedPositionalEmbedding
from transformers.modeling_outputs import Seq2SeqModelOutput
from torch.nn import Module
import torch
from colorama import Fore, Style

from models.local_attention import LocalSelfAttention

class DistillBartSummarizer(Module):
    def __init__(self,
        checkpoint:str = 'sshleifer/distilbart-cnn-12-6', 
        max_tokens: int = 4096, 
        generate_max_length: int = 220, 
        local_attention:bool = False, 
        attention_window=1024
        ):
        super(DistillBartSummarizer, self).__init__()
        self.inner_bart = BartForConditionalGeneration.from_pretrained(checkpoint)
        #self.inner_bart.gradient_checkpointing_enable()
        self.config = self.inner_bart.config
        self.max_tokens = max_tokens
        self.config.max_length = generate_max_length
        self.attention_window = attention_window
        # if max_tokens > self.config.max_position_embeddings:
        #     self.expand_positional_embedding()
        
        # if local_attention:
        #     self.swap_attention()

    def swap_attention(self):
        """
        based on HuggingFace v4.11.0, copied and adapted from: https://github.com/potsawee/longsum0

        BART: self.model.encoder.layers[0].self_attn
            - q_proj, k_proj, v_proj ---- Linear dim (embed_dim,embed_dim)
            - out_proj               ---- Linear dim (embed_dim,embed_dim)
            ***
            head_dim = embed_dim // num_heads

        """

        print("==================================================================================")
        print(f"===> attention_window: {Fore.LIGHTRED_EX}{self.attention_window}{Style.RESET_ALL}")
        print("==================================================================================")


        for i in range(len(self.inner_bart.model.encoder.layers)):
            local_attn = LocalSelfAttention(self.config, layer_id=i, attention_window=self.attention_window) # corrected!!

            # Copy local attention weights
            local_attn.self_attn.query.weight.data = self.inner_bart.model.encoder.layers[i].self_attn.q_proj.weight.data
            local_attn.self_attn.key.weight.data = self.inner_bart.model.encoder.layers[i].self_attn.k_proj.weight.data
            local_attn.self_attn.value.weight.data = self.inner_bart.model.encoder.layers[i].self_attn.v_proj.weight.data
            # Copy global attention weight (even if not used)
            local_attn.self_attn.query_global.weight.data = self.inner_bart.model.encoder.layers[i].self_attn.q_proj.weight.data
            local_attn.self_attn.key_global.weight.data = self.inner_bart.model.encoder.layers[i].self_attn.k_proj.weight.data
            local_attn.self_attn.value_global.weight.data = self.inner_bart.model.encoder.layers[i].self_attn.v_proj.weight.data
            # Copy output projection
            local_attn.out_proj.weight.data = self.inner_bart.model.encoder.layers[i].self_attn.out_proj.weight.data
            self.inner_bart.model.encoder.layers[i].self_attn = local_attn

        print("Swapped BART' encoder full self attention to local self attention")

    def expand_positional_embedding(self):
        # Based on idea from LoBART (https://github.com/potsawee/longsum0) that expanding positional embeddings
        # by flipping the original embeddings is better than random initialization
        # code by me
        print(f"Expanding Pos. Embedding to: {Fore.LIGHTRED_EX}{self.max_tokens}{Style.RESET_ALL}")
        previous_max_length = self.inner_bart.config.max_position_embeddings
        self.config.max_position_embeddings = self.max_tokens
        encoder_embeddings = self.inner_bart.model.encoder.embed_positions
        self.inner_bart.model.encoder.embed_positions = BartLearnedPositionalEmbedding(
            self.max_tokens,
            self.config.d_model
        )

        self.inner_bart.model.encoder.embed_positions.weight.requires_grad = False
        pos = 0
        length = previous_max_length+2
        flip = False
        with torch.no_grad():
            while pos < self.max_tokens:
                end = min(pos+length, self.max_tokens+2)
                if flip:
                    self.inner_bart.model.encoder.embed_positions.weight[pos:end,:] = torch.flip(encoder_embeddings.weight, [0])[:end-pos, :]
                else:
                    self.inner_bart.model.encoder.embed_positions.weight[pos:end,:] = encoder_embeddings.weight[:end-pos, :]
                
                pos += length
                if length == previous_max_length+2:
                    length = previous_max_length
                flip = not flip
                
        self.inner_bart.model.encoder.embed_positions.weight.requires_grad = True
        print('positional ncoding expanded to {}'.format(self.max_tokens))
        pass

    def forward(self, 
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None
        
        )-> Union[Tuple, Seq2SeqModelOutput]:

        return self.inner_bart.forward(input_ids, 
            attention_mask, 
            decoder_input_ids, 
            decoder_attention_mask, 
            head_mask,
            decoder_head_mask,
            cross_attn_head_mask,
            encoder_outputs,
            past_key_values,
            inputs_embeds,
            decoder_inputs_embeds,
            labels,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict)

    def generate(self, input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None):
        return self.inner_bart.generate(input_ids=input_ids, attention_mask=attention_mask)