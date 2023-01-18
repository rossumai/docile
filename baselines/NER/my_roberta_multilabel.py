from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss

# from torch.nn import CrossEntropyLoss
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.models.bert.modeling_bert import _CONFIG_FOR_DOC
from transformers.models.xlm_roberta import XLMRobertaModel

# from transformers.models.xlm_roberta.configuration_xlm_roberta import XLMRobertaConfig
from transformers.models.xlm_roberta.modeling_xlm_roberta import (
    XLM_ROBERTA_INPUTS_DOCSTRING,
    XLM_ROBERTA_START_DOCSTRING,
    XLMRobertaEncoder,
    XLMRobertaPreTrainedModel,
)
from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)


class MyXLMRobertaMLConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`XLMRobertaModel`] or a [`TFXLMRobertaModel`]. It
    is used to instantiate a XLM-RoBERTa model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the XLMRoBERTa
    [xlm-roberta-base](https://huggingface.co/xlm-roberta-base) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the XLM-RoBERTa model. Defines the number of different tokens that can be represented by
            the `inputs_ids` passed when calling [`XLMRobertaModel`] or [`TFXLMRobertaModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the `token_type_ids` passed when calling [`XLMRobertaModel`] or
            [`TFXLMRobertaModel`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        position_embedding_type (`str`, *optional*, defaults to `"absolute"`):
            Type of position embedding. Choose one of `"absolute"`, `"relative_key"`, `"relative_key_query"`. For
            positional embeddings use `"absolute"`. For more information on `"relative_key"`, please refer to
            [Self-Attention with Relative Position Representations (Shaw et al.)](https://arxiv.org/abs/1803.02155).
            For more information on `"relative_key_query"`, please refer to *Method 4* in [Improve Transformer Models
            with Better Relative Position Embeddings (Huang et al.)](https://arxiv.org/abs/2009.13658).
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        classifier_dropout (`float`, *optional*):
            The dropout ratio for the classification head.

    Examples:

    ```python
    >>> from transformers import XLMRobertaConfig, XLMRobertaModel

    >>> # Initializing a XLM-RoBERTa xlm-roberta-base style configuration
    >>> configuration = XLMRobertaConfig()

    >>> # Initializing a model (with random weights) from the xlm-roberta-base style configuration
    >>> model = XLMRobertaModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    # model_type = "my-xlm-roberta"
    # model_type = "my-xlm-roberta"
    model_type = "my-roberta-multilabel"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        position_embedding_type="absolute",
        use_cache=True,
        classifier_dropout=None,
        use_2d_positional_embeddings=False,
        use_1d_positional_embeddings=False,
        bb_emb_dim=2500,
        use_new_2D_pos_emb=False,
        pos_emb_dim=2500,
        quant_step_size=5,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout
        self.use_2d_positional_embeddings = use_2d_positional_embeddings
        self.use_1d_positional_embeddings = use_1d_positional_embeddings
        self.bb_emb_dim = bb_emb_dim
        self.use_new_2D_pos_emb = use_new_2D_pos_emb
        self.pos_emb_dim = pos_emb_dim
        self.quant_step_size = quant_step_size


class MyXLMRobertaMLPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # config_class = XLMRobertaConfig
    config_class = MyXLMRobertaMLConfig
    base_model_prefix = "roberta"
    supports_gradient_checkpointing = True
    _no_split_modules = []

    # Copied from transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, XLMRobertaEncoder):
            module.gradient_checkpointing = value

    def update_keys_to_ignore(self, config, del_keys_to_ignore):
        """Remove some keys from ignore list"""
        if not config.tie_word_embeddings:
            # must make a new list, or the class variable gets modified!
            self._keys_to_ignore_on_save = [
                k for k in self._keys_to_ignore_on_save if k not in del_keys_to_ignore
            ]
            self._keys_to_ignore_on_load_missing = [
                k for k in self._keys_to_ignore_on_load_missing if k not in del_keys_to_ignore
            ]


class MyRobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        # x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


@add_start_docstrings(
    """
    XLM-RoBERTa Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g.
    for Named-Entity-Recognition (NER) tasks.
    """,
    XLM_ROBERTA_START_DOCSTRING,
)
# Copied from transformers.models.roberta.modeling_roberta.RobertaForTokenClassification with Roberta->XLMRoberta, ROBERTA->XLM_ROBERTA
class MyXLMRobertaMLForTokenClassification(XLMRobertaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = XLMRobertaModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)

        self.use_2d_positional_embeddings = config.use_2d_positional_embeddings
        self.use_1d_positional_embeddings = config.use_1d_positional_embeddings
        self.use_2d_concat = config.use_2d_concat
        self.use_new_2D_pos_emb = config.use_new_2D_pos_emb

        self.quant_step_size = config.quant_step_size

        if config.use_2d_positional_embeddings and config.use_2d_concat:
            bb_emb_dim = config.bb_emb_dim
            self.bb_left_emb = nn.Embedding(bb_emb_dim, config.hidden_size // 4)
            self.bb_top_emb = nn.Embedding(bb_emb_dim, config.hidden_size // 4)
            self.bb_right_emb = nn.Embedding(bb_emb_dim, config.hidden_size // 4)
            self.bb_bottom_emb = nn.Embedding(bb_emb_dim, config.hidden_size // 4)
        elif config.use_2d_positional_embeddings:
            bb_emb_dim = config.bb_emb_dim
            self.bb_left_emb = nn.Embedding(bb_emb_dim, config.hidden_size)
            self.bb_top_emb = nn.Embedding(bb_emb_dim, config.hidden_size)
            self.bb_right_emb = nn.Embedding(bb_emb_dim, config.hidden_size)
            self.bb_bottom_emb = nn.Embedding(bb_emb_dim, config.hidden_size)
        elif config.use_new_2D_pos_emb:
            pos_emb_dim = int(config.pos_emb_dim / config.quant_step_size)
            self.pos2_cx_emb = nn.Embedding(pos_emb_dim, config.hidden_size)
            self.pos2_cy_emb = nn.Embedding(pos_emb_dim, config.hidden_size)
            self.pos2_w_emb = nn.Embedding(pos_emb_dim, config.hidden_size)
            self.pos2_h_emb = nn.Embedding(pos_emb_dim, config.hidden_size)

        if config.use_1d_positional_embeddings:
            self.pos_emb = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        try:
            self.use_classification_head = config.use_classification_head
        except Exception:
            self.use_classification_head = False

        if self.use_classification_head:
            self.classifier = MyRobertaClassificationHead(config)
        else:
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(
        XLM_ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        checkpoint="Jean-Baptiste/roberta-large-ner-english",
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output="['O', 'ORG', 'ORG', 'O', 'O', 'O', 'O', 'O', 'LOC', 'O', 'LOC', 'LOC']",
        expected_loss=0.01,
    )
    def forward(
        self,
        bboxes: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if self.use_1d_positional_embeddings:
            input_shape = input_ids.size()
            seq_length = input_shape[1]
            if position_ids is None:
                position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
                position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
            position_emb = self.pos_emb(position_ids)

        if self.use_2d_positional_embeddings:
            # create embeddings (each coordinate separate)
            # bb_left_emb = self.bb_left_emb(bboxes[:, 0])
            bb_left_emb = self.bb_left_emb(bboxes[:, :, 0])
            # bb_top_emb = self.bb_top_emb(bboxes[:, 1])
            bb_top_emb = self.bb_top_emb(bboxes[:, :, 1])
            # bb_right_emb = self.bb_right_emb(bboxes[:, 2])
            bb_right_emb = self.bb_right_emb(bboxes[:, :, 2])
            # bb_bottom_emb = self.bb_bottom_emb(bboxes[:, 3])
            bb_bottom_emb = self.bb_bottom_emb(bboxes[:, :, 3])
            # n_rep = outputs[0].shape[1]

            # final bbox embedding is a sum of all coordinate embeddings
            # bbox_embedding = bb_top_emb.unsqueeze(1).expand(-1, n_rep, -1) + bb_left_emb.unsqueeze(1).expand(-1, n_rep, -1) + bb_bottom_emb.unsqueeze(1).expand(-1, n_rep, -1) + bb_right_emb.unsqueeze(1).expand(-1, n_rep, -1)
            # bbox_embedding = bb_top_emb + bb_left_emb + bb_bottom_emb + bb_right_emb
            if self.use_2d_concat:
                bbox_embedding = torch.cat(
                    [bb_top_emb, bb_left_emb, bb_bottom_emb, bb_right_emb], dim=-1
                )
            else:
                bbox_embedding = bb_top_emb + bb_left_emb + bb_bottom_emb + bb_right_emb

        if self.use_new_2D_pos_emb:
            l = bboxes[:, :, 0]  # noqa: E741
            t = bboxes[:, :, 1]
            r = bboxes[:, :, 2]
            b = bboxes[:, :, 3]
            cx = (l + r) / 2
            cy = (t + b) / 2
            w = r - l
            h = b - t
            pos2_cx_emb = self.pos2_cx_emb((cx / self.quant_step_size + 0.5).int())
            pos2_cy_emb = self.pos2_cx_emb((cy / self.quant_step_size + 0.5).int())
            pos2_w_emb = self.pos2_cx_emb((w / self.quant_step_size + 0.5).int())
            pos2_h_emb = self.pos2_cx_emb((h / self.quant_step_size + 0.5).int())

            pos2_emb = pos2_cx_emb + pos2_cy_emb + pos2_w_emb + pos2_h_emb

        sequence_output = outputs[0]

        # add 1D positional embedding
        if self.use_1d_positional_embeddings:
            sequence_output += position_emb

        # add 2D positional embedding
        if self.use_2d_positional_embeddings:
            sequence_output += bbox_embedding

        if self.use_new_2D_pos_emb:
            sequence_output += pos2_emb

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            # loss_fct = CrossEntropyLoss()
            loss_fct = BCEWithLogitsLoss()
            # loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            loss = loss_fct(logits, labels.float())

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
