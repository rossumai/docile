from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.models.bert.modeling_bert import _CONFIG_FOR_DOC
from transformers.models.xlm_roberta import XLMRobertaModel
from transformers.models.xlm_roberta.modeling_xlm_roberta import (
    XLM_ROBERTA_INPUTS_DOCSTRING,
    XLM_ROBERTA_START_DOCSTRING,
    XLMRobertaPreTrainedModel,
)
from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)


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
            pos2_cy_emb = self.pos2_cy_emb((cy / self.quant_step_size + 0.5).int())
            pos2_w_emb = self.pos2_w_emb((w / self.quant_step_size + 0.5).int())
            pos2_h_emb = self.pos2_h_emb((h / self.quant_step_size + 0.5).int())

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
            loss_fct = BCEWithLogitsLoss()
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
