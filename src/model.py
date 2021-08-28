import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
from pytorch_transformers import BertModel

from attention import *
from embeddings import *

class DialogRE(nn.Module):

    def __init__(self, args, type_embeddings, dropout=0.2):
        super().__init__()
        self.args = args
        self.hidden_size = args.bert_output_size
        self.offset = args.offset  # offset of slide window for sequences longer than 512
        self.type_embeddings = type_embeddings
        self.position_embeddings = UtterenceAwarePosEmbedding(args.max_utter_nums, args.max_seq_lens, args.bert_output_size)

        self.bert_encoder = BertModel.from_pretrained(args.bert_path)

        self.self_att1 = MultiHeadAttention(model_dim=self.hidden_size, num_heads=args.num_heads, dropout=dropout)
        self.self_att2 = MultiHeadAttention(model_dim=self.hidden_size, num_heads=args.num_heads, dropout=dropout)

        self.fusion = InfoFusion(self.hidden_size)

        self.entity_repr = EntityRepr(self.hidden_size)

        self.classifier = nn.Sequential(nn.Dropout(dropout),
                                        nn.Linear(self.hidden_size * 4, self.hidden_size),
                                        nn.ReLU(),
                                        nn.Dropout(dropout),
                                        nn.Linear(self.hidden_size, args.rel_nums)
                                        )

        self.is_test = False
        self.logisitc = nn.Sigmoid()

        self.mention_att = MentionAttention(self.hidden_size, self.hidden_size)
        self.rela_repr = RelationRepr(self.hidden_size)

        self.rel_nums = args.rel_nums

    def forward(self, batch):
        # encoder
        input_tokens_1, input_tokens_2, input_mask_1, input_mask_2 = batch['tokens_ids_1'], batch['tokens_ids_2'], batch['tokens_mask_1'], batch['tokens_mask_2']
        max_seq_lens = batch['max_seq_lens']
        batch_size = input_tokens_1.shape[0]
        input_feat, input_mask = self.encoder(input_tokens_1, input_tokens_2, input_mask_1, input_mask_2, batch_size, max_seq_lens)

        type_ids = batch['entity_type_ids']
        argument_aware_mask = (type_ids != 0).float()
        type_ids = type_ids.cuda()
        type_embedding = self.type_embeddings(type_ids)

        # input feature + type embeddings + utterance aware embeddings
        utter_position = batch['utterence_position'].cuda()
        utterence_embeddings = self.position_embeddings(utter_position, argument_aware_mask)
        input_feat = input_feat + type_embedding + utterence_embeddings
        argument_aware_mask = argument_aware_mask.unsqueeze(2).cuda()

        # windows mask
        utter_sa_mask = batch['utterence_mask'].cuda()
        local_sa_mask = batch['local_context_mask'].cuda()
        context_sa_mask = batch['wide_context_mask'].cuda()
        # speaker aware mask
        speaker_sa_mask = batch['speaker_aware_mask'].cuda()

        utter_sa = self.self_att1(input_feat, attn_mask=utter_sa_mask) * argument_aware_mask
        local_sa = self.self_att1(input_feat, attn_mask=local_sa_mask) * argument_aware_mask
        context_sa = self.self_att1(input_feat, attn_mask=context_sa_mask) * argument_aware_mask

        # speaker-aware self-attention
        speaker_aware_sa = self.self_att2(input_feat, attn_mask=speaker_sa_mask) * argument_aware_mask
        input_mask = input_mask.squeeze(2).unsqueeze(1)
        other_speaker_mask = (~speaker_sa_mask.bool()).float() * input_mask
        other_speaker_sa = self.self_att2(input_feat, attn_mask=other_speaker_mask.bool()) * argument_aware_mask

        # token information aggregation
        context_repr = torch.stack([local_sa, context_sa]).transpose(0, 1)
        context_repr, _ = torch.max(context_repr, dim=1)
        local_rela_repr = self.fusion(utter_sa, context_repr)
        speaker_aware_repr = self.fusion(speaker_aware_sa, other_speaker_sa)

        # entity aware representation
        entity_map = batch['entity_map']
        entity_repr1, mention_repr1, mention_mask1 = self.entity_repr(local_rela_repr, entity_map)
        entity_aware_repr1 = self.mention_att(entity_repr1, mention_repr1, mention_mask1)

        entity_repr2, mention_repr2, mention_mask2 = self.entity_repr(speaker_aware_repr, entity_map)
        entity_aware_repr2 = self.mention_att(entity_repr2, mention_repr2, mention_mask2)

        # relation representation
        relation_repr1 = self.rela_repr(entity_aware_repr1)
        relation_repr2 = self.rela_repr(entity_aware_repr2)

        relation_repr = torch.cat([relation_repr1, relation_repr2], dim=-1)

        # classifiers
        pred = self.classifier(relation_repr)

        return pred

    def encoder(self, input_tokens_1, input_tokens_2, input_mask_1, input_mask_2, batch_size, max_lens):

        if input_tokens_2 is None:
            input_tokens_1, input_mask_1 = input_tokens_1.cuda(), input_mask_1.cuda()
            input_feat = self.bert_encoder(input_tokens_1, attention_mask=input_mask_1)
            input_feat = input_feat[0]
            input_mask = input_mask_1.unsqueeze(2)
            input_feat = input_feat * input_mask
        else:  # if the seq length is longer than 512:
            input_tokens_1, input_mask_1 = input_tokens_1.cuda(), input_mask_1.cuda()
            input_tokens_2, input_mask_2 = input_tokens_2.cuda(), input_mask_2.cuda()
            input1 = self.bert_encoder(input_tokens_1, attention_mask=input_mask_1)
            input2 = self.bert_encoder(input_tokens_2, attention_mask=input_mask_2)
            input1 = input1[0]
            input2 = input2[0]
            padding1 = torch.zeros((batch_size, max_lens - 512, self.args.bert_output_size), dtype=torch.float).cuda()
            padding_mask1 = torch.zeros((batch_size, max_lens - 512), dtype=torch.float, requires_grad=False).cuda()
            pad_input1 = torch.cat([input1, padding1], dim=1)
            pad_mask1 = torch.cat([input_mask_1, padding_mask1], dim=1).unsqueeze(2)

            input2_cls = input2.narrow(1, 0, 1)
            input2_cnt = input2.narrow(1, 1, input2.shape[1] - 1)
            input_mask_cls = input_mask_2.narrow(1, 0, 1)
            input_mask_cnt = input_mask_2.narrow(1, 1, input2.shape[1] - 1)
            padding2 = torch.zeros((batch_size, self.offset - 1, self.args.bert_output_size), dtype=torch.float).cuda()
            padding_mask2 = torch.zeros((batch_size, self.offset - 1), dtype=torch.float, requires_grad=False).cuda()
            pad_input2 = torch.cat([input2_cls, padding2, input2_cnt], dim=1)
            pad_mask2 = torch.cat([input_mask_cls,
                                   padding_mask2,
                                   input_mask_cnt],
                                  dim=1
                                  ).unsqueeze(dim=2)

            pad_input2 = pad_input2 * pad_mask2
            pad_mask2 = pad_mask2 + 1e-12
            pad_input1 = pad_input1 * pad_mask1
            pad_mask1 = pad_mask1 + 1e-12

            input_feat = pad_input1 + pad_input2
            input_mask = pad_mask1 + pad_mask2
            input_feat = input_feat / input_mask
            input_mask = (input_mask != 0).float()
            input_feat = input_feat * input_mask

        return input_feat, input_mask


class InfoFusion(nn.Module):

    def __init__(self, hidden_size, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.gate = nn.Sequential(nn.Dropout(dropout),
                                  nn.Linear(self.hidden_size * 4, self.hidden_size * 1),
                                  nn.Sigmoid()
                                  )

    def forward(self, input1, input2):
        input = torch.cat([input1, input2, input1 - input2, input1 * input2], dim=2)
        gate = self.gate(input)

        return gate * input1 + (1 - gate) * input2


class EntityRepr(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, token_reprs, entity_maps):
        entity_reprs, mentions_reprs, mentions_mask = [], [], []
        for token_repr, entity_map in zip(token_reprs, entity_maps):
            lens = len(entity_map)
            entities, mentions = [], []
            for id in range(lens):
                mentions_pos = entity_map[id]['pos_idx'].cuda()
                m_repr = token_repr[mentions_pos]
                m_repr = torch.mean(m_repr, dim=1)
                e_repr = torch.mean(m_repr, dim=0)
                entities.append(e_repr)
                mentions.append(m_repr)

            entities = torch.stack(entities).squeeze(0)
            entity_reprs.append(entities)
            mention_nums = [m.size(0) for m in mentions]
            m_mask = [torch.ones(nums) for nums in mention_nums]

            mentions = rnn.pad_sequence(mentions, batch_first=True, padding_value=0)
            m_mask = rnn.pad_sequence(m_mask, batch_first=True, padding_value=0)
            mentions_reprs.append(mentions)
            mentions_mask.append(m_mask)

        return entity_reprs, mentions_reprs, mentions_mask


class RelationRepr(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, inputs):
        res = []
        for input in inputs:
            input_t = input.transpose(0, 1).reshape(-1, self.hidden_size)
            input = input.reshape(-1, self.hidden_size)
            output = torch.cat([input, input_t], dim=-1)
            res.append(output)

        res = torch.cat(res, dim=0)
        return res


class LogicReg(nn.Module):

    def __init__(self, rel_nums, rule_nums):
        super(LogicReg, self).__init__()
        self.rel_nums = rel_nums
        self.rule_nums = rule_nums
        self.logistic = nn.Sigmoid()

    def forward(self, preds, labels, label_masks, triple_nums, eps=1e-15):
        loss_p, loss_n = 0.0, 0.0
        symmetric_rule_idx = [2, 3, 7, 8, 9, 10, 11, 13, 15, 16]
        inverse_rule_idx = [(5, 4), (4, 5), (14, 12), (12, 14), (31, 17), (32, 18), (33, 19), (34, 21), (35, 22), (17, 31), (18, 32), (19, 33), (21, 34), (22, 35)]

        pred_instances, label_list, mask_list = list(), list(), list()
        pred_norm = self.logistic(preds)
        k = 0
        for nums in triple_nums:
            pred_instances.append(pred_norm[k: k + nums * nums].cuda().contiguous().view(nums, nums, self.rel_nums))
            label_list.append(labels[k: k + nums * nums].cuda().contiguous().view(nums, nums, self.rel_nums))
            mask_list.append(label_masks[k: k + nums * nums].cuda().contiguous().view(nums, nums, 1))
            k = nums * nums

        pair_nums = torch.sum(label_masks) * self.rule_nums
        label_nums = 0
        for input, label, mask in zip(pred_instances, label_list, mask_list):
            for idx in symmetric_rule_idx:
                ls1, ls2, n = self.symmetric_rule(input, label, mask, idx)
                loss_p = loss_p + ls1
                loss_n = loss_n + ls2
                label_nums += n

            for head_idx, body_idx in inverse_rule_idx:
                ls1, ls2, n = self.inverse_rule(input, label, mask, head_idx, body_idx)
                loss_p = loss_p + ls1
                loss_n = loss_n + ls2
                label_nums += n

        loss_p = loss_p / (label_nums + eps)
        loss_n = loss_n / (pair_nums - label_nums)
        return (loss_p + loss_n) / 2


    def symmetric_rule(self, input, labels, mask, index):
        size = input.shape[0]
        zero_vec = torch.zeros(size * size, 1).cuda()
        input_T = input.transpose(0, 1)
        op_slice_t = input_T[:, :, index].contiguous().view(-1, 1)
        op_slice_r = input[:, :, index].view(-1, 1)

        label = labels[:, :, index]
        inv_label = (~label.bool()).float() * mask.squeeze(2)
        nums = torch.sum(label)
        loss_p = torch.sum((torch.max(op_slice_t - op_slice_r, zero_vec)) ** 2 * label.view(-1, 1))
        loss_n = torch.sum((torch.max(op_slice_r - op_slice_t, zero_vec)) ** 2 * inv_label.view(-1, 1))

        return loss_p, loss_n, nums

    def inverse_rule(self, input, labels, mask, head_idx, body_idx):
        size = input.shape[0]
        zero_vec = torch.zeros(size * size, 1).cuda()
        input_T = input.transpose(0, 1)
        op_slice_t = input_T[:, :, head_idx].contiguous().view(-1, 1)
        op_slice_r = input[:, :, body_idx].view(-1, 1)

        label = labels[:, :, body_idx]
        inv_label = (~label.bool()).float() * mask.squeeze(2)
        nums = torch.sum(label)
        loss_p = torch.sum((torch.max(op_slice_t - op_slice_r, zero_vec)) ** 2 * label.view(-1, 1))
        loss_n = torch.sum((torch.max( -op_slice_t + op_slice_r, zero_vec)) ** 2 * inv_label.view(-1, 1))

        return loss_p, loss_n, nums

    def transtive_rule(self, input, labels, mask, rel_1, rel_2, rel_3):
        op_slice_r = input[:, :, rel_3].unsqueeze(2)
        op_slice_1 = input[:, :, rel_1]
        op_slice_2 = input[:, :, rel_2]

        label = (labels == rel_3).float() * mask

        logic_output = (op_slice_1[:, :, None] + op_slice_2[:, None, :]) / 2
        logic_output, _ = torch.max(logic_output, dim=0)
        logic_output = logic_output.unsqueeze(2)
        com_slice = torch.cat([op_slice_r, logic_output], dim=2)
        com_slice1, _ = torch.max(com_slice, dim=2)

        nums = torch.sum(label)
        loss = torch.sum((op_slice_r.view(-1, 1) - com_slice1.contiguous().view(-1, 1)) ** 2 * label.view(-1, 1))
        return loss, nums