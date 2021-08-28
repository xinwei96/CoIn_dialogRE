import os
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.utils.rnn as rnn
from tqdm import tqdm

from data_utils import *


class REDataset(Dataset):

    def __init__(self, data, bert_tokenizer, type2id, rel_num, offset, data_path):
        self.rel_nums = rel_num
        self.offset = offset
        self.type2id = type2id
        if not os.path.isfile(data_path):
            self.data = self.get_feat(data, bert_tokenizer)
            with open(data_path, 'wb') as f:
                pickle.dump(self.data, f)
            print("Preprocessed data are saved in %s" % data_path)
        else:
            with open(data_path, 'rb') as f:
                self.data = pickle.load(f)


    def __len__(self):
        return len(self.data)

    def get_feat(self, raw_data, bert_tokenizer):
        data = list()
        k = 0
        maxx = 0
        for instance in tqdm(raw_data):
            tokens = bert_tokenizer.encode(instance['dialogue'])
            special_tokens_mask = tokens.special_tokens_mask

            for i, token_id in enumerate(tokens.ids):
                if token_id in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
                    special_tokens_mask[i] = token_id

            if len(tokens.ids) > 512:
                tokens_ids_w1 = torch.LongTensor(tokens.ids[0: 512])
                tokens_ids2 = [101] + tokens.ids[self.offset: ]
                tokens_ids_w2 = torch.LongTensor(tokens_ids2)
            else:
                tokens_ids_w1 = torch.LongTensor(tokens.ids[0: ])
                tokens_ids_w2 = torch.LongTensor([])

            utters_bound, utters_id, speaker_utters = find_utters_boundary(special_tokens_mask)

            dialogue_lens = len(tokens.ids)
            utters_nums = len(utters_bound)
            entities_nums = len(instance['entity2id'])
            if entities_nums == 0:
                print('non entity')
                continue

            entities_dict = wrap_entities(tokens.tokens, utters_id, instance['entity2id'], instance['entity_type'], bert_tokenizer)
            entity_type_ids = get_entity_type_embedding(entities_dict, dialogue_lens, self.type2id)


            labels, labels_mask = labels_tensor(instance['triples'], entities_nums, self.rel_nums)

            utters_position = get_utters_position(utters_bound, dialogue_lens)

            utterence_mask = windows_attention_mask(utters_bound, 1, dialogue_lens)
            local_context_mask = windows_attention_mask(utters_bound, 3, dialogue_lens)
            wide_context_mask = windows_attention_mask(utters_bound, 5, dialogue_lens)
            speaker_mask = speaker_attention_mask(speaker_utters, dialogue_lens)

            if len(utters_bound) > maxx:
                maxx = len(utters_bound)
            data.append({'item': k,
                         'tokens': tokens.tokens,
                         'tokens_ids_w1': tokens_ids_w1,
                         'w1_lens': len(tokens_ids_w1),
                         'tokens_ids_w2': tokens_ids_w2,
                         'w2_lens': len(tokens_ids_w2),
                         'entity_type_ids': entity_type_ids,
                         'utterence_mask': utterence_mask,
                         'local_context_mask': local_context_mask,
                         'wide_context_mask':wide_context_mask,
                         'speaker_mask': speaker_mask,
                         'utters_position': utters_position,
                         'labels': labels,
                         'labels_mask': labels_mask,
                         'entities_dict': entities_dict,
                         'dialogue_lens': dialogue_lens,
                         'utters_nums': utters_nums,
                         'speaker_nums': len(speaker_utters),
                         'entities_nums': entities_nums,
                         'triples': instance['triples'],
                         'id2entity': instance['id2entity']
                         })
            k += 1
        print("MAX Num. of Utterance: %d" % maxx)
        return data

    def __getitem__(self, item):
        return self.data[item]


class REDataloader(object):

    def __init__(self, dataset, args, batch_size, is_shuffle=False, num_workers=0, isEval=False, drop_last=True, sampler=None):
        self.batch_size = batch_size
        self.isEval = isEval
        self.args = args
        if sampler is None:
            self.dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=is_shuffle,
                                        num_workers=num_workers, collate_fn=self.collate_fn, drop_last=drop_last)
        else:
            self.dataloader = DataLoader(dataset=dataset, batch_size=batch_size, # shuffle=is_shuffle,
                                         num_workers=num_workers, collate_fn=self.collate_fn, drop_last=drop_last, sampler=sampler)

    def get_dataloader(self):
        return self.dataloader

    def collate_fn(self, batch):
        batch.sort(key=lambda data: data['dialogue_lens'], reverse=True)

        # padding input_ids and mask of the first window
        tokens_ids_w1 = [data['tokens_ids_w1'] for data in batch]
        w1_lens = [data['w1_lens'] for data in batch]
        tokens_mask_w1 = [torch.ones(w, dtype=torch.float, requires_grad=False) for w in w1_lens]

        tokens_ids_w2 = [data['tokens_ids_w2'] for data in batch]
        w2_lens = [data['w2_lens'] for data in batch]
        tokens_mask_w2 = [torch.ones(w, dtype=torch.float, requires_grad=False) for w in w2_lens]

        entity_type_ids = [data['entity_type_ids'] for data in batch]

        dialogue_lens = [data['dialogue_lens'] for data in batch]

        tokens_ids_w1 = rnn.pad_sequence(tokens_ids_w1, batch_first=True, padding_value=0)
        tokens_mask_w1 = rnn.pad_sequence(tokens_mask_w1, batch_first=True, padding_value=0)

        entity_type_ids = rnn.pad_sequence(entity_type_ids, batch_first=True, padding_value=0)

        if (torch.Tensor(dialogue_lens) > 512).sum() == 0:
            tokens_ids_w2 = None
            tokens_mask_w2 = None
        else:
            tokens_ids_w2 = rnn.pad_sequence(tokens_ids_w2, batch_first=True, padding_value=0)
            tokens_mask_w2 = rnn.pad_sequence(tokens_mask_w2, batch_first=True, padding_value=0)

        labels = [data['labels'] for data in batch]
        labels = torch.cat(labels, dim=0)
        labels_mask = [data['labels_mask'] for data in batch]
        labels_mask = torch.cat(labels_mask, dim=0)

        entity_map = [data['entities_dict'] for data in batch]

        max_dialogue_size = max(dialogue_lens)
        padding_mask = [torch.zeros((lens, max_dialogue_size - lens), dtype=bool) for lens in dialogue_lens]

        utterence_mask = [torch.cat([data['utterence_mask'], padding_mask[i]], dim=1) for i, data in enumerate(batch)]
        local_context_mask = [torch.cat([data['local_context_mask'], padding_mask[i]], dim=1) for i, data in enumerate(batch)]
        wide_context_mask = [torch.cat([data['wide_context_mask'], padding_mask[i]], dim=1) for i, data in enumerate(batch)]
        speaker_mask = [torch.cat([data['speaker_mask'], padding_mask[i]], dim=1) for i, data in enumerate(batch)]

        utterence_mask = rnn.pad_sequence(utterence_mask, batch_first=True, padding_value=0)
        local_context_mask = rnn.pad_sequence(local_context_mask, batch_first=True, padding_value=0)
        wide_context_mask = rnn.pad_sequence(wide_context_mask, batch_first=True, padding_value=0)
        speaker_mask = rnn.pad_sequence(speaker_mask, batch_first=True, padding_value=0)

        utters_position =[data['utters_position'] for data in batch]
        utters_position = rnn.pad_sequence(utters_position, batch_first=True, padding_value=0)

        triples = [data['triples'] for data in batch]
        data_ids = [data['item'] for data in batch]
        tokens = [data['tokens'] for data in batch]

        id2entity = [data['id2entity'] for data in batch]
        d = {}
        d['ids'] = data_ids
        d['tokens'] = tokens
        d['tokens_ids_1'] = tokens_ids_w1
        d['tokens_ids_2'] = tokens_ids_w2
        d['tokens_mask_1'] = tokens_mask_w1
        d['tokens_mask_2'] = tokens_mask_w2
        d['entity_type_ids'] = entity_type_ids
        d['utterence_mask'] = utterence_mask
        d['local_context_mask'] = local_context_mask
        d['wide_context_mask'] = wide_context_mask
        d['speaker_aware_mask'] = speaker_mask
        d['utterence_position'] = utters_position
        d['labels'] = labels
        d['labels_mask'] = labels_mask
        d['entity_map'] = entity_map
        d['max_seq_lens'] = max_dialogue_size
        d['dialogue_lens'] = dialogue_lens
        d['triples'] = triples
        d['id2entity'] = id2entity

        return d
