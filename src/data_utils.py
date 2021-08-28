import json
import os
import torch

word_pairs = {"he's been": "he has been", "she's been": "she has been", "it's been": "it has been", "it's": "it is", "don't": "do not", "doesn't": "does not", "didn't": "did not", "you'd": "you would",
              "you're": "you are", "you'll": "you will", "i'm": "i am", "they're": "they are", "that's": "that is",
              "what's": "what is", "couldn't": "could not", "i've": "i have", "we've": "we have", "can't": "cannot",
              "i'd": "i would", "i'll": "i will", "aren't": "are not", "isn't": "is not", "wasn't": "was not",
              "weren't": "were not", "won't": "will not", "there's": "there is", "there're": "there are", "he's": "he is", "she's": "she is"
            }


# concatenate utters into a dialogue
def get_dialogue(utters, word_pairs, args):
    dialogue = ''
    speaker_set = set()
    for utter in utters:

        if args.lower: utter = utter.lower()
        break_index = utter.find(':')
        speaker, utter = utter[:break_index], utter[break_index + 1:]
        speaker = ''.join(speaker.split())  # remove white space
        if ',' in speaker:
            speaker = speaker.replace(',', ' and ')

        utter = utter.strip()
        utter = utter.replace('\\','')

        utter = speaker + ' said: ' + utter
        speaker_set.add(speaker)

        if args.handle_abbr:
            for k, v in word_pairs.items():
                utter = utter.replace(k, v)

        dialogue += (utter + ' ')

    return dialogue, speaker


def get_entity_feat(feat_list):
    entity_name_list = list()
    entity_type_list = list()
    triple_list = dict()

    for feat in feat_list:
        x = feat['x'].lower()
        if 'speaker' in x:
            x = x.replace(' ', '')
        y = feat['y'].lower()
        if 'speaker' in y:
            y = y.replace(' ', '')
        if x not in entity_name_list:
            entity_name_list.append(x)
            entity_type_list.append(feat['x_type'])
        if y not in entity_name_list:
            entity_name_list.append(y)
            entity_type_list.append(feat['y_type'])

    entity_to_id = {entity_name_list[i]: i for i in range(len(entity_name_list))}
    id_to_entity = {entity_to_id[k]: k for k in (entity_to_id)}

    for feat in feat_list:
        x = feat['x'].lower()
        if 'speaker' in x:
            x = x.replace(' ', '')
        y = feat['y'].lower()
        if 'speaker' in y:
            y = y.replace(' ', '')
        triple_list.setdefault((entity_to_id[x], entity_to_id[y]), feat['rid'])

    return entity_to_id, id_to_entity, entity_type_list, triple_list


def load_data(filename, args):

    with open(filename, encoding='utf-8') as infile:
        data = json.load(infile)
    D = []
    speaker_set = set()
    for i in range(len(data)):
        utters = data[i][0]
        dialogue, speaker = get_dialogue(utters, word_pairs, args)
        d = dict()
        d['dialogue'] = dialogue
        d['entity2id'], d['id2entity'], d['entity_type'], d['triples'] = get_entity_feat(data[i][1])
        D.append(d)
        speaker_set.add(speaker)

    return D, speaker_set

def load_testset(args):
    print("Load dialogRE test dataset...")
    # load files
    print("loading files...")
    test_data, speakers = load_data(args.test_path, args)

    speaker_special_tokens = set()
    for speaker in speakers:
        ss = speaker.split()
        for s in ss:
            if s != 'and':
                speaker_special_tokens.add(s)

    print("all done.")

    return test_data, speaker_special_tokens

def load_dataset(args):

    print("Load dialogre dataset... (This can take some time)")
    # load files
    print("loading files...")

    train_data, speaker_trn = load_data(args.train_path, args)
    dev_data, speaker_dev = load_data(args.dev_path, args)
    test_data, speaker_tst = load_data(args.test_path, args)

    speakers = set.union(speaker_trn, speaker_dev, speaker_tst)
    speaker_special_tokens = set()
    for speaker in speakers:
        ss = speaker.split()
        for s in ss:
            if s != 'and':
                speaker_special_tokens.add(s)

    print("all done.")

    return train_data, dev_data, test_data, speaker_special_tokens

def find_utters_boundary(special_token_ids):
    boudaries = list()
    speaker_utter = {i - 1: [] for i in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]}

    last_mark = 1
    last_speaker = [1]
    for i, ids in enumerate(special_token_ids):
        if ids in [2, 3, 4, 5, 6, 7, 8, 9, 10,
                   11] and i - last_mark > 4:
            boudaries.append((last_mark, i - 1))
            for s in last_speaker:
                speaker_utter[s].append((last_mark, i - 1))
            last_mark = i
            last_speaker = [ids - 1]
        elif i != 1 and ids in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11] and i - last_mark <= 4:
            last_speaker.append(ids - 1)

    boudaries.append((last_mark, i - 1))
    for s in last_speaker:
        speaker_utter[s].append((last_mark, i - 1))

    speakers = dict()
    for k in speaker_utter:
        if len(speaker_utter[k]) != 0:
            speakers.setdefault(k, speaker_utter[k])

    utters_id = {i: bound for i, bound in enumerate(boudaries)}

    return boudaries, utters_id, speakers


def find_abl_pos(token, token_list):
    idx = 0
    res = list()
    while idx < len(token_list):
        if token[0] == token_list[idx] or token[0] + 's' == token_list[idx]:
            flag = True
            for k in range(1, len(token)):
                idx += 1
                if token[k] != token_list[idx] and token[k] + 's' != token_list[idx]:
                    flag = False
                    break
            if flag:
                res.append((idx - len(token) + 1, idx))
            idx += 1
        else:
            idx += 1
    if len(res) == 0:
        print(token)
    return res


def find_utter_pos(utters_ids, abl_pos):
    res = list()
    for start, end in abl_pos:
        for i in utters_ids:
            a, b = utters_ids[i]
            if start >= a and end <= b:
                res.append(i)

    return res


def wrap_entities(tokens, utters_ids, entity_dict, entity_type, tokenizer):
    entities = dict()
    for entity in entity_dict:
        id = entity_dict[entity]
        type = entity_type[id]
        entity_tokens = tokenizer.encode(entity).tokens
        entity_tokens = entity_tokens[1: len(entity_tokens) - 1]
        abl_pos = find_abl_pos(entity_tokens, tokens)
        utt_pos = find_utter_pos(utters_ids, abl_pos)

        entity_lens = abl_pos[0][1] - abl_pos[0][0] + 1
        pos_idx = []
        for start, _ in abl_pos:
            idx = [i for i in range(start, start + entity_lens)]
            pos_idx.append(idx)

        pos_idx = torch.LongTensor(pos_idx)

        entities.setdefault(id, {'type': type, 'abl_pos': abl_pos, 'utt_pos': utt_pos, 'pos_idx': pos_idx})

    return entities


def windows_attention_mask(utters_list, window_size, dialogue_lens):
    size = dialogue_lens
    res = torch.zeros((size, size), dtype=torch.float)

    for i in range(dialogue_lens):
        res[0][i] = 1   # for [CLS] token
        res[dialogue_lens - 1][i] = 1   # for [SEP] token

    for k, (start, end) in enumerate(utters_list):
        if window_size == 1:  # K=0
            mark_start, mark_end = start, end
            for i in range(start, end + 1):
                for j in range(mark_start, mark_end + 1):
                    res[i][j] = 1
        else:
            if window_size == 3:  # K=1
                mark_start1, mark_end1 = (start, start - 1) if k == 0 else (utters_list[k - 1][0], utters_list[k - 1][1])
                mark_start2, mark_end2 = (end, end - 1) if k == len(utters_list) - 1 else (utters_list[k + 1][0], utters_list[k + 1][1])
            else:   # K=2
                mark_start1, mark_end1 = (start, start - 1) if k in [0, 1] else (utters_list[k - 2][0], utters_list[k - 2][1])
                mark_start2, mark_end2 = (end, end - 1) if len(utters_list) - k in [1, 2] else (utters_list[k + 2][0], utters_list[k + 2][1])

            for i in range(start, end + 1):
                for j in range(mark_start1, mark_end1 + 1):
                    res[i][j] = 1
                for j in range(mark_start2, mark_end2 + 1):
                    res[i][j] = 1

    return res > 0


def speaker_attention_mask(speaker_list, dialogue_lens):
    size = dialogue_lens
    res = torch.zeros((size, size), dtype=torch.float, requires_grad=False)

    for i in range(dialogue_lens):
        res[0][i] = 1  # for [CLS] token
        res[dialogue_lens - 1][i] = 1  # for [SEP] token

    speaker_mask = [torch.zeros(size) for i in range(len(speaker_list))]

    for speaker in speaker_list:
        for start, end in speaker_list[speaker]:
            for i in range(start, end + 1):
                speaker_mask[speaker - 1][i] = 1

    for speaker in speaker_list:
        for start, end in speaker_list[speaker]:
            for i in range(start, end + 1):
                res[i] += speaker_mask[speaker - 1]

    return res > 0


def labels_tensor(triples, entity_nums, relation_nums):
    labels = torch.zeros((entity_nums, entity_nums, relation_nums), dtype=torch.float)
    labels_mask = torch.zeros((entity_nums, entity_nums), dtype=torch.float, requires_grad=False)
    for (s, e), r_list in triples.items():
        for r in r_list:
            if r > 36:  # Ignoring the 'unanswerable' relation, to be in line with the setting of baselines
                continue
            labels[s][e][r - 1] = 1
            labels_mask[s][e] = 1

    return labels.contiguous().view(-1, relation_nums), labels_mask.contiguous().view(-1, 1)


def get_entity_type_embedding(entity_dict, seq_lens, type2id):
    type_ids = torch.zeros(seq_lens, dtype=torch.long)
    for _, entity in entity_dict.items():
        type = type2id[entity['type']]
        abl_pos = entity['abl_pos']
        for a, b in abl_pos:
            for i in range(a, b + 1):
                type_ids[i] = type

    return type_ids


def get_utters_position(utters_list, seq_lens):
    res = torch.zeros(seq_lens, dtype=torch.long)
    for i, (start, end) in enumerate(utters_list):
        k = start
        while k <= end:
            res[k] = i
            k += 1

    return res

