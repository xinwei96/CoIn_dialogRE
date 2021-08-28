import argparse
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from embeddings import TypeEmbeddings
from tokenizers import BertWordPieceTokenizer
from dataset import REDataset, REDataloader
from data_utils import *
from tools import init_logger, calculate_metric, print_message
from model import DialogRE


def evaluate(args):

    timestamp = time.time()
    logger = init_logger(os.path.join(args.log_path, str(timestamp) + ".txt"))
    logger.info(args)
    device = torch.device(args.device_name)
    type_embeddings = TypeEmbeddings(args.type_dict_path, args.bert_output_size, is_freeze=False)
    type2ids = type_embeddings.get_type2id()

    bert_tokenizer = BertWordPieceTokenizer(os.path.join(args.bert_path, 'vocab.txt'), lowercase=True)
    tst_data, speakers = load_testset(args)
    ds_tst = REDataset(tst_data, bert_tokenizer, type2ids, args.rel_nums, args.offset, "../data/test_data.p")
    dl_tst = REDataloader(ds_tst, args, args.test_batch_size, isEval=True, is_shuffle=False,
                          drop_last=False).get_dataloader()
    label_criterion = nn.BCEWithLogitsLoss(reduction='none')

    model = DialogRE(args, type_embeddings)
    model.load_state_dict(torch.load(args.optimal_model_path))
    model = model.to(device)

    model.eval()
    start_time = time.time()
    pred_nums, target_nums, correct_nums = 0., 0., 0.  # TP+FP, TP+FN, TP
    total_eval_loss = 0.
    pred_list = list()

    with torch.no_grad():
        num = 0
        for _, batch in enumerate(dl_tst):
            num += 1
            preds = model(batch)
            labels = batch['labels'].cuda()  # B2 * 36
            labels_mask = batch['labels_mask'].cuda()

            loss = torch.sum(label_criterion(preds, labels) * labels_mask) / (
                    args.rel_nums * torch.sum(labels_mask) + 1e-15)

            total_eval_loss += loss.item()
            preds = F.sigmoid(preds)
            preds = preds * labels_mask  # B2 * 36

            target_nums += torch.sum(labels)
            pred_nums += torch.sum(preds >= args.threshold)
            correct_nums += torch.sum(((preds >= args.threshold) == labels.bool()) * (labels > 0))

            id2entity = batch['id2entity'][0]
            entity_nums = len(batch['entity_map'][0])
            triples = batch['triples']
            label_d = list()
            triples = triples[0]
            for k in triples:
                if 37 in triples[k]:
                    continue
                label_d.append({"HeadId": k[0], "TailId": k[1], "Head": id2entity[k[0]], "Tail": id2entity[k[1]],
                                "Rel": triples[k]})

            preds = preds.contiguous().view(entity_nums, entity_nums, args.rel_nums)
            preds_index = torch.nonzero(preds >= args.threshold)
            preds_d = dict()
            for p in preds_index:
                bound = str((p[0].item(), p[1].item()))
                if bound not in preds_d:
                    preds_d.setdefault(bound,
                                       {"HeadId": p[0].item(), "TailId": p[1].item(), "Head": id2entity[p[0].item()],
                                        "Tail": id2entity[p[1].item()], "rel": [p[2].item() + 1]})
                else:
                    preds_d[bound]["rel"].append(p[2].item() + 1)

            pred_l = list()
            for k in preds_d:
                pred_l.append(preds_d[k])

            pred_list.append({"labels": label_d, "preds": pred_l})

    correct_nums, pred_nums, target_nums = correct_nums.item(), pred_nums.item(), target_nums.item()
    precision, recall, f1 = calculate_metric(correct_nums, pred_nums, target_nums)

    print_message('-' * 90, logger)
    print_message('Evaluation Finished, ave_loss: %f, time: %f' % (total_eval_loss / num, time.time() - start_time),
                  logger)
    print_message('model: Precision: %f, Recall: %f, F1 scores: %f' % (precision, recall, f1), logger)
    print_message('-' * 90, logger)
    print_message('\n\n', logger)

    preds = json.dumps(pred_list, ensure_ascii=False)
    with open(os.path.join(args.preds_output_path, "results_" + str(timestamp) + ".json"), 'w+', encoding='utf-8') as f:
        f.write(preds)

    return precision, recall, f1


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='test', help='only for test')
    parser.add_argument('--test_path', type=str, default='../data/test.json')
    parser.add_argument('--bert_path', type=str, default='/home/longxinwei/bert_model_pytorch2')
    parser.add_argument('--lower', dest='lower', action='store_true', default=True, help='Lowercase all words.')
    parser.add_argument('--device_name', type=str, default='cuda')

    parser.add_argument('--max_input_lens', default=512, type=int)
    parser.add_argument('--bert_output_size', type=int, default=768)
    parser.add_argument('--test_batch_size', type=int, default=1, help='batch size of test')
    parser.add_argument('--optimal_model_path', type=str, help='Path of the optimal model', default='../model/33_.pkl')
    parser.add_argument('--log_path', type=str, default='../logs/', help='dir for log files')

    parser.add_argument('--handle_abbr', type=bool, default=True)

    parser.add_argument('--type_dict_path', type=str, default='../data/entity_type_id.json')
    parser.add_argument('--num_heads', type=int, default=12)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--rel_nums', type=int, default=36)
    parser.add_argument('--offset', type=int, default=256)
    parser.add_argument('--max_utter_nums', type=int, default=42)
    parser.add_argument('--max_seq_lens', type=int, default=725)
    parser.add_argument('--preds_output_path', type=str, default='../model')

    parser.add_argument('--output_path', type=str, default='../model/')

    args = parser.parse_args()

    if args.mode == 'test':
        evaluate(args)


if __name__ == '__main__':
    main()