import argparse
import json
import logging
import numpy as np
import os
import random
import torch
import time
import torch.nn as nn
from transformers import AdamW
from tokenizers import BertWordPieceTokenizer
import torch.distributed as dist
import torch.nn.functional as F

from embeddings import TypeEmbeddings
from model import DialogRE
from model import LogicReg
from data_utils import *
from dataset import REDataset, REDataloader
from tools import init_logger, calculate_metric, set_seed, print_message

torch.distributed.init_process_group(backend="nccl")
local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)


def train(args):
    if local_rank == 0:
        timestamp = time.time()
        logger = init_logger(os.path.join(args.log_path, str(timestamp) + ".txt"))
    else:
        logger = None

    print_message(args, logger, local_rank)

    device = torch.device(args.device_name)

    bert_tokenizer = BertWordPieceTokenizer(os.path.join(args.bert_path, 'vocab.txt'), lowercase=True)
    trn_data, dev_data, tst_data, speakers = load_dataset(args)

    # add special tokens
    bert_tokenizer.add_special_tokens([s for s in speakers])

    type_embeddings = TypeEmbeddings(args.type_dict_path, args.bert_output_size, is_freeze=False)
    type2ids = type_embeddings.get_type2id()

    ds_trn = REDataset(trn_data, bert_tokenizer, type2ids, args.rel_nums, args.offset, "../data/train_data.p")
    ds_dev = REDataset(dev_data, bert_tokenizer, type2ids, args.rel_nums, args.offset, "../data/dev_data.p")
    ds_tst = REDataset(tst_data, bert_tokenizer, type2ids, args.rel_nums, args.offset, "../data/test_data.p")

    train_sampler = torch.utils.data.distributed.DistributedSampler(ds_trn)

    dl_trn = REDataloader(ds_trn, args, args.batch_size, is_shuffle=True, sampler=train_sampler).get_dataloader()
    dl_dev = REDataloader(ds_dev, args, args.eval_batch_size, isEval=True, is_shuffle=False,
                          drop_last=False).get_dataloader()
    dl_tst = REDataloader(ds_tst, args, args.test_batch_size, isEval=True, is_shuffle=False,
                          drop_last=False).get_dataloader()

    model = DialogRE(args, type_embeddings).to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)
    logic_reg = LogicReg(args.rel_nums, args.rule_nums).to(device)
    label_criterion = nn.BCEWithLogitsLoss(reduction='none')

    bert_param_ids = list(map(id, model.module.bert_encoder.parameters()))
    base_params = filter(lambda p: p.requires_grad and id(p) not in bert_param_ids, model.parameters())

    optimizer = AdamW([
        {'params': model.module.bert_encoder.parameters(), 'lr': args.ft_lr},
        {'params': base_params, 'weight_decay': args.weight_decay}
    ], lr=args.lr)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.t_max)

    max_eval_f1 = 0.
    max_test_f1 = 0.
    optimal_model_path = args.optimal_model_path

    for ep in range(1, args.epoch_nums + 1):
        print_message('Start Training: %d' % ep, logger, local_rank)
        model.train()
        train_sampler.set_epoch(ep)
        start_time = time.time()

        loss_per_batch, loss_per_epoch, time_per_batch = 0., 0., 0.
        logic_loss_per_batch, logic_loss_per_epoch = 0., 0.

        nums = 0
        for _, batch in enumerate(dl_trn):
            nums += 1
            start_time_per_batch = time.time()

            optimizer.zero_grad()

            preds = model(batch)
            labels = batch['labels'].cuda()
            labels_mask = batch['labels_mask'].cuda()
            entity_map = batch['entity_map']

            triple_nums = list()
            pair_nums = 0
            for entities in entity_map:
                pair_nums += len(entities) ** 2
                triple_nums.append(len(entities))

            logic_loss = logic_reg(preds, labels, labels_mask, triple_nums)

            label_loss = torch.sum(label_criterion(preds, labels) * labels_mask) / (
                    args.rel_nums * torch.sum(labels_mask) + 1e-15)

            loss_per_batch += label_loss.item()
            loss_per_epoch += label_loss.item()
            logic_loss_per_batch += logic_loss.item()
            logic_loss_per_epoch += logic_loss.item()

            loss = (1 - args.sigma) * label_loss + args.sigma * logic_loss
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
            optimizer.step()
            scheduler.step(ep)

            time_per_batch += (time.time() - start_time_per_batch)
            if nums % args.report_every_batch == 0:
                message = 'Training %d batches, average label, logicReg loss is: %f, %f per batch, consuming time: %f \'s per batch' % \
                          (nums, loss_per_batch / args.report_every_batch,
                           logic_loss_per_batch / args.report_every_batch,
                           time_per_batch / args.report_every_batch)
                print_message(message, logger, local_rank)
                loss_per_batch, logic_loss_per_batch, time_per_batch = 0., 0., 0.

        consuming_time = time.time() - start_time
        ave_loss = loss_per_epoch / nums

        print_message('-' * 90, logger, local_rank)
        print_message("Training finished: %d epoch, average loss: %f, time: %4f." % (ep, ave_loss, consuming_time), logger, local_rank)
        print_message('-' * 90, logger, local_rank)

        # dev
        P, R, F = dev(args, dl_dev, model, label_criterion, logger)
        _, _, test_F = dev(args, dl_tst, model, label_criterion, logger,test=True)
        if F > max_eval_f1:
            max_eval_f1 = F
            max_test_f1 = test_F
            optimal_model_path = args.output_path + str(ep) + '_' + '.pkl'
            torch.save(model.module.state_dict(), optimal_model_path)
            print_message('Saving the optimal model at %s' % optimal_model_path, logger, local_rank)

        print_message("\n\n", logger, local_rank)

    # test
    print_message('The best model is saved at %s' % optimal_model_path, logger, local_rank)
    print_message('Its f1 score in dev set is %f' % max_eval_f1, logger, local_rank)
    print_message('Its f1 score in test set is %f' % max_test_f1, logger, local_rank)
    print_message('\n\n', logger, local_rank)


def dev(args, dev_dl, model, label_criterion, logger, test=False):
    model.eval()
    model.is_test = True
    start_time = time.time()

    pred_nums, target_nums, correct_nums = 0., 0., 0.  # TP+FP, TP+FN, TP

    total_eval_loss = 0.

    pred_list = list()
    num = 0
    with torch.no_grad():
        for _, batch in enumerate(dev_dl):
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
            a = preds >= args.threshold
            pred_nums += torch.sum(a)
            correct_nums += torch.sum((a == labels.bool()) * (labels > 0))

            if test:
                id2entity = batch['id2entity'][0]
                entity_nums = len(batch['entity_map'][0])
                triples = batch['triples']
                label_d = list()
                triples = triples[0]
                for k in triples:
                    if 37 in triples[k]:
                        continue
                    label_d.append({"HeadId": k[0], "TailId": k[1], "Head": id2entity[k[0]], "Tail": id2entity[k[1]],"Rel": triples[k]})

                preds = preds.contiguous().view(entity_nums, entity_nums, args.rel_nums)
                preds_index = torch.nonzero(preds >= args.threshold)
                preds_d = dict()
                for p in preds_index:
                    bound = str((p[0].item(), p[1].item()))
                    if bound not in preds_d:
                        preds_d.setdefault(bound, {"HeadId": p[0].item(), "TailId": p[1].item(), "Head": id2entity[p[0].item()], "Tail": id2entity[p[1].item()], "rel": [p[2].item() + 1]})
                    else:
                        preds_d[bound]["rel"].append(p[2].item() + 1)

                pred_l = list()
                for k in preds_d:
                    pred_l.append(preds_d[k])

                pred_list.append({"labels": label_d, "preds": pred_l})

    correct_nums, pred_nums, target_nums = correct_nums.item(), pred_nums.item(), target_nums.item()
    precision, recall, f1 = calculate_metric(correct_nums, pred_nums, target_nums)


    print_message('-' * 90, logger, local_rank)
    print_message('Evaluation Finished, ave_loss: %f, time: %f' % (total_eval_loss / num, time.time() - start_time), logger, local_rank)
    print_message('model: Precision: %f, Recall: %f, F1 scores: %f' % (precision, recall, f1), logger, local_rank)
    print_message('-' * 90, logger, local_rank)

    if test:
        preds = json.dumps(pred_list, ensure_ascii=False)
        with open(args.preds_output_path, 'w+', encoding='utf-8') as f:
            f.write(preds)
    model.is_test = False

    return precision, recall, f1


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='train', help='train | test')
    parser.add_argument('--train_path', type=str, default='../data/train.json')
    parser.add_argument('--dev_path', type=str, default='../data/dev.json')
    parser.add_argument('--test_path', type=str, default='../data/test.json')
    parser.add_argument('--bert_path', type=str, default='')
    parser.add_argument('--lower', dest='lower', action='store_true', default=True, help='Lowercase all words.')
    parser.add_argument('--device_name', type=str, default='cuda')

    parser.add_argument('--max_input_lens', default=512, type=int)

    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--ft_lr', type=float, default=2e-5, help='learning rate for fine tune the pretrained BERT')
    parser.add_argument('--weight_decay', type=float, default=1e-7)

    parser.add_argument('--epoch_nums', type=int, default=60, help='Number of total training epochs.')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size per gpu')
    parser.add_argument('--eval_batch_size', type=int, default=2, help='batch size of dev')
    parser.add_argument('--test_batch_size', type=int, default=1, help='batch size of test')
    parser.add_argument('--save_dir', type=str, default='../model/', help='Root dir for saving models.')
    parser.add_argument('--log_path', type=str, default='../logs/', help='dir for log files')

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--optimal_model_path', type=str, help='Path of the optimal model', default='./model/best.pkl')

    parser.add_argument('--handle_abbr', type=bool, default=True)

    parser.add_argument('--type_dict_path', type=str, default='../data/entity_type_id.json')
    parser.add_argument('--num_heads', type=int, default=12)
    parser.add_argument('--bert_output_size', type=int, default=768)
    parser.add_argument('--max_utter_nums', type=int, default=42)
    parser.add_argument('--max_seq_lens', type=int, default=725)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--preds_output_path', type=str, default='./results.json')

    parser.add_argument('--rel_nums', type=int, default=36)
    parser.add_argument('--rule_nums', type=int, default=24)
    parser.add_argument('--offset', type=int, default=256, help="offset of slide windows for sequences longer than 512")

    parser.add_argument('--dropout_rate', type=float, default=0.2, help="dropout rate for classifier [default: 0.2]")

    parser.add_argument('--report_every_batch', type=int, default=50)
    parser.add_argument('--output_path', type=str, default='../model/')
    parser.add_argument('--t_max', type=int, default=10)
    parser.add_argument('--sigma', type=float, default=0.1)

    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')

    args = parser.parse_args()
    set_seed(args.seed)
    if args.mode == 'train':
        train(args)


if __name__ == '__main__':
    main()
