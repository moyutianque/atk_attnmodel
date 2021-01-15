# coding=utf-8
# Copyleft 2019 project LXRT.

import os
import collections
import sys

import torch
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

from param import args
from tasks.gqa_model import GQAModel
from tasks.gqa_data import GQADataset, GQATorchDataset, GQAEvaluator

DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')


def get_tuple(splits: str, bs:int, shuffle=False, drop_last=False) -> DataTuple:
    dset = GQADataset(splits)
    tset = GQATorchDataset(dset)
    evaluator = GQAEvaluator(dset)
    data_loader = DataLoader(
        tset, batch_size=bs,
        shuffle=shuffle, num_workers=args.num_workers,
        drop_last=drop_last, collate_fn=tset.collate_fn
    )

    return DataTuple(dataset=dset, loader=data_loader, evaluator=evaluator)

def write_log(out_path, log_str):
    with open(out_path + "/log.log", 'a') as f:
        f.write(log_str)
        f.flush()

class GQA:
    def __init__(self):
        self.train_tuple = get_tuple(
            args.train, bs=args.batch_size, shuffle=True, drop_last=True
        )

        print(f'Avaliable number of GPU: {torch.cuda.device_count()}', file=sys.stderr)
        print(f'Use multiGPU: {args.multiGPU}', file=sys.stderr)
        print(f'Batch Size: {args.batch_size}', file=sys.stderr)

        if args.valid != "":
            valid_bsize = 2048 if args.multiGPU else 1024
            self.valid_tuple = get_tuple(
                args.valid, bs=valid_bsize,
                shuffle=False, drop_last=False
            )
        else:
            self.valid_tuple = None

        self.model = GQAModel(self.train_tuple.dataset.num_answers)

        # Load pre-trained weights
        if args.load_lxmert is not None:
            self.model.lxrt_encoder.load(args.load_lxmert)

        # GPU options
        self.model = self.model.cuda()
        if args.multiGPU:
            self.model.lxrt_encoder.multi_gpu()

        # Losses and optimizer
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mce_loss = nn.CrossEntropyLoss(ignore_index=-1)
        if 'bert' in args.optim:
            batch_per_epoch = len(self.train_tuple.loader)
            t_total = int(batch_per_epoch * args.epochs)
            print("Total Iters: %d" % t_total)
            from lxrt.optimization import BertAdam
            if args.ans_head_only:
                self.optim = BertAdam(list(self.model.logit_fc.parameters()),
                                  lr=args.lr,
                                  warmup=args.warmup,
                                  t_total=t_total)
            else:
                self.optim = BertAdam(list(self.model.parameters()),
                                  lr=args.lr,
                                  warmup=args.warmup,
                                  t_total=t_total)
        else:
            self.optim = args.optimizer(list(self.model.parameters()), args.lr)

        self.output = args.output
        os.makedirs(self.output, exist_ok=True)

    def train(self, train_tuple, eval_tuple):
        dset, loader, evaluator = train_tuple
        iter_wrapper = (lambda x: tqdm(x, total=len(loader))) if args.tqdm else (lambda x: x)

        # ADD (Zehao Wang @ Jan 13 2021) mid validation two time per epoch
        validation_gap = (len(loader) // 3) + 2

        best_valid = 0.
        for epoch in range(args.epochs):
            quesid2ans = {}
            for i, (ques_id, feats, boxes, sent, others) in iter_wrapper(enumerate(loader)):

                target = others['target']
                self.model.train()
                self.optim.zero_grad()

                feats, boxes, target = feats.cuda(), boxes.cuda(), target.cuda()
                logit = self.model(feats, boxes, sent, args.ans_head_only)
                assert logit.dim() == target.dim() == 2
                if args.mce_loss:
                    max_value, target = target.max(1)
                    loss = self.mce_loss(logit, target) * logit.size(1)
                else:
                    loss = self.bce_loss(logit, target)
                    loss = loss * logit.size(1)

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optim.step()

                score, label = logit.max(1)

                for qid, l in zip(ques_id, label.cpu().numpy()):
                    ans = dset.label2ans[l]
                    quesid2ans[qid] = ans

                # (ADD (Zehao @ Jan 11 2021): intermediate out
                if self.valid_tuple is not None and i % validation_gap == validation_gap-1:  # Do Validation
                    tmp_log_str = '\n'
                    valid_score = self.evaluate(eval_tuple)
                    if valid_score > best_valid:
                        best_valid = valid_score
                        self.save("BEST")

                    tmp_log_str += "Epoch %d/%d: Valid %0.2f\n" % (epoch, i, valid_score * 100.) + \
                                   "Epoch %d/%d: Best %0.2f\n" % (epoch, i, best_valid * 100.)
                    print(tmp_log_str)
                    write_log(self.output, tmp_log_str)

            log_str = "\nEpoch %d: Train %0.2f\n" % (epoch, evaluator.evaluate(quesid2ans) * 100.)

            if self.valid_tuple is not None:  # Do Validation
                valid_score = self.evaluate(eval_tuple)
                if valid_score > best_valid:
                    best_valid = valid_score
                    self.save("BEST")

                log_str += "Epoch %d: Valid %0.2f\n" % (epoch, valid_score * 100.) + \
                           "Epoch %d: Best %0.2f\n" % (epoch, best_valid * 100.)

            print(log_str, end='')
            write_log(self.output, log_str)

        self.save("LAST")

    def predict(self, eval_tuple: DataTuple, dump=None):
        self.model.eval()
        dset, loader, evaluator = eval_tuple
        quesid2ans = {}
        for i, datum_tuple in enumerate(tqdm(loader)):
            ques_id, feats, boxes, sent = datum_tuple[:4]   # avoid handling target
            with torch.no_grad():
                feats, boxes = feats.cuda(), boxes.cuda()
                logit = self.model(feats, boxes, sent)
                score, label = logit.max(1)
                for qid, l in zip(ques_id, label.cpu().numpy()):
                    ans = dset.label2ans[l]
                    quesid2ans[qid] = ans
        if dump is not None:
            evaluator.dump_result(quesid2ans, dump)
        return quesid2ans

    def evaluate(self, eval_tuple: DataTuple, dump=None):
        dset, loader, evaluator = eval_tuple
        quesid2ans = self.predict(eval_tuple, dump)
        return evaluator.evaluate(quesid2ans)

    @staticmethod
    def oracle_score(data_tuple):
        dset, loader, evaluator = data_tuple
        quesid2ans = {}
        iter_wrapper = (lambda x: tqdm(x, total=len(loader))) if args.tqdm else (lambda x: x)
        for i, (ques_id, feats, boxes, sent, others) in iter_wrapper(enumerate(loader)):
            target = others['target']
            _, label = target.max(1)
            for qid, l in zip(ques_id, label.cpu().numpy()):
                ans = dset.label2ans[l]
                quesid2ans[qid] = ans
        return evaluator.evaluate(quesid2ans)

    def save(self, name):
        torch.save(self.model.state_dict(),
                   os.path.join(self.output, "%s.pth" % name))

    def load(self, path):
        print("Load model from %s" % path)
        state_dict = torch.load("%s.pth" % path)
        for key in list(state_dict.keys()):
            if '.module' in key:
                state_dict[key.replace('.module', '')] = state_dict.pop(key)
        self.model.load_state_dict(state_dict, strict=False)


if __name__ == "__main__":
    # Build Class
    gqa = GQA()

    # Load Model
    if args.load is not None:
        gqa.load(args.load)

    # Test or Train
    if args.test is not None:
        print('Program run in TEST mode')
        args.fast = args.tiny = False       # Always loading all data in test
        if 'submit' in args.test:
            gqa.predict(
                get_tuple(args.test, bs=args.batch_size,
                          shuffle=False, drop_last=False),
                dump=os.path.join(args.output, 'submit_predict.json')
            )
        if 'testdev' in args.test:
            result = gqa.evaluate(
                get_tuple('testdev', bs=args.batch_size,
                          shuffle=False, drop_last=False),
                dump=os.path.join(args.output, 'testdev_predict.json')
            )
            print(result)
        elif 'valid' in args.test:
            result = gqa.evaluate(
                get_tuple('valid', bs=args.batch_size,
                          shuffle=False, drop_last=False),
                dump=os.path.join(args.output, 'val_predict.json')
            )
            print(result)
    else:
        print('Program run in TRAIN mode')
        print('Splits in Train data:', gqa.train_tuple.dataset.splits)
        if gqa.valid_tuple is not None:
            print('Splits in Valid data:', gqa.valid_tuple.dataset.splits)
        else:
            print("DO NOT USE VALIDATION")
        gqa.train(gqa.train_tuple, gqa.valid_tuple)



