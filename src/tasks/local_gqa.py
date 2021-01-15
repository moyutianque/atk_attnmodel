# coding=utf-8
# Copyleft 2019 project LXRT.

import os
import collections
import sys

import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader

from param import args
from tasks.gqa_model import GQAModel
from tasks.gqa_data import GQADataset, GQATorchDataset, GQAEvaluator

DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')

# INFO (Zehao Wang @ Jan 14 2021) : Hyperparameter for lossPA
LAMBDA = 1.

def get_tuple(splits: str, bs:int, shuffle=False, drop_last=False, requirements=None) -> DataTuple:
    dset = GQADataset(splits)
    tset = GQATorchDataset(dset, requirements=requirements)
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

requirements = {
    'txt-only': args.txt_only,
    'plausibleAns': True, # INFO (Zehao Wang @ Jan 14 2021) : plausible answer objective
}

# ADD (Zehao Wang @ Jan 14 2021): SoftCrossEntropy for PAO
def cross_entropy_soft(pred, soft_targets):
    return torch.mean(torch.sum(- soft_targets * F.log_softmax(pred, dim=1), 1))
# ENDMODIFY

class GQA:
    def __init__(self):
        self.train_tuple = get_tuple(
            args.train, bs=args.batch_size, shuffle=True, drop_last=True, requirements=requirements
        )

        print(f'Avaliable number of GPU: {torch.cuda.device_count()}', file=sys.stderr)
        print(f'Use multiGPU: {args.multiGPU}', file=sys.stderr)
        print(f'Batch Size: {args.batch_size}', file=sys.stderr)

        if args.valid != "":
            valid_bsize = 2048 if args.multiGPU else 1024
            self.valid_tuple = get_tuple(
                args.valid, bs=valid_bsize,
                shuffle=False, drop_last=False,
                requirements=requirements
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

        write_log(self.output, '===============================\n')
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

                # INFO (Zehao Wang @ Jan 14 2021) : loss of plausible answer target
                target_pa = others['target_pa']
                target_pa = target_pa.cuda()
                logit2 = self.model(torch.zeros_like(feats), torch.zeros_like(boxes), sent)
                assert logit2.dim() == target_pa.dim() == 2
                if args.mce_loss:
                    loss2 = cross_entropy_soft(logit2, target_pa)
                    loss2 = loss2 * logit2.size(1) * LAMBDA
                    loss2.backward()
                else:
                    raise NameError('loss is not correctly set to mce')

                nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optim.step()

                score, label = logit.max(1)

                for qid, l in zip(ques_id, label.cpu().numpy()):
                    ans = dset.label2ans[l]
                    quesid2ans[qid] = ans

                # ADD (Zehao @ Jan 11 2021): intermediate out
                if self.valid_tuple is not None and i % validation_gap == validation_gap-1:  # Do Validation
                    tmp_log_str = '\n'
                    valid_score, valid_score_txtonly = self.evaluate(eval_tuple)
                    if valid_score > best_valid:
                        best_valid = valid_score
                        self.save("BEST")

                    tmp_log_str += "Epoch %d/%d: loss(full) %0.2f loss(txt) %0.2f\n" % (epoch, i, loss.item(), loss2.item()) + \
                            "Epoch %d/%d: Valid %0.2f\n" % (epoch, i, valid_score * 100.) + \
                           "Epoch %d/%d: Valid (txt) %0.2f\n" % (epoch, i, valid_score_txtonly * 100.) + \
                           "Epoch %d: Best %0.2f\n" % (epoch, best_valid * 100.)
                    print(tmp_log_str)
                    write_log(self.output, tmp_log_str)

            log_str = "\nEpoch %d: Train %0.2f\n" % (epoch, evaluator.evaluate(quesid2ans) * 100.)

            if self.valid_tuple is not None:  # Do Validation
                valid_score, valid_score_txtonly = self.evaluate(eval_tuple)
                if valid_score > best_valid:
                    best_valid = valid_score
                    self.save("BEST")

                log_str += "Epoch %d: Valid %0.2f\n" % (epoch, valid_score * 100.) + \
                           "Epoch %d: Valid (txt) %0.2f\n" % (epoch, valid_score_txtonly * 100.) + \
                           "Epoch %d: Best %0.2f\n" % (epoch, best_valid * 100.)

            print(log_str, end='')
            write_log(self.output, log_str)

        self.save("LAST")

    def predict(self, eval_tuple: DataTuple, dump=None):
        self.model.eval()
        dset, loader, evaluator = eval_tuple
        quesid2ans = {}
        quesid2ans_txtonly = {}

        lossPAs = []
        for i, datum_tuple in enumerate(tqdm(loader)):
            ques_id, feats, boxes, sent, others = datum_tuple
            with torch.no_grad():
                feats, boxes = feats.cuda(), boxes.cuda()
                logit = self.model(feats, boxes, sent)
                score, label = logit.max(1)
                for qid, l in zip(ques_id, label.cpu().numpy()):
                    ans = dset.label2ans[l]
                    quesid2ans[qid] = ans

                # ADD (Zehao Wang @ Jan 14 2021) : text-only in test part
                logit2 = self.model(torch.zeros_like(feats), torch.zeros_like(boxes), sent)
                # text only accuracy
                score, label = logit2.max(1)
                for qid, l in zip(ques_id, label.cpu().numpy()):
                    ans = dset.label2ans[l]
                    quesid2ans_txtonly[qid] = ans

                # ADD (Zehao Wang @ Jan 14 2021) : calculate loss for for plausible answer target
                if 'target_pa' in others.keys():
                    target_pa = others['target_pa']
                    target_pa = target_pa.cuda()
                    # text only lossPA
                    assert logit2.dim() == target_pa.dim() == 2
                    if args.mce_loss:
                        loss_pa = cross_entropy_soft(logit2, target_pa)
                        lossPAs.append(loss_pa.item())
                    else:
                        raise NameError('loss is not correctly set to mce')
        if len(lossPAs) > 0:
            print(f'[INFO] The loss for Plausible Answer branch is {sum(lossPAs)/len(lossPAs)}')

        if dump is not None:
            evaluator.dump_result(quesid2ans, dump)
        return quesid2ans, quesid2ans_txtonly

    def evaluate(self, eval_tuple: DataTuple, dump=None):
        dset, loader, evaluator = eval_tuple
        quesid2ans, quesid2ans_txtonly = self.predict(eval_tuple, dump)
        return evaluator.evaluate(quesid2ans), evaluator.evaluate(quesid2ans_txtonly)

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
                          shuffle=False, drop_last=False, requirements=requirements),
                dump=os.path.join(args.output, 'testdev_predict.json')
            )
            print(result)
        elif 'valid' in args.test:
            result = gqa.evaluate(
                get_tuple('valid', bs=args.batch_size,
                          shuffle=False, drop_last=False, requirements=requirements),
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



