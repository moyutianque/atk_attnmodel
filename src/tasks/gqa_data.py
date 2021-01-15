# coding=utf-8
# Copyleft 2019 project LXRT.

import json

import numpy as np
import torch
from torch.utils.data import Dataset

from param import args
from utils import load_obj_h5, get_imgids_h5
from lxrt.tokenization import BertTokenizer
from torch.utils.data._utils.collate import default_collate
import os
# Load part of the dataset for fast checking.
# Notice that here is the number of images instead of the number of data,
# which means all related data to the images would be used.
TINY_IMG_NUM = 512
FAST_IMG_NUM = 5000


class GQADataset:
    """
    A GQA data example in json file:
    {
        "img_id": "2375429",
        "label": {
            "pipe": 1.0
        },
        "question_id": "07333408",
        "sent": "What is on the white wall?"
    }
    """
    def __init__(self, splits: str):
        self.name = splits
        self.splits = splits.split(',')

        # Loading datasets to data
        self.data = []
        for split in self.splits:
            self.data.extend(json.load(open("data/gqa/annotations/%s.json" % split)))
        print("Load %d data from split(s) %s." % (len(self.data), self.name))

        # List to dict (for evaluation and others)
        self.id2datum = {
            datum['question_id']: datum
            for datum in self.data
        }

        # Answers
        self.ans2label = json.load(open("data/gqa/trainval_ans2label.json"))
        self.label2ans = json.load(open("data/gqa/trainval_label2ans.json"))
        assert len(self.ans2label) == len(self.label2ans)
        for ans, label in self.ans2label.items():
            assert self.label2ans[label] == ans

    @property
    def num_answers(self):
        return len(self.ans2label)

    def __len__(self):
        return len(self.data)


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids


def convert_sents_to_features(sents, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (i, sent) in enumerate(sents):

        # ADD sent
        sent = "let's say, " + sent.strip()

        tokens = tokenizer.tokenize(sent.strip())

        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens) > max_seq_length - 2:
            tokens = tokens[:(max_seq_length - 2)]

        # Keep segment id which allows loading BERT-weights.
        tokens = ["[CLS]"] + tokens + ["[SEP]"]

        segment_ids = [0] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids))
    return features


class GQATorchDataset(Dataset):
    MAX_SEQ_LENGTH = 20

    def __init__(self, dataset: GQADataset, requirements=None):
        super().__init__()
        self.raw_dataset = dataset

        self.requirements = requirements
        if self.requirements is None:
            self.requirements = dict()

        # Loading detection features to img_data
        # Since images in train and valid both come from Visual Genome,
        # buffer the image loading to save memory.
        img_data = []
        if 'testdev' in dataset.splits or 'testdev_all' in dataset.splits:     # Always loading all the data in testdev
            # ADD (Zehao Wang @ Jan 10 2021): set h5name
            self.h5name = '/esat/jade/tmp/zwang/dataset/GQA/GQA_h5/gqa_testdev_obj36.h5'
            # ENDMODIFY
        else:
            # ADD (Zehao Wang @ Jan 10 2021): set h5name
            self.h5name = '/esat/jade/tmp/zwang/dataset/GQA/GQA_h5/vg_gqa_obj36.h5'
            # ENDMODIFY

        # ADD (Zehao Wang @ Jan 10 2021): Image ids
        self.img_ids = set(get_imgids_h5(self.h5name, self.raw_dataset.splits))
        # ENDMODIFY

        # Plausible Answers data
        self.plausibleAns = None
        if 'plausibleAns' in self.requirements.keys() and self.requirements['plausibleAns'] == True:
            if os.path.exists(f"/esat/jade/tmp/zwang/dataset/GQA/choices/{dataset.splits[0]}_pa.json"):
                with open(f"/esat/jade/tmp/zwang/dataset/GQA/choices/{dataset.splits[0]}_pa.json", 'r') as f:
                    self.plausibleAns = json.load(f)  # {question_id: plausible choices}

        # Only kept the data with loaded image features
        self.data = []
        from tqdm import tqdm
        for datum in tqdm(self.raw_dataset.data):
            if str(datum['img_id']) in self.img_ids:
                self.data.append(datum)
        if args.tiny:
            self.data = self.data[:FAST_IMG_NUM]
        print("Use %d data in torch dataset" % (len(self.data)))
        print()

        # Using the bert tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased",
            do_lower_case=True
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: int):
        datum = self.data[item]

        img_id = datum['img_id']
        ques_id = datum['question_id']
        ques = datum['sent']

        # Get image info
        # ADD (Zehao Wang @ Jan 10 2021) : online load img features
        img_info = load_obj_h5(self.h5name,img_id)
        # ENDMODIFY
        obj_num = img_info['num_boxes']
        boxes = img_info['boxes'].copy()
        feats = img_info['features'].copy()
        assert len(boxes) == len(feats) == obj_num

        # Normalize the boxes (to 0 ~ 1)
        img_h, img_w = img_info['img_h'], img_info['img_w']
        boxes[:, (0, 2)] /= img_w
        boxes[:, (1, 3)] /= img_h
        np.testing.assert_array_less(boxes, 1+1e-5)
        np.testing.assert_array_less(-boxes, 0+1e-5)

        others = dict()
        # Create target
        if 'label' in datum:
            label = datum['label']
            target = torch.zeros(self.raw_dataset.num_answers)
            for ans, score in label.items():
                if ans in self.raw_dataset.ans2label:
                    target[self.raw_dataset.ans2label[ans]] = score
            others['target'] = target

            if self.plausibleAns is not None:
                if ques_id not in self.plausibleAns.keys():
                    choices = ['yes', 'no']
                else:
                    choices = self.plausibleAns[ques_id]
                target_pa = torch.zeros_like(target)

                choices_label = [self.raw_dataset.ans2label[choice] for choice in choices if
                                 choice in self.raw_dataset.ans2label.keys()]
                if len(choices_label)>0:
                    prob = 1.0 / len(choices_label)
                    for choice in choices_label:
                        target_pa[choice] = prob
                others['target_pa'] = target_pa

        # Deal with requirements
        if 'txt-only' in self.requirements.keys() and self.requirements['txt-only'] == True:
            feats = np.zeros_like(feats)
            boxes = np.zeros_like(boxes)

        return ques_id, feats, boxes, ques, others

    def collate_fn(self, batch):
        # find largest number of boxes in this batch
        elem = batch[0]

        ques_id = [item[0] for item in batch]

        feats = default_collate([item[1] for item in batch])
        boxes = default_collate([item[2] for item in batch])
        # feats = torch.stack([item[1] for item in batch], dim=0)
        # boxes = torch.stack([item[2] for item in batch], dim=0)
        ques = [item[3] for item in batch]

        txt_feats = convert_sents_to_features(ques, max_seq_length=self.MAX_SEQ_LENGTH, tokenizer=self.tokenizer)

        # Deal with data other than standard inputs
        others = dict()
        others['sents'] = ques
        if 'target' in elem[-1].keys():
            target = default_collate([item[4]['target'] for item in batch])
            others['target'] = target

        if 'target_pa' in elem[-1].keys():
            target_pa = default_collate([item[4]['target_pa'] for item in batch])
            others['target_pa'] = target_pa

        return ques_id, feats, boxes, txt_feats, others


class GQAEvaluator:
    def __init__(self, dataset: GQADataset):
        self.dataset = dataset

    def evaluate(self, quesid2ans: dict):
        score = 0.
        for quesid, ans in quesid2ans.items():
            datum = self.dataset.id2datum[quesid]
            label = datum['label']
            if ans in label:
                score += label[ans]
        return score / len(quesid2ans)

    def dump_result(self, quesid2ans: dict, path):
        """
        Dump the result to a GQA-challenge submittable json file.
        GQA json file submission requirement:
            results = [result]
            result = {
                "questionId": str,      # Note: it's a actually an int number but the server requires an str.
                "prediction": str
            }

        :param quesid2ans: A dict mapping question id to its predicted answer.
        :param path: The file path to save the json file.
        :return:
        """
        with open(path, 'w') as f:
            result = []
            for ques_id, ans in quesid2ans.items():
                result.append({
                    'questionId': ques_id,
                    'prediction': ans
                })
            json.dump(result, f, indent=4, sort_keys=True)


