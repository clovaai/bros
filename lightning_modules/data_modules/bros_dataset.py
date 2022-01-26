"""
BROS
Copyright 2022-present NAVER Corp.
Apache License v2.0
"""

import itertools
import json
import os

import numpy as np
import torch
from torch.utils.data.dataset import Dataset

from utils import get_class_names


class BROSDataset(Dataset):
    def __init__(
        self,
        dataset,
        task,
        backbone_type,
        model_head,
        dataset_root_path,
        tokenizer,
        max_seq_length=512,
        mode=None,
    ):
        self.dataset = dataset
        self.task = task
        self.backbone_type = backbone_type
        self.model_head = model_head

        self.dataset_root_path = dataset_root_path
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.mode = mode

        self.pad_token_id = self.tokenizer.vocab["[PAD]"]
        self.cls_token_id = self.tokenizer.vocab["[CLS]"]
        self.sep_token_id = self.tokenizer.vocab["[SEP]"]
        self.unk_token_id = self.tokenizer.vocab["[UNK]"]

        self.examples = self._load_examples()

        if not (self.dataset == "funsd" and self.model_head == "bies"):
            self.class_names = get_class_names(self.dataset_root_path)
            self.class_idx_dic = dict(
                [(class_name, idx) for idx, class_name in enumerate(self.class_names)]
            )

            self.bio_class_names = ["O"]
            for class_name in self.class_names:
                self.bio_class_names.extend([f"B_{class_name}", f"I_{class_name}"])
            self.bio_class_idx_dic = dict(
                [
                    (bio_class_name, idx)
                    for idx, bio_class_name in enumerate(self.bio_class_names)
                ]
            )

    def _load_examples(self):
        examples = []
        with open(
            os.path.join(self.dataset_root_path, f"preprocessed_files_{self.mode}.txt"),
            "r",
            encoding="utf-8",
        ) as fp:
            for line in fp.readlines():
                preprocessed_file = os.path.join(self.dataset_root_path, line.strip())
                examples.append(
                    json.load(open(preprocessed_file, "r", encoding="utf-8"))
                )

        return examples

    def __len__(self):
        return len(self.examples)

    def _getitem_for_funsd_bies(self, idx):
        json_obj = self.examples[idx]

        width = json_obj["meta"]["imageSize"]["width"]
        height = json_obj["meta"]["imageSize"]["height"]

        input_ids = np.array(json_obj["parse"]["input_ids"])
        attention_mask = np.array(json_obj["parse"]["input_mask"])
        labels = np.array(json_obj["parse"]["label_ids"])

        input_ids_check = np.ones(self.max_seq_length, dtype=int) * self.pad_token_id
        bbox = np.zeros((self.max_seq_length, 8), dtype=np.float32)

        input_ids_list = [self.cls_token_id]
        cls_bb = [0.0] * 8
        sep_bb = [1.0] * 8

        bbox_list = [cls_bb]
        for word in json_obj["words"]:
            tokens = word["tokens"]
            input_ids_list.extend(tokens)

            bb = []
            for point in word["boundingBox"]:
                bb.extend([point[0] / width, point[1] / height])

            bbox_list.extend([bb] * len(tokens))

        input_ids_list.append(self.sep_token_id)
        bbox_list.append(sep_bb)

        input_ids_check[: len(input_ids_list)] = input_ids_list
        assert np.sum(input_ids - input_ids_check) == 0

        bbox[: len(bbox_list)] = bbox_list

        if self.backbone_type == "layoutlm":
            bbox = bbox[:, [0, 1, 4, 5]]
            bbox = bbox * 1000
            bbox = bbox.astype(int)

        input_ids = torch.from_numpy(input_ids)
        bbox = torch.from_numpy(bbox)
        attention_mask = torch.from_numpy(attention_mask)
        labels = torch.from_numpy(labels)

        return_dict = {
            "input_ids": input_ids,
            "bbox": bbox,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        return return_dict

    def _getitem_bio(self, idx):
        json_obj = self.examples[idx]

        width = json_obj["meta"]["imageSize"]["width"]
        height = json_obj["meta"]["imageSize"]["height"]

        input_ids = np.ones(self.max_seq_length, dtype=int) * self.pad_token_id
        bbox = np.zeros((self.max_seq_length, 8), dtype=np.float32)
        attention_mask = np.zeros(self.max_seq_length, dtype=int)

        bio_labels = np.zeros(self.max_seq_length, dtype=int)
        are_box_first_tokens = np.zeros(self.max_seq_length, dtype=np.bool_)

        list_tokens = []
        list_bbs = []
        box2token_span_map = []

        cls_bbs = [0.0] * 8

        for word_idx, word in enumerate(json_obj["words"]):
            tokens = word["tokens"]
            bb = word["boundingBox"]
            if len(tokens) == 0:
                tokens.append(self.unk_token_id)

            if len(list_tokens) + len(tokens) > self.max_seq_length - 2:
                break

            box2token_span_map.append(
                [len(list_tokens) + 1, len(list_tokens) + len(tokens) + 1]
            )  # including st_idx
            list_tokens += tokens

            # min, max clipping
            for coord_idx in range(4):
                bb[coord_idx][0] = max(0.0, min(bb[coord_idx][0], width))
                bb[coord_idx][1] = max(0.0, min(bb[coord_idx][1], height))

            bb = list(itertools.chain(*bb))
            bbs = [bb for _ in range(len(tokens))]
            list_bbs.extend(bbs)

        sep_bbs = [width, height] * 4

        # For [CLS] and [SEP]
        list_tokens = (
            [self.cls_token_id]
            + list_tokens[: self.max_seq_length - 2]
            + [self.sep_token_id]
        )
        if len(list_bbs) == 0:
            # When len(json_obj["words"]) == 0 (no OCR result)
            list_bbs = [cls_bbs] + [sep_bbs]
        else:  # len(list_bbs) > 0
            list_bbs = [cls_bbs] + list_bbs[: self.max_seq_length - 2] + [sep_bbs]

        len_list_tokens = len(list_tokens)
        input_ids[:len_list_tokens] = list_tokens
        attention_mask[:len_list_tokens] = 1

        bbox[:len_list_tokens, :] = list_bbs

        # Normalize bbox -> 0 ~ 1
        bbox[:, [0, 2, 4, 6]] = bbox[:, [0, 2, 4, 6]] / width
        bbox[:, [1, 3, 5, 7]] = bbox[:, [1, 3, 5, 7]] / height

        if self.backbone_type == "layoutlm":
            bbox = bbox[:, [0, 1, 4, 5]]
            bbox = bbox * 1000
            bbox = bbox.astype(int)

        # Label
        classes_dic = json_obj["parse"]["class"]
        for class_name in self.class_names:
            if class_name == "O":
                continue
            if class_name not in classes_dic:
                continue

            for word_list in classes_dic[class_name]:
                # At first, connect the class and the first box
                is_first, last_word_idx = True, -1
                for word_idx in word_list:
                    if word_idx >= len(box2token_span_map):
                        break
                    box2token_span_start, box2token_span_end = box2token_span_map[
                        word_idx
                    ]
                    for converted_word_idx in range(
                        box2token_span_start, box2token_span_end
                    ):
                        if converted_word_idx >= self.max_seq_length:
                            break

                        if is_first:
                            bio_labels[converted_word_idx] = self.bio_class_idx_dic[
                                f"B_{class_name}"
                            ]
                            is_first = False
                        else:
                            bio_labels[converted_word_idx] = self.bio_class_idx_dic[
                                f"I_{class_name}"
                            ]

            st_indices, _ = zip(*box2token_span_map)
            st_indices = [
                st_idx for st_idx in st_indices if st_idx < self.max_seq_length
            ]
            are_box_first_tokens[st_indices] = True

        input_ids = torch.from_numpy(input_ids)
        bbox = torch.from_numpy(bbox)
        attention_mask = torch.from_numpy(attention_mask)

        bio_labels = torch.from_numpy(bio_labels)
        are_box_first_tokens = torch.from_numpy(are_box_first_tokens)

        return_dict = {
            "input_ids": input_ids,
            "bbox": bbox,
            "attention_mask": attention_mask,
            "bio_labels": bio_labels,
            "are_box_first_tokens": are_box_first_tokens,
        }

        return return_dict

    def _getitem_spade(self, idx):
        json_obj = self.examples[idx]

        width = json_obj["meta"]["imageSize"]["width"]
        height = json_obj["meta"]["imageSize"]["height"]

        input_ids = np.ones(self.max_seq_length, dtype=int) * self.pad_token_id
        bbox = np.zeros((self.max_seq_length, 8), dtype=np.float32)
        attention_mask = np.zeros(self.max_seq_length, dtype=int)

        itc_labels = np.zeros(self.max_seq_length, dtype=int)
        are_box_first_tokens = np.zeros(self.max_seq_length, dtype=np.bool_)

        # stc_labels stores the index of the previous token.
        # A stored index of max_seq_length (512) indicates that
        # this token is the initial token of a word box.
        stc_labels = np.ones(self.max_seq_length, dtype=np.int64) * self.max_seq_length

        list_tokens = []
        list_bbs = []

        box_to_token_indices = []
        cum_token_idx = 0

        cls_bbs = [0.0] * 8

        for word_idx, word in enumerate(json_obj["words"]):
            this_box_token_indices = []

            tokens = word["tokens"]
            bb = word["boundingBox"]
            if len(tokens) == 0:
                tokens.append(self.unk_token_id)

            if len(list_tokens) + len(tokens) > self.max_seq_length - 2:
                break

            list_tokens += tokens

            # min, max clipping
            for coord_idx in range(4):
                bb[coord_idx][0] = max(0.0, min(bb[coord_idx][0], width))
                bb[coord_idx][1] = max(0.0, min(bb[coord_idx][1], height))

            bb = list(itertools.chain(*bb))
            bbs = [bb for _ in range(len(tokens))]

            for _ in tokens:
                cum_token_idx += 1
                this_box_token_indices.append(cum_token_idx)

            list_bbs.extend(bbs)
            box_to_token_indices.append(this_box_token_indices)

        sep_bbs = [width, height] * 4

        # For [CLS] and [SEP]
        list_tokens = (
            [self.cls_token_id]
            + list_tokens[: self.max_seq_length - 2]
            + [self.sep_token_id]
        )
        if len(list_bbs) == 0:
            # When len(json_obj["words"]) == 0 (no OCR result)
            list_bbs = [cls_bbs] + [sep_bbs]
        else:  # len(list_bbs) > 0
            list_bbs = [cls_bbs] + list_bbs[: self.max_seq_length - 2] + [sep_bbs]

        len_list_tokens = len(list_tokens)
        input_ids[:len_list_tokens] = list_tokens
        attention_mask[:len_list_tokens] = 1

        bbox[:len_list_tokens, :] = list_bbs

        # Normalize bbox -> 0 ~ 1
        bbox[:, [0, 2, 4, 6]] = bbox[:, [0, 2, 4, 6]] / width
        bbox[:, [1, 3, 5, 7]] = bbox[:, [1, 3, 5, 7]] / height

        if self.backbone_type == "layoutlm":
            bbox = bbox[:, [0, 1, 4, 5]]
            bbox = bbox * 1000
            bbox = bbox.astype(int)

        st_indices = [
            indices[0]
            for indices in box_to_token_indices
            if indices[0] < self.max_seq_length
        ]
        are_box_first_tokens[st_indices] = True

        # Label
        classes_dic = json_obj["parse"]["class"]
        for class_name in self.class_names:
            if class_name == "others":
                continue
            if class_name not in classes_dic:
                continue

            for word_list in classes_dic[class_name]:
                is_first, last_word_idx = True, -1
                for word_idx in word_list:
                    if word_idx >= len(box_to_token_indices):
                        break
                    box2token_list = box_to_token_indices[word_idx]
                    for converted_word_idx in box2token_list:
                        if converted_word_idx >= self.max_seq_length:
                            break  # out of idx

                        if is_first:
                            itc_labels[converted_word_idx] = self.class_idx_dic[
                                class_name
                            ]
                            is_first, last_word_idx = False, converted_word_idx
                        else:
                            stc_labels[converted_word_idx] = last_word_idx
                            last_word_idx = converted_word_idx

        input_ids = torch.from_numpy(input_ids)
        bbox = torch.from_numpy(bbox)
        attention_mask = torch.from_numpy(attention_mask)

        itc_labels = torch.from_numpy(itc_labels)
        are_box_first_tokens = torch.from_numpy(are_box_first_tokens)
        stc_labels = torch.from_numpy(stc_labels)

        return_dict = {
            "input_ids": input_ids,
            "bbox": bbox,
            "attention_mask": attention_mask,
            "itc_labels": itc_labels,
            "are_box_first_tokens": are_box_first_tokens,
            "stc_labels": stc_labels,
        }

        return return_dict

    def _getitem_spade_rel(self, idx):
        json_obj = self.examples[idx]

        width = json_obj["meta"]["imageSize"]["width"]
        height = json_obj["meta"]["imageSize"]["height"]

        input_ids = np.ones(self.max_seq_length, dtype=int) * self.pad_token_id
        bbox = np.zeros((self.max_seq_length, 8), dtype=np.float32)
        attention_mask = np.zeros(self.max_seq_length, dtype=int)

        are_box_first_tokens = np.zeros(self.max_seq_length, dtype=np.bool_)
        el_labels = np.ones(self.max_seq_length, dtype=int) * self.max_seq_length

        list_tokens = []
        list_bbs = []
        box2token_span_map = []

        box_to_token_indices = []
        cum_token_idx = 0

        cls_bbs = [0.0] * 8

        for word_idx, word in enumerate(json_obj["words"]):
            this_box_token_indices = []

            tokens = word["tokens"]
            bb = word["boundingBox"]
            if len(tokens) == 0:
                tokens.append(self.unk_token_id)

            if len(list_tokens) + len(tokens) > self.max_seq_length - 2:
                break

            box2token_span_map.append(
                [len(list_tokens) + 1, len(list_tokens) + len(tokens) + 1]
            )  # including st_idx
            list_tokens += tokens

            # min, max clipping
            for coord_idx in range(4):
                bb[coord_idx][0] = max(0.0, min(bb[coord_idx][0], width))
                bb[coord_idx][1] = max(0.0, min(bb[coord_idx][1], height))

            bb = list(itertools.chain(*bb))
            bbs = [bb for _ in range(len(tokens))]

            for _ in tokens:
                cum_token_idx += 1
                this_box_token_indices.append(cum_token_idx)

            list_bbs.extend(bbs)
            box_to_token_indices.append(this_box_token_indices)

        sep_bbs = [width, height] * 4

        # For [CLS] and [SEP]
        list_tokens = (
            [self.cls_token_id]
            + list_tokens[: self.max_seq_length - 2]
            + [self.sep_token_id]
        )
        if len(list_bbs) == 0:
            # When len(json_obj["words"]) == 0 (no OCR result)
            list_bbs = [cls_bbs] + [sep_bbs]
        else:  # len(list_bbs) > 0
            list_bbs = [cls_bbs] + list_bbs[: self.max_seq_length - 2] + [sep_bbs]

        len_list_tokens = len(list_tokens)
        input_ids[:len_list_tokens] = list_tokens
        attention_mask[:len_list_tokens] = 1

        bbox[:len_list_tokens, :] = list_bbs

        # bounding box normalization -> [0, 1]
        bbox[:, [0, 2, 4, 6]] = bbox[:, [0, 2, 4, 6]] / width
        bbox[:, [1, 3, 5, 7]] = bbox[:, [1, 3, 5, 7]] / height

        if self.backbone_type == "layoutlm":
            bbox = bbox[:, [0, 1, 4, 5]]
            bbox = bbox * 1000
            bbox = bbox.astype(int)

        st_indices = [
            indices[0]
            for indices in box_to_token_indices
            if indices[0] < self.max_seq_length
        ]
        are_box_first_tokens[st_indices] = True

        # Label
        relations = json_obj["parse"]["relations"]
        for relation in relations:
            if relation[0] >= len(box2token_span_map) or relation[1] >= len(
                box2token_span_map
            ):
                continue
            if (
                box2token_span_map[relation[0]][0] >= self.max_seq_length
                or box2token_span_map[relation[1]][0] >= self.max_seq_length
            ):
                continue

            word_from = box2token_span_map[relation[0]][0]
            word_to = box2token_span_map[relation[1]][0]
            el_labels[word_to] = word_from

        input_ids = torch.from_numpy(input_ids)
        bbox = torch.from_numpy(bbox)
        attention_mask = torch.from_numpy(attention_mask)

        are_box_first_tokens = torch.from_numpy(are_box_first_tokens)
        el_labels = torch.from_numpy(el_labels)

        return_dict = {
            "input_ids": input_ids,
            "bbox": bbox,
            "attention_mask": attention_mask,
            "are_box_first_tokens": are_box_first_tokens,
            "el_labels": el_labels,
        }

        return return_dict

    def __getitem__(self, idx):
        if self.model_head == "bies":
            assert self.dataset == "funsd"
            return_dict = self._getitem_for_funsd_bies(idx)
        elif self.model_head == "bio":
            return_dict = self._getitem_bio(idx)
        elif self.model_head == "spade":
            return_dict = self._getitem_spade(idx)
        elif self.model_head == "spade_rel":
            return_dict = self._getitem_spade_rel(idx)
        else:
            raise ValueError(f"Unknown self.model_head={self.model_head}")

        return return_dict
