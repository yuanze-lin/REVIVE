import pdb
import torch
import random
import json
import numpy as np

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 opt,
                 data,
                 context_prefix='context:',
                 question_prefix='question:',
                 em_title_prefix='entity:',
                 em_passage_prefix='description:',
                 im_title_prefix='candidate:',
                 im_passage_prefix='evidence:'):
        self.data = data
        self.n_box = opt.box_number
        self.context_prefix = context_prefix
        self.n_im_context = opt.n_im_context
        self.n_ex_context = opt.n_ex_context
        self.question_prefix = question_prefix
        self.em_title_prefix = em_title_prefix
        self.em_passage_prefix = em_passage_prefix
        self.im_title_prefix = im_title_prefix
        self.im_passage_prefix = im_passage_prefix
        self.n_tags = opt.n_tags
        self.sort_data()

    def __len__(self):
        return len(self.data)

    def get_target(self, example):
        if 'target' in example:
            target = example['target']
            return target + ' </s>'
        elif 'answers_list' in example:
            return random.choice(example['answers_list']) + ' </s>'
        else:
            return None

    def __getitem__(self, index):
        example = self.data[index]
        question = self.question_prefix + " " + example['question']
        if example['vis_feat'].shape[0] >= self.n_box:
            vis_feat = example['vis_feat'][:self.n_box]
            pos = example['boxes'][:self.n_box]
        else:
            nums = self.n_box - len(example['vis_feat'])
         
            vis_feat_tmp = np.tile(example['vis_feat'][0], (nums, 1))
            vis_feat = np.concatenate((vis_feat_tmp, example['vis_feat']), axis=0)
            
            pos_tmp = np.tile(example['boxes'][0], (nums,1))
            pos = np.concatenate((pos_tmp, example['boxes']), axis=0)
        target = self.get_target(example)

        if 'ex_ctxs' in example and self.n_ex_context is not None:
            f = self.em_title_prefix + " {} \n " + self.em_passage_prefix + " {} \n"
            if len(example['ex_ctxs'])<self.n_ex_context:
                contexts = [example['ex_ctxs'][0]] * (self.n_ex_context-len(example['ex_ctxs'])) \
                                + example['ex_ctxs']
            else:
                contexts = example['ex_ctxs'][:self.n_ex_context]
            passages = [f.format(c['title'], c['text']) for c in contexts]
            scores = [float(c['score']) for c in contexts]
            scores = torch.tensor(scores)
            if len(contexts) == 0:
                contexts = [question]
        if 'im_ctxs' in example:
            f = self.im_title_prefix + " {} \n " + self.im_passage_prefix + " {}"
            contexts2 = example['im_ctxs'][:self.n_im_context]
            passages2 = [f.format(c['title'], c['text']) for c in contexts2]
            scores2 = [float(c['score']) for c in contexts2]
            scores2 = torch.tensor(scores2)
            if passages:
                passages.extend(passages2)
                scores = torch.cat((scores, scores2))
            else:
                passages, scores = passages2, scores2
        else:
            passages, scores = None, None
       
        caption = self.context_prefix + " " + example['caption']
        tags = example['tags'][:self.n_tags]
        question = caption + '. ' + ', '.join(tags) + '\n' + question

        return {
            'index' : index,
            'question' : question,
            'target' : target,
            'vis_feat': vis_feat,
            'pos': pos,
            'passages' : passages,
            'scores' : scores
        }

    def sort_data(self):
        if self.n_ex_context is None or self.n_im_context is None or not 'score' in self.data[0]['im_ctxs'][0] or not 'score' in self.data[0]['ex_ctxs'][0]:
            return
        for ex in self.data:
            ex['im_ctxs'].sort(key=lambda x: float(x['score']), reverse=True)
            ex['ex_ctxs'].sort(key=lambda x: float(x['score']), reverse=True)

    def get_example(self, index):
        return self.data[index]

    def extract_im_element(self, im_context):
        im_context_new, entities = [], []
        for i in im_context:
            key = i['title']
            if key not in entities:
                entities.append(key)
                im_context_new.append(i)

        return im_context_new

def encode_passages(batch_text_passages, tokenizer, max_length):
    passage_ids, passage_masks = [], []
    for k, text_passages in enumerate(batch_text_passages):
        p = tokenizer.batch_encode_plus(
            text_passages,
            max_length=max_length,
            pad_to_max_length=True,
            return_tensors='pt',
            truncation=True
        )
        passage_ids.append(p['input_ids'][None])
        passage_masks.append(p['attention_mask'][None])

    passage_ids = torch.cat(passage_ids, dim=0)
    passage_masks = torch.cat(passage_masks, dim=0)
    return passage_ids, passage_masks.bool()

class Collator(object):
    def __init__(self, text_maxlength, tokenizer, answer_maxlength=20):
        self.tokenizer = tokenizer
        self.text_maxlength = text_maxlength
        self.answer_maxlength = answer_maxlength

    def __call__(self, batch):
        assert(batch[0]['target'] != None)
        index = torch.tensor([ex['index'] for ex in batch])
        target = [ex['target'] for ex in batch]
        target = self.tokenizer.batch_encode_plus(
            target,
            max_length=self.answer_maxlength if self.answer_maxlength > 0 else None,
            pad_to_max_length=True,
            return_tensors='pt',
            truncation=True if self.answer_maxlength > 0 else False,
        )
        target_ids = target["input_ids"]
        target_mask = target["attention_mask"].bool()
        
        vis_feats = [torch.unsqueeze(torch.from_numpy(ex['vis_feat']), 0) for ex in batch]
        pos = [torch.unsqueeze(torch.from_numpy(ex['pos']), 0) for ex in batch]
        vis_feats = torch.cat(vis_feats, dim=0)
        pos = torch.cat(pos, dim=0)
        target_ids = target_ids.masked_fill(~target_mask, -100)

        def append_question(example):
            if example['passages'] is None:
                return [example['question']]
            return [example['question'] + " \n " + t for t in example['passages']]
        text_passages = [append_question(example) for example in batch]
        passage_ids, passage_masks = encode_passages(text_passages,
                                                     self.tokenizer,
                                                     self.text_maxlength)

        return (index, target_ids, target_mask, passage_ids, passage_masks, vis_feats, pos)

def load_data(data_path=None, global_rank=-1, world_size=-1):
    assert data_path
    if data_path.endswith('.jsonl'):
        data = open(data_path, 'r')
    elif data_path.endswith('.json'):
        with open(data_path, 'r') as fin:
            data = json.load(fin)
    else:
        data = np.load(data_path, allow_pickle=True)
    examples = []
    for k, example in enumerate(data):
        if global_rank > -1 and not k%world_size==global_rank:
            continue
        if data_path is not None and data_path.endswith('.jsonl'):
            example = json.loads(example)
        if not 'id' in example:
            example['id'] = k
        for i, c in enumerate(example['im_ctxs']):
            if not 'score' in c:
                c['score'] = 1.0 / (i + 1)

        for i, c in enumerate(example['ex_ctxs']):
            if not 'score' in c:
                c['score'] = 1.0 / (i + 1)
        examples.append(example)

    if data_path is not None and data_path.endswith('.jsonl'):
        data.close()

    return examples

class RetrieverCollator(object):
    def __init__(self, tokenizer, passage_maxlength=200, question_maxlength=40):
        self.tokenizer = tokenizer
        self.passage_maxlength = passage_maxlength
        self.question_maxlength = question_maxlength

    def __call__(self, batch):
        index = torch.tensor([ex['index'] for ex in batch])

        question = [ex['question'] for ex in batch]
        question = self.tokenizer.batch_encode_plus(
            question,
            pad_to_max_length=True,
            return_tensors="pt",
            max_length=self.question_maxlength,
            truncation=True
        )
        question_ids = question['input_ids']
        question_mask = question['attention_mask'].bool()

        if batch[0]['scores'] is None or batch[0]['passages'] is None:
            return index, question_ids, question_mask, None, None, None

        scores = [ex['scores'] for ex in batch]
        scores = torch.stack(scores, dim=0)

        passages = [ex['passages'] for ex in batch]
        passage_ids, passage_masks = encode_passages(
            passages,
            self.tokenizer,
            self.passage_maxlength
        )

        return (index, question_ids, question_mask, passage_ids, passage_masks, scores)

class TextDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data,
                 title_prefix='title:',
                 passage_prefix='context:'):
        self.data = data
        self.title_prefix = title_prefix
        self.passage_prefix = passage_prefix

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        example = self.data[index]
        text = self.title_prefix + " " + example[2] + " " + \
            self.passage_prefix + " " + example[1]
        return example[0], text

class TextCollator(object):
    def __init__(self, tokenizer, maxlength=200):
        self.tokenizer = tokenizer
        self.maxlength = maxlength

    def __call__(self, batch):
        index = [x[0] for x in batch]
        encoded_batch = self.tokenizer.batch_encode_plus(
            [x[1] for x in batch],
            pad_to_max_length=True,
            return_tensors="pt",
            max_length=self.maxlength,
            truncation=True
        )
        text_ids = encoded_batch['input_ids']
        text_mask = encoded_batch['attention_mask'].bool()

        return index, text_ids, text_mask
