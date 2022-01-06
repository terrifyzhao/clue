import os
import argparse
import torch
import pandas as pd
from transformers import AdamW
from sklearn.utils import shuffle
from annlp import fix_seed, ptm_path, get_device, BertForMultiClassification
from trainer import Trainer
import json


class MyTrainer(Trainer):

    def sentence_concat(self, text):
        text = str(text)
        if len(text) > 128:
            text = text[0:63] + text[63:]
        return text

    def get_train_data(self):

        train_label = []
        train_text1 = []
        train_text2 = []
        if self.task_name == 'afqmc':
            with open('data/afqmc/train.json') as file:
                for line in file.readlines():
                    j = json.loads(line.strip())
                    train_label.append(int(j['label']))
                    train_text1.append(j['sentence1'])
                    train_text2.append(j['sentence2'])

        return self.tokenizer_(train_text1, train_text2), train_label

    def get_dev_data(self):
        dev_label = []
        dev_text1 = []
        dev_text2 = []
        if self.task_name == 'afqmc':
            with open('data/afqmc/train.json') as file:
                for line in file.readlines():
                    j = json.loads(line.strip())
                    dev_label.append(int(j['label']))
                    dev_text1.append(j['sentence1'])
                    dev_text2.append(j['sentence2'])

        return self.tokenizer_(dev_text1, dev_text2), dev_label

    def get_test_data(self):
        pass

    def get_aug_data(self):
        pass

    def configure_optimizer(self):
        return AdamW(self.model.parameters(), lr=self.lr)

    def train_step(self, data, mode):
        input_ids = data['input_ids'].to(self.device)
        attention_mask = data['attention_mask'].to(self.device)
        labels = data['labels'].to(self.device).long()

        outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
        output = outputs.logits
        return outputs.loss, output, labels.cpu().numpy()

    def predict_step(self, data):
        input_ids = data['input_ids'].to(self.device)
        attention_mask = data['attention_mask'].to(self.device)

        outputs = self.model(input_ids, attention_mask=attention_mask)
        output = outputs.logits.argmax(dim=-1).cpu().numpy()
        return output


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--max_length', type=int, default=125, metavar='ML', help='text max length')
    parser.add_argument('--lr', type=float, default=5e-5, metavar='LR', help='learning rate')
    parser.add_argument('--batch_size', type=int, default=128, metavar='BS', help='batch size')
    parser.add_argument('--epochs', type=int, default=20, metavar='E', help='number of epochs')
    parser.add_argument('--save_model', type=str, default='best_model.bin', metavar='N', help='name of saved model')
    parser.add_argument('--monitor', type=str, default='acc', help='evaluating indicator')
    parser.add_argument('--mode', type=str, default='train', help='train or predict')
    parser.add_argument('--task_name', type=str, default='afqmc', help='task name')
    parser.add_argument('--mixup', type=str, default=None, help='type of mixup',
                        choices=['pooler', 'embedding', 'inner'])

    args = parser.parse_args()

    if args.mode == 'train':
        do_train = True
        do_dev = True
        do_test = False
        load_model = False
    else:
        do_train = False
        do_dev = False
        do_test = True
        load_model = True

    fix_seed()

    task_name = args.task_name
    num_labels = 2
    if task_name == 'afqmc':
        num_labels = 2

    model_name = 'best_model.p'
    model_path = ptm_path('roberta')
    print(model_path)

    model = BertForMultiClassification.from_pretrained(model_path, num_labels=num_labels)

    if os.path.exists(model_name) and load_model:
        print('************load model************')
        model.load_state_dict(torch.load(model_name, map_location=get_device()))

    trainer = MyTrainer(model, batch_size=args.batch_size, lr=args.lr, max_length=args.max_length,
                        model_path=model_path, do_train=do_train, do_dev=do_dev, do_test=do_test,
                        save_model_name=model_name, attack=False, monitor=args.monitor, epochs=args.epochs,
                        mix=args.mixup, task_name=args.task_name)

    trainer.configure_metrics(do_acc=True, do_f1=False, do_recall=False, do_precision=False, do_kappa=False,
                              print_report=True, average='macro')
    y_pred = trainer.run()
    if do_test:
        test = pd.read_csv('data/positive_dev_new.csv')
        test['pred_label'] = y_pred
        if not os.path.exists('output/'):
            os.mkdir('output/')
        test.to_csv('output/result.csv', index=False, encoding='utf_8_sig')


if __name__ == '__main__':
    main()
