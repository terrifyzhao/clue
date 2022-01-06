import os
from typing import Union
from typing import Optional, List, Dict, Tuple
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import ExponentialLR, StepLR
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils import clip_grad_value_
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from transformers.modeling_utils import PreTrainedModel
from transformers.models.bert import BertTokenizer
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, \
    classification_report, cohen_kappa_score
from annlp.tricks.adversarial import FGM
from annlp.util.utils import sentence_percentile_length, is_oom_error, kl_div

DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 5e-5


class Callback:

    def __init__(self, trainer=None):
        self.trainer = trainer

    def set_trainer(self, trainer):
        self.trainer = trainer

    def on_train_epoch_start(self):
        pass

    def on_train_epoch_end(self):
        pass

    def on_train_step_start(self):
        pass

    def on_train_step_end(self):
        pass

    def on_dev_start(self):
        pass

    def on_dev_end(self):
        pass


class BaseDataset(Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])


class AugDataset(Dataset):
    def __init__(self, encodings, aug_encodings, labels, aug_labels, aug_rate=0.0):
        self.encodings = encodings
        self.aug_encodings = aug_encodings
        self.aug_length = int(len(self.aug_encodings['input_ids']) * aug_rate)
        # 随机采样
        index = np.random.choice(a=len(self.aug_encodings['input_ids']), size=self.aug_length, replace=False)
        for k in self.encodings.keys():
            aug_value = self.aug_encodings[k].index_select(dim=0, index=torch.from_numpy(index))
            self.encodings[k] = torch.cat([self.encodings[k], aug_value], dim=0)

        self.labels = labels + np.array(aug_labels)[index].tolist()

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class Trainer(ABC):
    def __init__(self,
                 model: PreTrainedModel,
                 model_path: str,
                 lr: float = DEFAULT_LEARNING_RATE,
                 batch_size: Union[int, None] = None,
                 max_length: Union[int, None] = None,
                 epochs: int = 100,
                 do_train: bool = True,
                 do_dev: bool = True,
                 do_test: bool = False,
                 test_with_label: bool = False,
                 save_model_name: str = 'best_model.bin',
                 attack: bool = False,
                 monitor: str = 'loss',
                 valid_every_n_epoch: int = 1,
                 warmup_steps=None,
                 decay_gamma: float = 1.0,
                 decay_epoch: int = -1,
                 clip_value: Union[float, None] = None,
                 call_back: Union[Callback, None] = None,
                 save_metric=None,
                 use_apex=True,
                 mix=None,
                 augmentation=False,
                 r_drop: bool = False,
                 cutoff_rate: float = 0.0,
                 cutoff_direction: str = 'row',
                 task_name=None):
        """
        :param model: 实例化的模型对象
        :param model_path: 预训练模型的存储路径
        :param lr: 学习率
        :param batch_size: 批次大小
        :param max_length: 序列的最大长度
        :param epochs: 训练轮数
        :param do_train: 是否训练
        :param do_dev: 是否验证
        :param do_test: 是否测试
        :param test_with_label: 测试时是否有标签
        :param save_model_name: 保存的模型名
        :param attack: 是否做对抗训练
        :param monitor: 保存模型的监控指标
        :param valid_every_n_epoch: 经过多少个epoch做一次验证
        :param warmup_steps: warm的步数
        :param decay_gamma: 学习率衰减率，默认是1即不衰减
        :param decay_epoch: 学习率每隔多少epoch衰减一次，默认使用ExponentialLR且衰减率是1，如果设置了decay_epoch则执行StepLR衰减
        :param clip_value: 梯度裁剪的值
        :param call_back: 训练过程的回调
        :param save_metric: 保存metric的文件名
        :param use_apex: 是否使用半精度
        :param mix: 是否使用mixup，可以对embedding、pool进行mixup
        :param augmentation: 是否使用数据增强的数据进行curriculum learning
        :r_drop: 是否使用r_drop数据增强方法
        :param cutoff_rate: 若cutoff_rate != 0，则使用cutoff增强
        :param cutoff_direction: 做cutoff增强的维度，可选['raw','column','random']
        """
        self.model = model
        self.batch_size = batch_size
        self.lr = lr
        self.max_length = max_length
        self.model_path = model_path
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(self.device)
        self.model.train()
        self.optimizer = self.configure_optimizer()
        self.do_train = do_train
        self.do_dev = do_dev
        self.do_test = do_test
        self.test_with_label = test_with_label
        self.epochs = epochs
        self.save_model_name = save_model_name
        self.attack = attack
        self.monitor = monitor
        self.valid_every_n_epoch = valid_every_n_epoch
        self.warmup_steps = warmup_steps
        self.decay_gamma = decay_gamma
        self.decay_epoch = decay_epoch
        self.clip_value = clip_value
        self.call_back = call_back
        self.use_apex = use_apex
        self.augmentation = augmentation
        self.num_labels = model.num_labels
        self.r_drop = r_drop
        self.cutoff_rate = cutoff_rate
        self.cutoff_direction = cutoff_direction
        self.task_name = task_name

        if self.call_back is not None:
            self.call_back.set_trainer(self)
        self.data_preprocess()

        # 是否做对抗训练
        if attack:
            self.fgm = FGM(model)

        if mix is not None:
            from .feature_augmentation import MixUp
            self.mix = MixUp(self.model, self.tokenizer, num_labels=self.num_labels, layer=mix)
        else:
            self.mix = None

        if cutoff_rate > 0:
            from .feature_augmentation import Cutoff
            self.cutoff = Cutoff(self.model, self.tokenizer, cutoff_rate, cutoff_direction)
        else:
            self.cutoff = None

        # 读取数据时支持传递分词结果和dataset
        if do_train:
            train_res = self.get_train_data()
            if isinstance(train_res, Dataset):
                self.train_loader = self._data2loader(self.get_train_data())
            else:
                self.train_data, self.train_label = train_res
                if self.augmentation:
                    self.aug_data, self.aug_label = self.get_aug_data()
                self.train_loader = self._data2loader(BaseDataset(self.train_data, self.train_label))

        if do_dev:
            dev_res = self.get_dev_data()
            if isinstance(dev_res, Dataset):
                self.dev_loader = self._data2loader(self.get_dev_data())
            else:
                self.dev_loader = self._data2loader(BaseDataset(*self.get_dev_data()))

        if do_test:
            test_res = self.get_test_data()
            if isinstance(test_res, Dataset):
                self.test_loader = self._data2loader(self.get_test_data())
            else:
                self.test_loader = self._data2loader(BaseDataset(self.get_test_data()))

        # 是否使用半精度
        if torch.cuda.is_available() and self.use_apex:
            self.amp = __import__('apex').amp
            self.model, self.optimizer = self.amp.initialize(self.model, self.optimizer, opt_level="O1")

        self.do_acc = True
        self.do_recall = False
        self.do_precision = False
        self.do_f1 = False
        self.do_kappa = False
        self.print_report = False
        self.average = None

        self.save_metric = save_metric
        self.metric_list = []

    @abstractmethod
    def get_train_data(self) -> Tuple[Dict, List]:
        raise NotImplementedError

    @abstractmethod
    def get_dev_data(self) -> Tuple[Dict, List]:
        raise NotImplementedError

    def get_test_data(self) -> Tuple[Dict, Optional[List]]:
        pass

    def get_aug_data(self) -> Tuple[Dict, List]:
        pass

    @abstractmethod
    def configure_optimizer(self):
        raise NotImplementedError

    def train_step(self, data, mode) -> Tuple[torch.Tensor, torch.Tensor, List]:
        raise NotImplementedError

    def predict_step(self, data) -> Union[list, np.ndarray]:
        pass

    def adversarial(self, data):
        """
        仅针对bert类模型有效
        """
        self.optimizer.zero_grad()
        # 添加扰动
        self.fgm.attack(emb_name='embeddings.word_embeddings.weight')
        # 重新计算梯度
        adv_loss = self.model(input_ids=data['input_ids'].to(self.device),
                              attention_mask=data['attention_mask'].to(self.device),
                              labels=data['labels'].to(self.device)).loss
        # bp得到新的梯度
        adv_loss.backward()
        self.fgm.restore(emb_name='embeddings.word_embeddings.weight')

    def tokenizer_(self,
                   text,
                   text_pair=None,
                   return_tensors='pt',
                   truncation=True,
                   padding='max_length',
                   max_length=None):
        if max_length is None and self.max_length is None:
            max_length = min(sentence_percentile_length(text), 512)
        elif max_length is None and self.max_length is not None:
            max_length = self.max_length

        return self.tokenizer(text=text,
                              text_pair=text_pair,
                              return_tensors=return_tensors,
                              truncation=truncation,
                              padding=padding,
                              max_length=max_length)

    def _data2loader(self, dataset):
        # shuffle要用False，不然测试集顺序会打乱
        return DataLoader(dataset,
                          self.batch_size,
                          pin_memory=True if torch.cuda.is_available() else False,
                          shuffle=False)

    def data_preprocess(self):
        pass

    def _adjust_batch_size(self, dataset):
        """
        采用二分的方法搜寻较为合适的batch_size
        :param dataset: 数据集
        """
        low = self.batch_size
        high = low
        max_trials = 10
        count = 0
        while 1:
            try:
                loader = dataset[0:high]
                self.train_step(loader, mode='train')
                # 如果没有问题就把bs扩大一倍
                low, high = high, high * 2
                count += 1
                if count >= max_trials:
                    break
            except RuntimeError as exception:
                if is_oom_error(exception):
                    # 如果oom就缩小一点，low不变
                    high = (high + low) // 2
                else:
                    raise
            torch.cuda.empty_cache()
            if high - low <= 1:
                break
        torch.cuda.empty_cache()
        return int(high * 0.9)

    def warmup_schedule(self, total_step):
        """
        warmup_schedule
        :param total_step 总的训练步数
        """
        if self.warmup_steps is not None:
            warm_schedule = get_linear_schedule_with_warmup(self.optimizer,
                                                            num_warmup_steps=self.warmup_steps,
                                                            num_training_steps=total_step)
            return warm_schedule

    def decay_schedule(self):
        """
        学习率衰减
        """
        if self.decay_epoch != -1:
            decay_schedule = StepLR(self.optimizer, step_size=self.decay_epoch, gamma=self.decay_gamma)
        elif self.decay_gamma != 1:
            decay_schedule = ExponentialLR(self.optimizer, gamma=self.decay_gamma)
        else:
            decay_schedule = None
        return decay_schedule

    def back_propagation(self, loss):
        if torch.cuda.is_available() and self.use_apex:
            with self.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

    def mixup_backward(self, data):
        mix_loss = self.mix.augmentation(data)
        if mix_loss is not None:
            self.back_propagation(mix_loss / 2)

    def cutoff_backward(self, data):
        self.optimizer.zero_grad()
        cutoff_loss = self.cutoff.augmentation(data)
        if cutoff_loss is not None:
            self.back_propagation(cutoff_loss)
        self.optimizer.step()

    def _train_func(self, loader):
        train_loss = 0
        all_label = []
        all_pred = []
        pbar = tqdm(loader)

        # warmup
        warm_schedule = self.warmup_schedule(len(loader) * self.epochs)
        # 学习率衰减
        decay_schedule = self.decay_schedule()

        for batch in pbar:
            if self.call_back is not None:
                self.call_back.on_train_step_start()

            self.optimizer.zero_grad()

            loss, logits, label = self.train_step(batch, mode='train')
            pred = logits.argmax(dim=1).cpu().numpy()
            if pred is not None and label is not None:
                all_label.extend(label)
                all_pred.extend(pred)

            if self.r_drop:
                loss_v1, logits_v1, _ = self.train_step(batch, mode='train')
                loss += loss_v1
                kl_loss = (kl_div(logits, logits_v1) + kl_div(logits_v1, logits)).mean() / 4.0
                loss += (kl_loss * 4.0)

            if self.mix is not None:
                loss = loss / 2
            self.back_propagation(loss)

            # 对抗训练
            if self.attack:
                self.adversarial(batch)

            # 梯度裁剪
            if self.clip_value is not None:
                clip_grad_value_(self.model.parameters(), clip_value=self.clip_value)

            # mixup
            if self.mix is not None:
                self.mixup_backward(batch)

            # cutoff
            if self.cutoff is not None:
                self.cutoff_backward(batch)

            self.optimizer.step()

            if self.call_back is not None:
                self.call_back.on_train_step_end()

            if warm_schedule is not None:
                warm_schedule.step()
            if decay_schedule is not None:
                decay_schedule.step()

            train_loss += loss.item()

            pbar.update()
            pbar.set_description(f'loss:{loss.item():.4f}')

        self._print_metrics('train', train_loss / len(loader), **self._calculate_metrics(all_label, all_pred))

    def _dev_func(self, loader, mode):
        dev_loss = 0
        all_label = []
        all_pred = []
        metrics = {}

        for batch in tqdm(loader):
            with torch.no_grad():
                loss, logits, label = self.train_step(batch, mode)
                output = logits.argmax(dim=1).cpu().numpy()
                all_pred.extend(output)
                all_label.extend(label)
                dev_loss += loss.item()

        # 打印评价指标
        if output is not None:
            metrics = self._calculate_metrics(all_label, all_pred)
            self._print_metrics(mode, dev_loss / len(loader), **metrics)
            metrics.update({'loss': dev_loss / len(loader)})
            self.metric_list.append(metrics)
            if self.print_report:
                target_names = [f'class {i}' for i in range(len(set(all_label)))]
                print(classification_report(all_label, all_pred, target_names=target_names))

        # 返回monitor指标
        if self.monitor == 'loss':
            return dev_loss / len(loader)
        elif output is not None:
            return metrics[self.monitor]

    def _predict_func(self, loader):
        all_out = []
        for batch in tqdm(loader):
            with torch.no_grad():
                output = self.predict_step(batch)
                all_out.extend(output)
        return all_out

    def _print_metrics(self, mode, loss, acc, recall, precision, f1, kappa):
        result_str = f'{mode} loss:{loss:.4f}'
        if self.do_acc:
            result_str += f' acc:{acc:.4f}'
        if self.do_recall:
            result_str += f' recall:{recall:.4f}'
        if self.do_precision:
            result_str += f' precision:{precision:.4f}'
        if self.do_f1:
            result_str += f' f1:{f1:.4f}'
        if self.do_kappa:
            result_str += f' kappa:{kappa:.4f}'
        print(result_str)
        return result_str

    def configure_metrics(self, do_acc=True, do_recall=False, do_precision=False, do_f1=False,
                          do_kappa=False, print_report=False, average=None):
        self.do_acc = do_acc
        self.do_recall = do_recall
        self.do_precision = do_precision
        self.do_f1 = do_f1
        self.do_kappa = do_kappa
        self.print_report = print_report
        self.average = average

    def _calculate_metrics(self, y_true, y_pred):
        acc = 0
        recall = 0
        precision = 0
        f1 = 0
        kappa = 0
        if self.do_acc:
            acc = accuracy_score(y_true, y_pred)
        if self.do_recall:
            recall = recall_score(y_true, y_pred, average=self.average)
        if self.do_precision:
            precision = precision_score(y_true, y_pred, average=self.average)
        if self.do_f1:
            f1 = f1_score(y_true, y_pred, average=self.average)
        if self.do_kappa:
            kappa = cohen_kappa_score(y_true, y_pred)
        result = {'acc': acc, 'recall': recall, 'precision': precision, 'f1': f1, 'kappa': kappa}

        return result

    def _update_train_data(self, aug_rate):
        aug_rate = min(0.5, aug_rate)
        if 0 < aug_rate <= 0.5:
            print('update dataloader, rate:', aug_rate)
            trainer_dataset = AugDataset(self.train_data, self.aug_data,
                                         self.train_label, self.aug_label,
                                         aug_rate=aug_rate)
            self.train_loader = self._data2loader(trainer_dataset)

    def run(self):
        if self.monitor == 'loss':
            dev_metric = float('inf')
        else:
            dev_metric = 0
        try:
            for epoch in range(self.epochs):
                if self.do_train:
                    # 如果有做数据增强，就采用curriculum learning，每个epoch增加一部分数据
                    if self.augmentation:
                        self._update_train_data(epoch * 0.1)
                    print(f'***********epoch: {epoch + 1}***********')
                    if self.call_back is not None:
                        self.call_back.on_train_epoch_start()
                    # 训练
                    self._train_func(self.train_loader)
                    # 每个epoch逐渐增加cutoff_rate
                    if self.cutoff is not None:
                        if self.cutoff.cutoff_rate < 0.1:
                            self.cutoff.cutoff_rate += 0.01

                    if self.call_back is not None:
                        self.call_back.on_train_epoch_end()

                    if not self.do_dev:
                        torch.save(self.model.state_dict(), self.save_model_name)
                        print('save model')
                if self.do_dev:
                    if (epoch + 1) % self.valid_every_n_epoch == 0:
                        if self.call_back is not None:
                            self.call_back.on_dev_start()
                        metric = self._dev_func(self.dev_loader, mode='dev')
                        if self.call_back is not None:
                            self.call_back.on_dev_end()
                        if self.monitor == 'loss':
                            if dev_metric > metric:
                                dev_metric = metric
                                torch.save(self.model.state_dict(), self.save_model_name)
                                print('save model')
                        else:
                            if dev_metric < metric:
                                dev_metric = metric
                                torch.save(self.model.state_dict(), self.save_model_name)
                                print('save model')
                        print(f'best {self.monitor}:{dev_metric}')
                if not self.do_train:
                    break
        finally:
            if self.save_metric is not None:
                self._save_metric()

        if self.do_test:
            self.model.eval()
            if self.test_with_label:
                self._dev_func(self.test_loader, mode='test')
            else:
                y_pred = self._predict_func(self.test_loader)
                return y_pred

    def _save_metric(self):
        """
        保存指标信息
        """
        dic = {}
        for metric in self.metric_list:
            for k, v in metric.items():
                if k in dic:
                    dic[k].append(v)
                else:
                    dic[k] = [v]
        epoch = [i + 1 for i in range(len(self.metric_list))]
        dic.update({'epoch': epoch})
        dic.update({'model': self.save_metric})
        df = pd.DataFrame(dic)
        if not os.path.exists('./metrics'):
            os.mkdir('./metrics')
        df.to_csv('./metrics/' + self.save_metric + '_metric.csv', index=False, encoding='utf_8_sig')

    def set_label_dict(self, label_dict):
        """
        用于label embedding设置标签信息
        """
        from annlp import BertForLabelEmbedding

        max_length = max([len(v) for k, v in label_dict.items()]) + 2

        if isinstance(self.model, BertForLabelEmbedding):
            label_dict = {k: self.tokenizer(v, return_tensors='pt', max_length=max_length, padding='max_length',
                                            truncation=True).input_ids.to(self.device) for k, v in
                          label_dict.items()}
            self.model.label_dict = label_dict
