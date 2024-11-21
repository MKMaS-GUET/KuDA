import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


__all__ = ['MMDataLoader']


class MMDataset(Dataset):
    def __init__(self, args, mode='train'):
        self.mode = mode
        self.args = args
        DATA_MAP = {
            'mosi': self.__init_mosi,
            'mosei': self.__init_mosei,
            'sims': self.__init_sims,
            'simsv2': self.__init_simsv2,
            'external_knowledge': self.__init_external_knowledge
        }
        DATA_MAP[args.datasetName]()

    def __init_mosi(self):
        with open(self.args.dataPath, 'rb') as f:
            data = pickle.load(f)

        self.args.use_bert = True
        self.args.need_truncated = True
        self.args.need_data_aligned = False

        if self.args.use_bert:
            self.text = data[self.mode]['text_bert'].astype(np.float32)
        else:
            self.text = data[self.mode]['text'].astype(np.float32)

        self.vision = data[self.mode]['vision'].astype(np.float32)
        self.audio = data[self.mode]['audio'].astype(np.float32)

        self.rawText = data[self.mode]['raw_text']
        self.ids = data[self.mode]['id']
        self.labels = {
            'M': data[self.mode][self.args.train_mode+'_labels'].astype(np.float32)
        }
        if 'sims' in self.args.datasetName:
            for m in "TAV":
                self.labels[m] = data[self.mode][self.args.train_mode+'_labels_'+m]

        if not self.args.need_data_aligned:
            self.audio_lengths = data[self.mode]['audio_lengths']
            self.vision_lengths = data[self.mode]['vision_lengths']

        # Clear dirty data
        self.audio[self.audio == -np.inf] = 0
        self.vision[self.vision == -np.inf] = 0

        self.__gen_mask(data[self.mode])
        if self.args.need_truncated:
            self.__truncated()

    def __init_mosei(self):
        return self.__init_mosi()

    def __init_sims(self):
        return self.__init_mosi()

    def __init_simsv2(self):
        return self.__init_mosi()

    def __init_external_knowledge(self):
        with open(self.args.dataPath, 'rb') as f:
            data = pickle.load(f)

        with open('./pretrainedModel/pretrained_text.pkl', 'rb') as f2:
            data_t_en = pickle.load(f2)

        self.text = data[self.mode]['text_bert'].astype(np.float32)

        self.rawText = data[self.mode]['raw_text']
        if self.mode == 'train':
            self.rawText = data_t_en['en'][0:1368]
        elif self.mode == 'valid':
            self.rawText = data_t_en['en'][1368:1824]
        else:
            self.rawText = data_t_en['en'][1824:]

        self.vision = data[self.mode]['vision'].astype(np.float32)
        self.audio = data[self.mode]['audio'].astype(np.float32)

        self.ids = data[self.mode]['id']
        self.labels = {
            'M': data[self.mode][self.args.train_mode+'_labels'].astype(np.float32)
        }
        for m in "TAV":
            self.labels[m] = data[self.mode][self.args.train_mode+'_labels_'+m]

        self.audio_lengths = data[self.mode]['audio_lengths']
        self.vision_lengths = data[self.mode]['vision_lengths']

        # Clear dirty data
        self.audio[self.audio == -np.inf] = 0
        self.vision[self.vision == -np.inf] = 0

        self.__gen_mask(data[self.mode])
        self.__truncated()

    def __truncated(self):
        # NOTE: Here for dataset we manually cut the input into specific length.
        def Truncated(modal_features, length):
            if length == modal_features.shape[1]:
                return modal_features
            truncated_feature = []
            padding = np.array([0 for i in range(modal_features.shape[2])])
            for instance in modal_features:
                for index in range(modal_features.shape[1]):
                    if ((instance[index] == padding).all()):
                        if (index + length >= modal_features.shape[1]):
                            truncated_feature.append(instance[index:index+length])
                            break
                    else:
                        truncated_feature.append(instance[index:index+length])
                        break
            truncated_feature = np.array(truncated_feature)
            return truncated_feature

        text_length, video_length, audio_length = self.args.seq_lens

        self.vision = Truncated(self.vision, video_length)
        self.audio = Truncated(self.audio, audio_length)

    def __gen_mask(self, data):
        vision_tmp = torch.tensor([[True for i in range(data['vision'].shape[1])] for j in range(data['vision'].shape[0])])
        for i in range(len(vision_tmp)):
            vision_tmp[i][0:data['vision_lengths'][i]] = False

        vision_mask = torch.cat((vision_tmp[:, 0:1], vision_tmp), dim=-1)
        for i in range(self.__len__()):
            vision_mask[i][0] = False
        self.vision_padding_mask = vision_mask

        audio_tmp = torch.tensor([[True for i in range(data['audio'].shape[1])] for j in range(data['audio'].shape[0])])
        for i in range(len(audio_tmp)):
            audio_tmp[i][0:data['audio_lengths'][i]] = False

        audio_mask = torch.cat((audio_tmp[:, 0:1], audio_tmp), dim=-1)
        for i in range(self.__len__()):
            audio_mask[i][0] = False
        self.audio_padding_mask = audio_mask

    def __len__(self):
        return len(self.labels['M'])

    def __getitem__(self, index):
        sample = {
            'raw_text': self.rawText[index],
            'text': torch.Tensor(self.text[index]),
            'audio': torch.Tensor(self.audio[index]),
            'vision': torch.Tensor(self.vision[index]),
            'index': index,
            'id': self.ids[index],
            'labels': {k: torch.Tensor(v[index].reshape(-1)) for k, v in self.labels.items()},
            'mask': self.mask[index] if self.mode == 'train_mix' else []
        }
        sample['audio_lengths'] = self.audio_lengths[index]
        sample['vision_lengths'] = self.vision_lengths[index]
        sample['vision_padding_mask'] = self.vision_padding_mask[index]
        sample['audio_padding_mask'] = self.audio_padding_mask[index]
        return sample


def MMDataLoader(args):
    datasets = {
        'train': MMDataset(args, mode='train'),
        'valid': MMDataset(args, mode='valid'),
        'test': MMDataset(args, mode='test')
    }

    dataLoader = {
        ds: DataLoader(
            datasets[ds],
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True
        )
        for ds in datasets.keys()
    }

    return dataLoader
