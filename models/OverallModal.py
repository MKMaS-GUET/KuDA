import torch
from torch import nn
from models.Encoder_KIAdapter import UnimodalEncoder
from models.DyRoutFusion_CLS import DyRoutTrans, SentiCLS
from core.utils import calculate_ratio_senti


class KMSA(nn.Module):
    def __init__(self, opt, dataset, bert_pretrained='bert-base-uncased'):
        super(KMSA, self).__init__()
        # Unimodal Encoder & Knowledge Inject Adapter
        self.UniEncKI = UnimodalEncoder(opt, bert_pretrained)

        # Multimodal Fusion
        self.DyMultiFus = DyRoutTrans(opt)

        # Output Classification for Sentiment Analysis
        self.CLS = SentiCLS(opt)

    def forward(self, inputs_data_mask, multi_senti):
        # Unimodal Encoder & Knowledge Inject // Unimodal Sentiment Prediction
        uni_fea, uni_senti = self.UniEncKI(inputs_data_mask)    # [T, V, A]
        uni_mask = inputs_data_mask['mask']

        # Dynamic Multimodal Fusion using Dynamic Route Transformer with Unimodal Sentiment Prediction
        if multi_senti is not None:
            senti_ratio = calculate_ratio_senti(uni_senti, multi_senti, k=0.1)
        else:
            senti_ratio = None
        multimodal_features, nce_loss = self.DyMultiFus(uni_fea, uni_mask, senti_ratio)

        # Sentiment Classification
        prediction = self.CLS(multimodal_features)     # uni_fea['T'], uni_fea['V'], uni_fea['A']

        return prediction, nce_loss

    def preprocess_model(self, pretrain_path):
        # 加载预训练模型
        ckpt_t = torch.load(pretrain_path['T'])
        self.UniEncKI.enc_t.load_state_dict(ckpt_t)
        ckpt_v = torch.load(pretrain_path['V'])
        self.UniEncKI.enc_v.load_state_dict(ckpt_v)
        ckpt_a = torch.load(pretrain_path['A'])
        self.UniEncKI.enc_a.load_state_dict(ckpt_a)
        # 冻结外部知识注入参数
        for name, parameter in self.UniEncKI.named_parameters():
            if 'adapter' in name or 'decoder' in name:
                parameter.requires_grad = False


def build_model(opt):
    if 'sims' in opt.datasetName:
        l_pretrained = './BERT/bert-base-chinese'
    else:
        l_pretrained = './BERT/bert-base-uncased'

    model = KMSA(opt, dataset=opt.datasetName, bert_pretrained=l_pretrained)

    return model
