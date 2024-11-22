import argparse
import torch
from tqdm import tqdm
from core.dataset import MMDataLoader
from core.scheduler import get_scheduler
from core.utils import AverageMeter, setup_seed, save_model, save_print_results
from models.Encoder_KIAdapter import UniPretrain
from core.metric import MetricsTop


def parse_opts():
    parser = argparse.ArgumentParser(description='Pretrained Adapter')

    parser.add_argument('--datasetName', type=str, default='external_knowledge',
                        help='select external knowledge base for pre-training')
    parser.add_argument('--train_mode', type=str, default='regression',
                        help='type of pre-training labels')

    parser.add_argument('--dataPath', type=str, default='/opt/data/private/Project/Datasets/MSA_Datasets/SIMSv2/SIMSv2s/Processed/unaligned.pkl',
                        help='path for checkpointing')
    parser.add_argument('--savePath', type=str, default='./pretrainedModel/',
                        help='path for checkpointing')

    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='')
    parser.add_argument('--n_epochs', type=list, default=[100, 100, 50],
                        help='epoch number for training')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='')
    parser.add_argument('--seq_lens', type=list, default=[50, 232, 925],
                        help='features length of each modality for pre-training')
    parser.add_argument('--fea_dims', type=list, default=[768, 177, 25],
                        help='features length of each modality for pre-training')
    parser.add_argument('--lr', type=int, default=1e-4,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=int, default=1e-3,
                        help='learning rate')

    args = parser.parse_args()
    return args


def train(modality, model, device, train_loader, optimizer, loss_fn, epoch, metrics):
    train_pbar = tqdm(train_loader)
    losses = AverageMeter()
    y_pred, y_true = [], []

    model.train()
    for data in train_pbar:
        inputs = {
            'V': data['vision'].to(device),
            'A': data['audio'].to(device),
            'T': data['text'].to(device),
            'mask': {
                'V': data['vision_padding_mask'][:, 1:].to(device),
                'A': data['audio_padding_mask'][:, 1:].to(device),
                'T': []
            }
        }
        label = data['labels'][modality].to(device)
        label = label.view(-1, 1)
        batchsize = inputs['V'].shape[0]

        output = model(inputs)
        loss = loss_fn(output[1], label)
        losses.update(loss.item(), batchsize)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        y_pred.append(output[1].cpu())
        y_true.append(label.cpu())

        train_pbar.set_description('train')
        train_pbar.set_postfix({
            'epoch': '{}'.format(epoch),
            'loss': '{:.5f}'.format(losses.value_avg),
            'lr:': '{:.2e}'.format(optimizer.state_dict()['param_groups'][0]['lr'])
        })

    pred, true = torch.cat(y_pred), torch.cat(y_true)
    train_results = metrics(pred, true)

    return train_results


def evaluate(modality, model, device, eval_loader, optimizer, loss_fn, epoch, metrics):
    test_pbar = tqdm(eval_loader)
    losses = AverageMeter()
    y_pred, y_true = [], []

    model.eval()
    with torch.no_grad():
        for data in test_pbar:
            inputs = {
                'V': data['vision'].to(device),
                'A': data['audio'].to(device),
                'T': data['text'].to(device),
                'mask': {
                    'V': data['vision_padding_mask'][:, 1:].to(device),
                    'A': data['audio_padding_mask'][:, 1:].to(device),
                    'T': []
                }
            }
            label = data['labels'][modality].to(device)
            label = label.view(-1, 1)
            batchsize = inputs['V'].shape[0]

            output = model(inputs)
            y_pred.append(output[1].cpu())
            y_true.append(label.cpu())

            loss = loss_fn(output[1], label)
            losses.update(loss.item(), batchsize)

            test_pbar.set_description('eval')
            test_pbar.set_postfix({
                'epoch': '{}'.format(epoch),
                'loss': '{:.5f}'.format(losses.value_avg),
                'lr:': '{:.2e}'.format(optimizer.state_dict()['param_groups'][0]['lr'])
            })

        pred, true = torch.cat(y_pred), torch.cat(y_true)
        valid_results = metrics(pred, true)

    return valid_results


def test(modality, model, device, test_loader, optimizer, loss_fn, epoch, metrics):
    test_pbar = tqdm(test_loader)
    losses = AverageMeter()
    y_pred, y_true = [], []

    model.eval()
    with torch.no_grad():
        for data in test_pbar:
            inputs = {
                'V': data['vision'].to(device),
                'A': data['audio'].to(device),
                'T': data['text'].to(device),
                'mask': {
                    'V': data['vision_padding_mask'][:, 1:].to(device),
                    'A': data['audio_padding_mask'][:, 1:].to(device),
                    'T': []
                }
            }
            label = data['labels'][modality].to(device)
            label = label.view(-1, 1)
            batchsize = inputs['V'].shape[0]

            output = model(inputs)
            y_pred.append(output[1].cpu())
            y_true.append(label.cpu())

            loss = loss_fn(output[1], label)
            losses.update(loss.item(), batchsize)

            test_pbar.set_description('test')
            test_pbar.set_postfix({
                'epoch': '{}'.format(epoch),
                'loss': '{:.5f}'.format(losses.value_avg),
                'lr:': '{:.2e}'.format(optimizer.state_dict()['param_groups'][0]['lr'])
            })

        pred, true = torch.cat(y_pred), torch.cat(y_true)
        test_results = metrics(pred, true)

    return test_results


def main(i, modality):
    opt = parse_opts()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setup_seed(opt.seed)

    dataLoader = MMDataLoader(opt)
    model = UniPretrain(modality, num_patches=opt.seq_lens[i], fea_size=opt.fea_dims[i]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opt.lr,
        weight_decay=opt.weight_decay
    )

    loss_fn = torch.nn.MSELoss()
    metrics = MetricsTop().getMetics(opt.datasetName)
    scheduler_warmup = get_scheduler(optimizer, opt.n_epochs[i])

    for epoch in range(1, opt.n_epochs[i]+1):
        if epoch == 34:
            break
        train_results = train(modality, model, device, dataLoader['train'], optimizer, loss_fn, epoch, metrics)
        valid_results = evaluate(modality, model, device, dataLoader['valid'], optimizer, loss_fn, epoch, metrics)
        test_results = test(modality, model, device, dataLoader['test'], optimizer, loss_fn, epoch, metrics)

        save_print_results(opt, None, train_results, valid_results, test_results)
        scheduler_warmup.step()

    # 保存单模态预训练模型
    save_model(opt.savePath, test_results, modality, model)


if __name__ == '__main__':
    for i, m in enumerate(["A"]):
        main(i+2, modality=m)
