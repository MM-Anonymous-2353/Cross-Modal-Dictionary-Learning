import argparse
import os
import importlib
from tqdm import tqdm
import torchvision.transforms as tf
import yaml
import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
from PIL import Image
from torchvision.models.utils import load_state_dict_from_url
import pdb
import random
import utils
import network_feature_group
from dataset import transforms, Datasets, Cityscapes, SUNRGBD
import torchvision
# from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', default='config/default.yaml')

# def collate_fn_train(batch):

#     masks = torch.tensor([mask[3].tolist() for mask in batch])
#     # rgbs = torch.tensor([rgb[0].tolist() for rgb in batch if not (rgb[0] == 0).all()])
#     # depths = torch.tensor([depth[1].tolist() for depth in batch if not (depth[1] == 0).all()])
#     rgbs = torch.tensor([rgb[0].tolist() for rgb in batch])
#     depths = torch.tensor([depth[1].tolist() for depth in batch])
#     labels = torch.tensor([label[2].tolist() for label in batch])

#     return rgbs, depths, labels, masks

def L1_penalty(var):
    return torch.abs(var).sum()

def orthogonal_regularization(model, device, beta=1e-4):
    r"""
        author: Xu Mingle
        time: 2019年2月19日15:12:43
        input:
            model: which is the model we want to use orthogonal regularization, e.g. Generator or Discriminator
            device: cpu or gpu
            beta: hyperparameter
        output: loss
    """
    
    # beta * (||W^T.W * (1-I)||_F)^2 or 
    # beta * (||W.W.T * (1-I)||_F)^2
    # 若 H < W,可以使用前者， 若 H > W, 可以使用后者，这样可以适当减少内存
    
    
    loss_orth = torch.tensor(0., dtype=torch.float32, device=device)
    
    for name, param in model.named_parameters():
#         print('name is {}'.format(name))
#         print('shape is {}'.format(param.shape))
        if ('kernel_rgb' in name or 'kernel_depth' in name) and param.requires_grad and len(param.shape)==4:
        # 是weight，而不是bias
        # 当然是指定被训练的参数
        # 只对卷积层参数做这样的正则化，而不包括嵌入层（维度是2）等。
            
#             print('shape is {}'.format(param.shape))
#             print('name {}'.format(name))
            
            N, C, H, W = param.shape
#             print('param shape {}'.format(param.shape))
            
            weight = param.view(N * C, H, W)
#             print('flatten shape {}'.format(weight.shape))
            
            weight_squared = torch.bmm(weight, weight.permute(0, 2, 1)) # (N * C) * H * H
#             print('beta_squared shape {}'.format(weight_squared.shape))
            
            ones = torch.ones(N * C, H, H, dtype=torch.float32) # (N * C) * H * H
#             print('ones shape {}'.format(ones.shape))
            
            diag = torch.eye(H, dtype=torch.float32) # (N * C) * H * H
#             print('diag shape {}'.format(diag.shape))
            
            loss_orth += ((weight_squared * (ones - diag).to(device)) ** 2).sum()
            
    return loss_orth * beta

def get_train_dataset(config):
    """ Dataset And Augmentation
    """
    train_transform = transforms.Compose([
        # transforms.Resize(config['width'], config['height']),
        # transforms.Randomcrop((config['height'], config['width'])),
        # transforms.RandomRotation(13),
        # transforms.RandomPerspective((0.05, 0.1)),
        # transforms.RandomScale((0.5, 2.0)),
        transforms.Randomcrop((config['height'], config['width'])),
        # transforms.RandomResizedCrop(size=(config['height'], config['width']), scale=(0.8, 0.9)),
        transforms.ColorJitter(brightness=0.5, contrast=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    
    train_data = Datasets(root=config['train_data'], dataset=config['dataset'], image_set='train', transform=train_transform)
    
    return train_data
###################################################################################################################
def get_val_test_dataset(config, is_test, experiment='train', missing_rate=0, count=1):
    """ Dataset And Augmentation
    """
    if experiment == 'train':
        val_test_transform = transforms.Compose([
            transforms.CenterCrop((config['height'], config['width'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
    else:
        val_test_transform = transforms.Compose([
            # transforms.Randomcrop((config['height'], config['width'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
    # train_data = Datasets(root=config['train_date'], image_set='train', transform=train_transform)
    if is_test:
        val_test_data = Datasets(root=config['train_data'], dataset=config['dataset'], image_set='test', transform=val_test_transform, missing_rate=missing_rate, count=count)
    else:
        val_test_data = Datasets(root=config['train_data'], dataset=config['dataset'], image_set='val', transform=val_test_transform, missing_rate=missing_rate, count=count)
    return val_test_data


def validate(config, model, loader, device, metrics, criterion):
    metrics.reset()
    save_val_results = False
    if save_val_results:
        if not os.path.exists('results'):
            os.mkdir('results')
        denorm = utils.utils.Denormalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])

    img_id = 0
    cityscapes = Cityscapes()
    sunrgbd = SUNRGBD()

    interval_loss = 0

    with torch.no_grad():
        for i, (images, depths, labels, masks) in tqdm(enumerate(loader)):
            images = images.to(device, dtype=torch.float32)
            depths = depths.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            masks = masks.to(device)

            masks[:, 0] = 0
            # masks[:, 1] = 0

            # outputs = model(images, depths)
            if config['modality'] == 'rgb':
                outputs = model(images)
            elif config['modality'] == 'depth':
                outputs = model(depths)
            elif config['modality'] == 'missing':
                outputs, _, _, _, _ = model(images, depths, masks)
                # outputs = model(images, depths, masks)
            
            loss = criterion(outputs, labels.to(device))
            np_loss = loss.detach().cpu().numpy()
            interval_loss += np_loss
            
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)
            
            if save_val_results:
                for i in range(len(images)):
                    image = images[i].detach().cpu().numpy()
                    target = targets[i]
                    pred = preds[i]

                    image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
                    
                    target = cityscapes.decode_target(target).astype(np.uint8)
                    pred = cityscapes.decode_target(pred).astype(np.uint8)
                    # target = sunrgbd.decode_target(target).astype(np.uint8)
                    # pred = sunrgbd.decode_target(pred).astype(np.uint8)

                    Image.fromarray(image).save('results/crossdic_%.1f/%d_image.png' % (config['missing_rate'], img_id))
                    Image.fromarray(target).save('results/crossdic_%.1f/%d_target.png' % (config['missing_rate'], img_id))
                    Image.fromarray(pred).save('results/crossdic_%.1f/%d_pred.png' % (config['missing_rate'], img_id))

                    img_id += 1

        # print("val loss: ", interval_loss/config['save_step'])
        interval_loss = 0.0
        

        score = metrics.get_results()
        
    return score

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

def train(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = config['gpu_id']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Setup random seed
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)

    # module = importlib.import_module('models.' + config['model'])
    # model = getattr(module, config['model']+'_network')
    # Set up model
    model_map = {
        'deeplabv3_resnet50': network_feature_group.deeplabv3_resnet50,
        'deeplabv3plus_resnet50': network_feature_group.deeplabv3plus_resnet50,
        'deeplabv3_resnet101': network_feature_group.deeplabv3_resnet101,
        'deeplabv3plus_resnet101': network_feature_group.deeplabv3plus_resnet101,
        'deeplabv3_mobilenet': network_feature_group.deeplabv3_mobilenet,
        'deeplabv3plus_mobilenet': network_feature_group.deeplabv3plus_mobilenet
    }
    model = model_map['deeplabv3_resnet101'](config['num_classes'], output_stride=16)
    # print(model)

    ######################################## 修改
    # model = model(num_classes=config['num_classes'], expert_not_fixed=True)

    # pretrain_params = load_state_dict_from_url(model_urls['resnet50'])
    # model.load_state_dict(pretrain_params, strict=False)
    ######################################## 修改
    # for name, modules in model.named_parameters():
    #     if 'model1' in name or 'model2' in name:
    #         modules.requires_grad = True
    #     else:
    #         modules.requires_grad = True

    # encoder_layers = nn.ModuleList([model.backbone_rgb, model.backbone_depth])
    # # encoder_layers = model.model1
    # encoder_params = list(map(id, encoder_layers.parameters()))

    # cross_layers = nn.ModuleList([model.classifier.kernel_rgb, model.classifier.kernel_depth])
    # cross_params = list(map(id, cross_layers.parameters()))

    # decoder_params = filter(lambda p: id(p) not in cross_params and id(p) not in encoder_params, model.parameters())

    ######################################## 修改
    # optimizer = torch.optim.Adam(params=[
    #     {'params': encoder_layers.parameters(), 'lr': config['encoder_lr']},
    #     {'params': decoder_params, 'lr': config['decoder_lr']}], weight_decay=0.0005)
    # optimizer = torch.optim.SGD(params=[
    #     {'params': model.parameters(), 'lr': config['encoder_lr']}
    # ], lr=config['encoder_lr'], momentum=0.9, weight_decay=0.0001)

    # cross_layers = nn.ModuleList([model.classifier.kernel_rgb, model.classifier.kernel_depth])
    # cross_params = list(map(id, cross_layers.parameters()))
    # # print(cross_params)
    # decoder_params = filter(lambda p: id(p) not in cross_params, model.classifier.parameters())
    # print(cross_layers)
    optimizer = torch.optim.SGD(params=[
        {'params': model.backbone_rgb.parameters(), 'lr': config['encoder_lr']},
        {'params': model.backbone_depth.parameters(), 'lr': config['encoder_lr']},
        # {'params': cross_params, 'lr': 0.1 * config['encoder_lr']},
        # {'params': decoder_params, 'lr': config['encoder_lr']},
        {'params': model.classifier.parameters(), 'lr': config['encoder_lr']},
    ], lr=config['encoder_lr'], momentum=0.9, weight_decay=0.0001)
    # scheduler = utils.PolyLR(optimizer, max_iters=config['decay_steps'], last_epoch=, power=config['power'])
    scheduler = utils.PolyLR(optimizer, max_iters=config['max_iteration'], power=config['power'], min_lr=1e-6)

    metrics = utils.StreamSegMetrics(config['num_classes'])

    train_dst = get_train_dataset(config)
    print(len(train_dst))
    train_loader = data.DataLoader(
        train_dst, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_worker'])
    if not config['test_only']:
        val_dst = get_val_test_dataset(config, is_test=config['is_test'], experiment='train', missing_rate=0, count=1)
    # else:
    #     val_dst = get_val_test_dataset(config, is_test=config['is_test'], experiment='test')
        val_loader = data.DataLoader(
            val_dst, batch_size=config['val_test_batch_size'], shuffle=True, num_workers=0)

    # criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
    criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
    # content_loss = utils.PerceptualLoss(torch.nn.MSELoss())
    content_loss = nn.MSELoss()
    softmax = nn.Softmax(dim=1)
    relu = nn.ReLU(inplace=True)
    tanh = nn.Tanh()
    # content_loss = nn.KLDivLoss(reduction='batchmean')
    # align_loss = nn.KLDivLoss(reduction='batchmean')
    # content_loss = nn.CrossEntropyLoss()
    # sim_loss = utils.CMD()
    # diff_loss = utils.DiffLoss()

    def save_ckpt(path):
        """save current model"""
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)

    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0

    if config['checkpoint'] is not None and os.path.isfile(config['checkpoint']):
        checkpoint = torch.load(config['checkpoint'], map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        # print(model.classifier.ASPP_rgb.project[1].bias)
        model = nn.DataParallel(model)
        model.to(device)
        if config['continue_training']:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            cur_itrs = checkpoint["cur_itrs"]
            best_score = checkpoint['best_score']
            print("Training state restored from %s" % config['checkpoint'])
        else:
            print('Model Loaded')
        del checkpoint
    else:
        if 'checkpoint1' in config and 'checkpoint2' in config:
            modalities = {'rgb':config['checkpoint1'], 'depth':config['checkpoint2']} #######三种模态
            for modality in modalities:
                pretrain_params = torch.load(modalities[modality])
                model.load_state_dict(pretrain_params['model_state'], strict=False)
                model = nn.DataParallel(model)
                model.to(device)
                print('Pretrained Unimodal Model Initialization')
        else:
            print('[!] Retrain')
            model = nn.DataParallel(model)
            model.to(device)

    if config['test_only']:
        iterations = 1 if config['missing_rate'] == 0 else 5
        val_scores = {'Overall Acc': [], 'Mean Acc': [], 'Mean IoU': []}
        for i in range(iterations):
            val_dst = get_val_test_dataset(config, is_test=config['is_test'], experiment='test', missing_rate=config['missing_rate'], count=i+1)
            val_loader = data.DataLoader(
                val_dst, batch_size=config['val_test_batch_size'], shuffle=False, num_workers=0)
            model.eval()
            val_score = validate(config, model=model, loader=val_loader, device=device, metrics=metrics, criterion=criterion)
            val_scores['Overall Acc'].append(val_score['Overall Acc'])
            val_scores['Mean Acc'].append(val_score['Mean Acc'])
            val_scores['Mean IoU'].append(val_score['Mean IoU'])
            print('testing missing rate: %d%% + %d%%, iterations: %d' % (int(config['missing_rate']/2 * 100), int(config['missing_rate']/2 * 100), i+1))
            print(metrics.to_str(val_score))
        if iterations != 1:
            val_score_mean = {'Overall Acc': np.mean(val_scores['Overall Acc']), 'Mean Acc': np.mean(val_scores['Mean Acc']), 'Mean IoU': np.mean(val_scores['Mean IoU'])}
            print('testing missing rate: %d%% + %d%%' % (int(config['missing_rate']/2 * 100), int(config['missing_rate']/2 * 100)))
            print(metrics.to_str(val_score_mean))
        return

    interval_loss1 = 0
    interval_loss2 = 0
    interval_loss3 = 0
    interval_loss4 = 0
    interval_loss5 = 0
    # writer = SummaryWriter(log_dir='./logs')

    while 1:
        model.train()
        cur_epochs += 1
        rgb_num, depth_num = 0, 0
        for (images, depths, labels, masks) in train_loader:
            cur_itrs += 1
            images = images.to(device, dtype=torch.float32)
            depths = depths.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            masks = masks.to(device)
            
            ##############################################################  random missing
            missing_rate = random.uniform(0, 1.0)
            sample_random = np.random.uniform(low=0, high=1.0, size=(masks.shape[0]))
            index = np.where(sample_random < missing_rate)
            modality_random = np.random.uniform(low=0, high=1.0, size=(len(index[0])))
            modality1_index = np.where(modality_random <= 0.5)
            modality2_index = np.where(modality_random > 0.5)

            masks[:, 0][index[0][modality1_index]] = 0
            masks[:, 1][index[0][modality2_index]] = 0
            #print(masks)
            # print(((masks[:, 0]==1)==True).sum(
            ###############################################################
            
            optimizer.zero_grad()


            # outputs = model(images, depths)
            # outputs, encoder_out = model(images)
            if config['modality'] == 'rgb':
                outputs = model(images)
            elif config['modality'] == 'depth':
                outputs = model(depths)
            elif config['modality'] == 'missing':
                outputs, rgb_pair, depth_pair, sparse_list, sim_list = model(images, depths, masks)
                # outputs = model(images, depths, masks)
            # with writer:
                # writer.add_graph(model, input_to_model=(images, depths, masks))
            # writer.add_image('img/img1', images[0], dataformats='CHW')
                # writer.add_graph(model, input_to_model=(images.tensors, depths.tensors))

            # out_rgb = nn.functional.conv2d(rgb, weight=model.module.classifier.classifier.weight[:, :256], bias=model.module.classifier.classifier.bias/2)
            # out_depth = nn.functional.conv2d(depth, weight=model.module.classifier.classifier.weight[:, 256:], bias=model.module.classifier.classifier.bias/2)
            
            rgb_loss = content_loss(rgb_pair[0], rgb_pair[1])
            depth_loss = content_loss(depth_pair[0], depth_pair[1])
            # cross_loss = (rgb_loss + depth_loss) / 2.0
            cross_loss = rgb_loss + depth_loss

            sim_loss = content_loss(sim_list[0], sim_list[1])
            

            pred_loss = criterion(outputs, labels)

            orth_loss = orthogonal_regularization(model, device, beta=1e-4)
            L1_norm = sum([L1_penalty(m).cuda() for m in sparse_list])

            loss = pred_loss + cross_loss + 2e-4 * L1_norm + orth_loss + sim_loss
            loss.backward()
            optimizer.step()

            np_loss1 = pred_loss.detach().cpu().numpy()
            interval_loss1 += np_loss1
            
            np_loss2 = cross_loss.detach().cpu().numpy()
            interval_loss2 += np_loss2

            np_loss3 = (2e-4 * L1_norm).detach().cpu().numpy()
            interval_loss3 += np_loss3

            np_loss4 = orth_loss.detach().cpu().numpy()
            interval_loss4 += np_loss4
            
            np_loss5 = sim_loss.detach().cpu().numpy()
            interval_loss5 += np_loss5

            if (cur_itrs) % config['skip_step'] == 0:
                interval_loss1 = interval_loss1/config['skip_step']
                interval_loss2 = interval_loss2/config['skip_step']
                interval_loss3 = interval_loss3/config['skip_step']
                interval_loss4 = interval_loss4/config['skip_step']
                interval_loss5 = interval_loss5/config['skip_step']
                print("Epoch %d, Itrs %d/%d, pred_loss=%f,cross_loss=%f,l1=%f,orth=%f,sim=%f" % (cur_epochs, cur_itrs, config['max_iteration'], interval_loss1, interval_loss2, interval_loss3, interval_loss4, interval_loss5))
                interval_loss1 = 0.0
                interval_loss2 = 0.0
                interval_loss3 = 0.0
                interval_loss4 = 0.0
                interval_loss5 = 0.0


            if (cur_itrs) % config['save_step'] == 0:
                # save_ckpt('checkpoints/deeplabv3_latest_%s_%s.pth' % (config['model'], config['dataset']))
                save_ckpt('checkpoints/sim_orth_bnsigmoid_loss_2_feautre_2e-4l1sparse_kernel_3_randommiss_lr_0.01_60000_bs16_enc_1_1_resnet101_deeplabv3_latest_%s_%s.pth' % (config['modality'], config['dataset']))
                print("validation...")
                model.eval()
                val_score = validate(config, model=model, loader=val_loader, device=device, metrics=metrics, criterion=criterion)
                print(metrics.to_str(val_score))
                if val_score['Mean IoU'] > best_score:
                    best_score = val_score['Mean IoU']
                    # save_ckpt('checkpoints/deeplabv3_best_%s_%s.pth' % (config['model'], config['dataset']))
                    save_ckpt('checkpoints/sim_orth_bnsigmoid_loss_2_feature_2e-4l1sparse_kernel_3_randommiss_lr_0.01_60000_bs16_enc_1_1_resnet101_deeplabv3_best_%s_%s.pth' % (config['modality'], config['dataset']))

                model.train()
            scheduler.step()

            if cur_itrs >= config['max_iteration']:
                return


def main():
    args = parser.parse_args()
    if args.config:
        file_address = open(args.config)
        config = yaml.safe_load(file_address)
    else:
        print('--config config_file_address missing')
    train(config)


if __name__ == '__main__':
    main()
