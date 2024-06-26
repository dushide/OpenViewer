import argparse
import copy
import os

import scipy
import warnings
import sys
import random
import numpy as np
import torch
from sklearn import metrics

sys.path.append("./util")
from evaluation_metrics import EvaluationMetrics
from config import load_config
from loadMatData import load_data, sparse_mx_to_torch_sparse_tensor
import scipy  as sp
from label_utils import reassign_labels, special_train_test_split
from data import Multi_view_data, generate_partition
from models import Net, tensor_center_loss

np.set_printoptions(threshold=np.inf)
import torch.nn.functional as F

def mixup_data(x, y, alpha=0.3, use_cuda=True):
    lam = np.random.beta(alpha, alpha)
    batch_size = y.size()[0]
    sample = 0
    trynum = 0
    while (sample < batch_size / 2 and trynum < 10):
        if use_cuda:
            index = torch.randperm(batch_size).cuda()
        else:
            index = torch.randperm(batch_size)
        #   剔除同类对
        index_dc = torch.nonzero(y[index] != y).squeeze()
        sample = len(index_dc)
        trynum += 1
    mixed_x = {}
    mixed_x[0] = lam * x[0][index_dc] + (1 - lam) * x[0][index][index_dc]
    for i in range(1, len(x)):
        if use_cuda:
            index2 = torch.randperm(batch_size).cuda()
        else:
            index2 = torch.randperm(batch_size)
        mixed_x[i] = lam * x[i][index_dc] + (1 - lam) * x[i][index2][index_dc]
    return mixed_x, y[index_dc], lam


def main(args, device):
    def valid(model, loader, device):
        model.eval()
        with torch.no_grad():
            label = []
            test_prob = []
            correct, num_samples = 0, 0
            for batch in loader:
                x, y = batch['x'], batch['y']
                for k in x.keys():
                    x[k] = x[k].to(device)

                evidence = model.infer(x)
                prob = F.softmax(evidence, 1)
                test_prob.append(prob.cpu().numpy())
                pred_y = prob.argmax(dim=-1)
                correct += torch.sum(pred_y.cpu().eq(batch['y'])).item()
                num_samples += len(batch['y'])
                label.append(batch['y'].cpu().numpy())
        prob = np.concatenate(test_prob)
        label = np.concatenate(label)
        test_truth_label_open = [NCLASSES if i == unseen_label_index else i for i in label]
        softmax_ccr, softmax_fpr, softmax_ccrs = EvaluationMetrics.ccr_at_fpr(np.array(test_truth_label_open),
                                                                              prob,
                                                                              NCLASSES)

        return softmax_ccrs[-1]

    def train(model, train_loader, valid_loader, device):
        model = model.to(device)

        optimizer = torch.optim.Adam([
            {'params': (p for n, p in model.named_parameters() if p.requires_grad and 'weight' in n),
             'weight_decay': 1e-2},
            {'params': (p for n, p in model.named_parameters() if p.requires_grad and 'weight' not in n)},
        ], lr=args.learning_rate)
        step_lr = torch.optim.lr_scheduler.StepLR(optimizer, step_size=23, gamma=0.1)
        loss_list = list()
        best_valid_ccr = 0.0
        criterion = torch.nn.CrossEntropyLoss()
        euclidean_dist_obj = torch.nn.PairwiseDistance(p=2).to(device)
        for epoch in range(1, args.num_epoch + 1):
            model.train()
            train_loss, correct, num_samples = 0, 0, 0
            train_pred_label = []
            output = []
            for batch in train_loader:
                x, y, index = batch['x'], batch['y'], batch['index']
                for k in x.keys():
                    x[k] = x[k].to(device)
                y = y.long().to(device)

                x_mix, y_mix, lam = mixup_data(x, y)

                x_all = {}
                for i in range(len(x)):
                    x_all[i] = torch.cat([x[i], x_mix[i]], dim=0)

                evidence, centers = model(x_all, y, epoch)

                prob = F.softmax(evidence, dim=1)[:y.shape[0]]

                loss_center = euclidean_dist_obj(evidence[:y.shape[0]], centers[y, :]).mean()

                loss_Oen = 0
                for c in range(num_classes):
                    target_c = torch.LongTensor(y_mix.shape[0]).random_(c, c + 1).to(device)
                    loss_Oen += criterion(evidence[y.shape[0]:], target_c)

                mag = evidence[:y.shape[0]].norm(p=2, dim=1)
                mag_diff_from_ring = torch.clamp(args.knownsMinimumMag - mag, min=0.0)
                loss_ce = criterion(evidence[:y.shape[0]], y) + args.u * torch.norm(mag_diff_from_ring, 2).mean()

                mag_1 = evidence[y.shape[0]:].norm(p=2, dim=1)
                loss = loss_ce + alpha * (loss_Oen / num_classes + args.u * torch.pow(mag_1, 2).mean()) + beta * loss_center

                optimizer.zero_grad()

                loss.backward()
                optimizer.step()
                train_loss += loss.item()

                train_pred_label.append(y.cpu().detach().numpy())
                output.append(prob.cpu().detach().numpy())

                num_samples += len(y)
                correct += torch.sum(prob.argmax(dim=-1).eq(y)).item()

            train_loss = train_loss
            loss_list.append(train_loss)
            train_acc = correct/num_samples

            valid_ccr = valid(model, valid_loader, device)
            if valid_ccr:
                if best_valid_ccr < valid_ccr:
                    best_valid_ccr = valid_ccr
                    best_model_wts = copy.deepcopy(model.state_dict())
            else:
                valid_ccr = 0.0

            step_lr.step()
            print(
                f'Epoch {epoch:3d}: lr {step_lr.get_last_lr()[0]:.6f}, train loss: {train_loss:.4f}, train acc: {train_acc:.4f}, valid ccr: {valid_ccr:.4f}')

        model.load_state_dict(best_model_wts)
        test_Z, softmax_scores, test_truth_label = test(model, test_loader, device)
        test_truth_label_open = [NCLASSES if i == unseen_label_index else i for i in test_truth_label]
        softmax_pred_label = np.argmax(softmax_scores, 1)
        softmax_ACC = metrics.accuracy_score(test_truth_label_open,
                                             softmax_pred_label)
        softmax_ccr, softmax_fpr, softmax_ccrs = EvaluationMetrics.ccr_at_fpr(np.array(test_truth_label_open),
                                                                              softmax_scores,
                                                                              NCLASSES)
        if saveResp:
            print("save %s/%s_epoch%d.mat......" % (path, data, epoch))
            data_path = "%s/" % (path)
            if not os.path.exists(data_path):
                os.makedirs(data_path)
            scipy.io.savemat("%s/%s_epoch%d.mat" % (data_path, data, epoch),
                             {'test_y': test_truth_label_open, 'test_Z': test_Z,
                              'label_open': test_truth_label_open, 'label': test_truth_label,
                              'softmax_ccr': softmax_ccr.tolist(), 'softmax_fpr': softmax_fpr.tolist(),
                              'resp': test_Z
                              })
        with open(file, "a") as f:
            f.write(
                'epoch:{},knownsMinimumMag:{},u:{},openness:{}, unseen_num:{},alpha:{},beta:{}\n'.format(
                    args.num_epoch, args.knownsMinimumMag, args.u, args.openness, unseen_num, alpha, beta))
            f.write(' dataset:{},softmax_acc:{},softmax_ccrs:{}\n'.format(
                data, round(
                    softmax_ACC * 100,
                    2), softmax_ccrs))
            f.write('loss:{}\n'.format(loss_list))



        return model

    def test(model, loader, device):
        model.eval()
        with torch.no_grad():
            test_Z = []
            test_prob = []
            label = []
            correct, num_samples = 0, 0
            for batch in loader:
                x, y = batch['x'], batch['y']
                for k in x.keys():
                    x[k] = x[k].to(device)

                evidence = model.infer(x)
                prob = F.softmax(evidence, 1)
                test_Z.append(evidence.cpu().numpy())
                test_prob.append(prob.cpu().numpy())
                label.append(batch['y'].cpu().numpy())
                pred_y = prob.argmax(dim=-1)
                correct += torch.sum(pred_y.cpu().eq(batch['y'])).item()
                num_samples += len(batch['y'])
        label = np.concatenate(label)
        prob = np.concatenate(test_prob)
        test_Z = np.concatenate(test_Z)
        return test_Z, prob, label

    model = Net(n_feats, n_view, num_classes, args.thre, device)

    print('---------------------------- Experiment ------------------------------')
    print('Number of views:', len(train_data.x), ' views with dims:', [v.shape[-1] for v in train_data.x.values()])
    print('Number of training samples:', len(train_data))
    print('Number of validating samples:', len(valid_data))
    print('Trainable Parameters:')
    for n, p in model.named_parameters():
        print('%-40s' % n, '\t', p.data.shape)
    print('----------------------------------------------------------------------')
    train(model, train_loader, valid_loader, device)



if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=50, metavar='N',
                        help='input batch size for training')
    parser.add_argument('--num_epoch', type=int, default=100, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--u', type=float, default=0.05, help='mu')
    parser.add_argument('--knownsMinimumMag', type=float, default=5, help='knownsMinimumMag')
    parser.add_argument('--openness', type=float, default=0.1, help='openness')

    parser.add_argument('--learning_rate', type=float, default=1e-2, metavar='LR',
                        help='learning rate')
    parser.add_argument('--thre', type=float, default=0.01, metavar='thre',
                        help='thre')
    parser.add_argument("--save_file", type=str, default="res_TMC0312.txt")
    parser.add_argument('--save_results', default=True)
    parser.add_argument('--fix_seed', default=True)
    args = parser.parse_args()

    args.device = '0'
    device = torch.device('cpu' if args.device == 'cpu' else 'cuda:' + args.device)

    # tab_printer(args)

    dataset_dict = {1: "AwA", 2: 'animals', 3: 'esp_game', 4: 'VGGFace2_50', 5: 'NUSWide20k', 6: 'NUSWIDEOBJ', }

    select_dataset = [1, 2, 3, 4, 5, 6]
    training_rate = 0.1
    valid_rate = 0.1

    unseen_label_index = -100
    train_seed = 100
    saveResp = True

    file = './result/openness%.1f_res.txt' % (args.openness)
    path = './result/openness%.1f' % (args.openness)
    args.path = path
    args.file = file

    for ii in select_dataset:
        data = dataset_dict[ii]
        args.data = data
        config = load_config('./config/' + data)
        alpha = config['alpha']
        beta = config['beta']

        features, labels = load_data(dataset_dict[ii], 'E:/code/data/')
        n_view = len(features)
        n_feats = [x.shape[1] for x in features]
        n = features[0].shape[0]
        n_classes = len(np.unique(labels))

        print(data, n, n_view, n_feats)

        open2 = (1 - args.openness) * (1 - args.openness)
        unseen_num = round((1 - open2 / (2 - open2)) * n_classes)
        print("unseen_num:%d" % unseen_num)

        original_num_classes = len(np.unique(labels))
        seen_labels = list(range(original_num_classes - unseen_num))
        y_true = reassign_labels(labels, seen_labels, unseen_label_index)

        train_indices, test_valid_indices = special_train_test_split(y_true, unseen_label_index,
                                                                     test_size=1 - training_rate)
        valid_indices, test_indices = generate_partition(y_true[test_valid_indices], test_valid_indices,
                                                         (valid_rate) / (1 - training_rate))

        num_classes = np.max(y_true) + 1
        NCLASSES = num_classes
        print('data:{}\tseen_labels:{}\trandom_seed:{}\tunseen_num:{}\tnum_classes:{}'.format(
            data,
            seen_labels,
            train_seed,
            unseen_num,
            num_classes))

        train_data = Multi_view_data(n_view, train_indices, features, y_true)
        valid_data = Multi_view_data(n_view, valid_indices, features, y_true)
        test_data = Multi_view_data(n_view, test_indices, features, y_true)

        train_loader = torch.utils.data.DataLoader(train_data
                                                   , batch_size=args.batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_data
                                                   , batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_data
                                                  , batch_size=args.batch_size, shuffle=False)

        labels = torch.from_numpy(labels).long().to(device)
        y_true = torch.from_numpy(y_true).to(device)
        train_indices = torch.LongTensor(train_indices).to(device)

        N_mini_batches = len(train_loader)
        print('The number of training images = %d' % N_mini_batches)
        args.num_classes = num_classes
        args.seen_labels = seen_labels
        args.unseen_label_index = unseen_label_index

        if args.fix_seed:
            seed = 20
            torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
            torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        main(args, device)
        with open(file, "a") as f:
            f.write('\n')
