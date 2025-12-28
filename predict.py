import torch
import numpy as np
import json
import sys
import os
import argparse
from sklearn.metrics import roc_auc_score
from data_loader import ValTestDataLoader
from model import Net
from build_q_matrix import build_expert_q_matrix


# can be changed according to config.txt
exer_n = 17746
knowledge_n = 123
student_n = 4163


def test(epoch=None, use_dynamic_q=False, d_model=128, use_q_init=False, use_best=False):
    """
    测试函数
    
    参数:
        epoch: 模型 epoch 编号（如果 use_best=True，则忽略此参数，使用最佳epoch）
        use_dynamic_q: 是否使用 DynamicQ（必须与训练时一致）
        d_model: DynamicQ 的嵌入维度（必须与训练时一致）
        use_q_init: 是否使用专家 Q 矩阵初始化（必须与训练时一致）
        use_best: 是否使用验证集上表现最好的 epoch（默认 False，使用指定的 epoch）
    """
    data_loader = ValTestDataLoader('test')
    
    # 构建专家 Q 矩阵（如果使用 DynamicQ 且需要初始化）
    q_init = None
    if use_dynamic_q and use_q_init:
        print("Building expert Q matrix from training data...")
        q_init = build_expert_q_matrix('data/train_set.json', exer_n, knowledge_n)
        print(f"Expert Q matrix shape: {q_init.shape}")
    
    # 确定使用哪个 epoch
    if use_best:
        # 读取最佳 epoch
        best_epoch_file = 'model/best_epoch.txt'
        if os.path.exists(best_epoch_file):
            with open(best_epoch_file, 'r') as f:
                best_epoch = int(f.read().strip())
            print(f'[INFO] Using best epoch from validation: {best_epoch}')
            model_path = 'model/best_model.pt'
        else:
            print('[WARN] best_epoch.txt not found, falling back to specified epoch or last epoch')
            use_best = False
    
    if not use_best:
        if epoch is None:
            raise ValueError("epoch must be specified if use_best=False")
        best_epoch = epoch
        model_path = 'model/model_epoch' + str(epoch)
    
    # 初始化模型（必须与训练时参数一致）
    net = Net(
        student_n, 
        exer_n, 
        knowledge_n,
        use_dynamic_q=use_dynamic_q,
        d_model=d_model,
        q_init=q_init
    )
    device = torch.device('cpu')
    print('testing model...')
    data_loader.reset()
    load_snapshot(net, model_path)
    net = net.to(device)
    net.eval()
    
    if use_dynamic_q:
        print(f"✅ Using DynamicQ (sparse Q) with d_model={d_model}")
    else:
        print("✅ Using static expert Q matrix (original NCDM)")

    correct_count, exer_count = 0, 0
    pred_all, label_all = [], []
    while not data_loader.is_end():
        input_stu_ids, input_exer_ids, input_knowledge_embs, labels = data_loader.next_batch()
        input_stu_ids, input_exer_ids, input_knowledge_embs, labels = input_stu_ids.to(device), input_exer_ids.to(
            device), input_knowledge_embs.to(device), labels.to(device)
        out_put = net(input_stu_ids, input_exer_ids, input_knowledge_embs)
        out_put = out_put.view(-1)
        # compute accuracy
        for i in range(len(labels)):
            if (labels[i] == 1 and out_put[i] > 0.5) or (labels[i] == 0 and out_put[i] < 0.5):
                correct_count += 1
        exer_count += len(labels)
        pred_all += out_put.tolist()
        label_all += labels.tolist()

    pred_all = np.array(pred_all)
    label_all = np.array(label_all)
    # compute accuracy
    accuracy = correct_count / exer_count
    # compute RMSE
    rmse = np.sqrt(np.mean((label_all - pred_all) ** 2))
    # compute AUC
    auc = roc_auc_score(label_all, pred_all)
    print('epoch= %d, accuracy= %f, rmse= %f, auc= %f' % (best_epoch, accuracy, rmse, auc))
    with open('result/model_test.txt', 'a', encoding='utf8') as f:
        f.write('epoch= %d, accuracy= %f, rmse= %f, auc= %f\n' % (best_epoch, accuracy, rmse, auc))


def load_snapshot(model, filename):
    f = open(filename, 'rb')
    model.load_state_dict(torch.load(f, map_location=lambda s, loc: s))
    f.close()


def get_status():
    '''
    An example of getting student's knowledge status
    :return:
    '''
    net = Net()
    load_snapshot(net, 'model/model_epoch12')       # load model
    net.eval()
    with open('result/student_stat.txt', 'w', encoding='utf8') as output_file:
        for stu_id in range(student_n):
            # get knowledge status of student with stu_id (index)
            status = net.get_knowledge_status(torch.LongTensor([stu_id])).tolist()[0]
            output_file.write(str(status) + '\n')


def get_exer_params():
    '''
    An example of getting exercise's parameters (knowledge difficulty and exercise discrimination)
    :return:
    '''
    net = Net()
    load_snapshot(net, 'model/model_epoch12')    # load model
    net.eval()
    exer_params_dict = {}
    for exer_id in range(exer_n):
        # get knowledge difficulty and exercise discrimination of exercise with exer_id (index)
        k_difficulty, e_discrimination = net.get_exer_params(torch.LongTensor([exer_id]))
        exer_params_dict[exer_id + 1] = (k_difficulty.tolist()[0], e_difficulty.tolist()[0])
    with open('result/exer_params.txt', 'w', encoding='utf8') as o_f:
        o_f.write(str(exer_params_dict))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test NCDM with optional DynamicQ')
    parser.add_argument('epoch', type=int, nargs='?', default=None,
                       help='Model epoch number (optional if --use_best is set)')
    parser.add_argument('--use_dynamic_q', action='store_true',
                       help='Use DynamicQ (must match training settings)')
    parser.add_argument('--d_model', type=int, default=128,
                       help='Embedding dimension for DynamicQ (must match training, default: 128)')
    parser.add_argument('--use_q_init', action='store_true',
                       help='Initialize DynamicQ with expert Q matrix (must match training)')
    parser.add_argument('--use_best', action='store_true',
                       help='Use best epoch from validation set (recommended)')
    
    args = parser.parse_args()

    # 如果指定了 --use_best，epoch 参数可选
    if not args.use_best and args.epoch is None:
        parser.error("Either specify epoch or use --use_best flag")

    # global student_n, exer_n, knowledge_n
    with open('config.txt') as i_f:
        i_f.readline()
        student_n, exer_n, knowledge_n = list(map(eval, i_f.readline().split(',')))

    test(
        epoch=args.epoch,
        use_dynamic_q=args.use_dynamic_q,
        d_model=args.d_model,
        use_q_init=args.use_q_init,
        use_best=args.use_best
    )
