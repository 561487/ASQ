import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import sys
import argparse
from sklearn.metrics import roc_auc_score
from data_loader import TrainDataLoader, ValTestDataLoader
from model import Net
from build_q_matrix import build_expert_q_matrix

# will be overwritten by config.txt
exer_n = 17746
knowledge_n = 123
student_n = 4163

device = torch.device(('cuda:0') if torch.cuda.is_available() else 'cpu')
epoch_n = 5

def train(use_dynamic_q=False, d_model=128, use_q_init=False, patience=5):
    """
    训练函数
    
    参数:
        use_dynamic_q: 是否使用 DynamicQ（稀疏 Q）替换静态专家 Q
        d_model: DynamicQ 的嵌入维度
        use_q_init: 是否使用专家 Q 矩阵初始化 DynamicQ
        patience: Early stopping 的耐心值，连续多少个 epoch 没有提升就停止（默认5，设为0或负数表示不早停）
    """
    data_loader = TrainDataLoader()
    
    # 构建专家 Q 矩阵（如果使用 DynamicQ 且需要初始化）
    q_init = None
    if use_dynamic_q and use_q_init:
        print("Building expert Q matrix from training data...")
        q_init = build_expert_q_matrix('data/train_set.json', exer_n, knowledge_n)
        print(f"Expert Q matrix shape: {q_init.shape}")
    
    # 初始化模型
    net = Net(
        student_n, 
        exer_n, 
        knowledge_n,
        use_dynamic_q=use_dynamic_q,
        d_model=d_model,
        q_init=q_init
    )
    net = net.to(device)
    
    if use_dynamic_q:
        print(f"✅ Using DynamicQ (sparse Q) with d_model={d_model}")
        if use_q_init:
            print("✅ Using expert Q matrix for initialization")
    else:
        print("✅ Using static expert Q matrix (original NCDM)")

    optimizer = optim.Adam(net.parameters(), lr=0.002)
    loss_function = nn.BCELoss()
    print('training model...')

    # 记录最佳验证集 AUC 和对应的 epoch
    best_auc = -1.0
    best_epoch = 1
    no_improve_count = 0  # 连续没有提升的 epoch 数

    for epoch in range(epoch_n):
        data_loader.reset()
        running_loss = 0.0
        batch_count = 0

        while not data_loader.is_end():
            batch_count += 1
            input_stu_ids, input_exer_ids, input_knowledge_embs, labels = data_loader.next_batch()
            input_stu_ids = input_stu_ids.to(device)
            input_exer_ids = input_exer_ids.to(device)
            input_knowledge_embs = input_knowledge_embs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            output_1 = net.forward(input_stu_ids, input_exer_ids, input_knowledge_embs)  # shape [B, 1]
            loss = loss_function(output_1.view(-1), labels.float())  # Flatten to [B]

            if torch.isnan(loss):
                print("NaN in loss! Exiting.")
                exit(1)

            loss.backward()
            optimizer.step()
            net.apply_clipper()

            running_loss += loss.item()
            if batch_count % 200 == 199:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_count + 1, running_loss / 200))
                running_loss = 0.0

        # 验证 & 保存模型
        rmse, auc = validate(net, epoch, use_dynamic_q, d_model, use_q_init)
        save_snapshot(net, 'model/model_epoch' + str(epoch + 1))
        
        # 记录最佳 epoch
        if auc > best_auc:
            best_auc = auc
            best_epoch = epoch + 1
            no_improve_count = 0  # 重置计数
            # 保存最佳模型为 best_model.pt
            save_snapshot(net, 'model/best_model.pt')
            print(f'[INFO] New best AUC: {best_auc:.4f} at epoch {best_epoch}')
        else:
            no_improve_count += 1
        
        # Early stopping 检查（如果启用了早停）
        if patience > 0 and no_improve_count >= patience:
            print(f'\n[Early Stopping] No improvement for {patience} epochs, stopping training.')
            print(f'[INFO] Best validation AUC: {best_auc:.4f} at epoch {best_epoch}')
            break
    
    # 将最佳 epoch 写入文件，供 predict.py 读取
    with open('model/best_epoch.txt', 'w') as f:
        f.write(str(best_epoch))
    
    if no_improve_count < patience or patience <= 0:
        print(f'\n[INFO] Training completed. Best validation AUC: {best_auc:.4f} at epoch {best_epoch}')
    print(f'[INFO] Best model saved to model/best_model.pt')


def validate(model, epoch, use_dynamic_q=False, d_model=128, use_q_init=False):
    """
    验证函数
    
    参数:
        model: 训练中的模型
        epoch: 当前 epoch
        use_dynamic_q: 是否使用 DynamicQ（必须与训练时一致）
        d_model: DynamicQ 的嵌入维度（必须与训练时一致）
        use_q_init: 是否使用专家 Q 矩阵初始化（必须与训练时一致）
    """
    data_loader = ValTestDataLoader('validation')
    
    # 构建专家 Q 矩阵（如果使用 DynamicQ 且需要初始化）
    q_init = None
    if use_dynamic_q and use_q_init:
        q_init = build_expert_q_matrix('data/train_set.json', exer_n, knowledge_n)
    
    # 初始化模型（必须与训练时参数一致）
    net = Net(
        student_n, 
        exer_n, 
        knowledge_n,
        use_dynamic_q=use_dynamic_q,
        d_model=d_model,
        q_init=q_init
    )
    print('validating model...')

    data_loader.reset()
    net.load_state_dict(model.state_dict())
    net = net.to(device)
    net.eval()

    correct_count, exer_count = 0, 0
    pred_all, label_all = [], []

    while not data_loader.is_end():
        input_stu_ids, input_exer_ids, input_knowledge_embs, labels = data_loader.next_batch()
        input_stu_ids = input_stu_ids.to(device)
        input_exer_ids = input_exer_ids.to(device)
        input_knowledge_embs = input_knowledge_embs.to(device)
        labels = labels.to(device)

        output = net.forward(input_stu_ids, input_exer_ids, input_knowledge_embs)  # [B,1]
        output = output.view(-1)

        # Accuracy
        preds = (output > 0.5).float()
        correct_count += torch.sum(preds == labels).item()
        exer_count += len(labels)

        pred_all += output.detach().cpu().tolist()
        label_all += labels.detach().cpu().tolist()

    pred_all = np.array(pred_all)
    label_all = np.array(label_all)

    # 检查 NaN
    if np.isnan(pred_all).any() or np.isnan(label_all).any():
        print("NaN in predictions or labels. Skipping AUC/RMSE.")
        return 0.0, 0.0

    accuracy = correct_count / exer_count
    rmse = np.sqrt(np.mean((label_all - pred_all) ** 2))
    auc = roc_auc_score(label_all, pred_all)

    print('epoch= %d, accuracy= %.4f, rmse= %.4f, auc= %.4f' % (epoch+1, accuracy, rmse, auc))
    with open('result/model_val.txt', 'a', encoding='utf8') as f:
        f.write('epoch= %d, accuracy= %.4f, rmse= %.4f, auc= %.4f\n' % (epoch+1, accuracy, rmse, auc))

    return rmse, auc


def save_snapshot(model, filename):
    torch.save(model.state_dict(), filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train NCDM with optional DynamicQ')
    parser.add_argument('device', type=str, help='Device (cpu or cuda:0)')
    parser.add_argument('epoch', type=int, help='Number of epochs')
    parser.add_argument('--use_dynamic_q', action='store_true', 
                       help='Use DynamicQ (sparse Q) instead of static expert Q')
    parser.add_argument('--d_model', type=int, default=128,
                       help='Embedding dimension for DynamicQ (default: 128)')
    parser.add_argument('--use_q_init', action='store_true',
                       help='Initialize DynamicQ with expert Q matrix')
    parser.add_argument('--patience', type=int, default=0,
                       help='Early stopping patience (default: 0 = no early stopping)')
    
    args = parser.parse_args()
    
    if (args.device != 'cpu') and ('cuda:' not in args.device):
        print('Error: device must be "cpu" or "cuda:X"')
        exit(1)
    
    device = torch.device(args.device)
    epoch_n = args.epoch

    with open('config.txt') as i_f:
        i_f.readline()
        student_n, exer_n, knowledge_n = list(map(eval, i_f.readline().split(',')))

    train(
        use_dynamic_q=args.use_dynamic_q,
        d_model=args.d_model,
        use_q_init=args.use_q_init,
        patience=args.patience
    )
