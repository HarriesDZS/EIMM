# -*-coding:utf-8-*-
"""
训练同步分割和分类的网络
"""

from torch.utils.data import DataLoader,RandomSampler,SequentialSampler
import torch
from tqdm import tqdm
import numpy as np


from InterpreJointNetwork import InterpreStarJoinNet
from Loss_functions import JI_and_Focal_loss,AutomaticWeightedLoss
from utils import EarlyStopping,DiceLoss
from transform import MedicalTransformCompose

device_id = 0


def train(data_loader, net, criterion, optimizer):
    """
    模型训练的具体方法
    :param data_loader:
    :param net:
    :param scheduler:
    :param early_stopping:
    :param criterion
    :return:
    """
    tbar = tqdm(data_loader, ascii=True, desc='train', dynamic_ncols=True)
    for batch_idx,(case_id,  ct_data, aug_ct_data, aug_same_id_ct_data, aug_diff_id_ct_data, label, cls_label, edge_label) in enumerate(tbar):
        ct_data = ct_data.cuda(device=device_id)
        aug_ct_data = aug_ct_data.cuda(device=device_id)
        aug_same_id_ct_data = aug_same_id_ct_data.cuda(device=device_id)
        aug_diff_id_ct_data = aug_diff_id_ct_data.cuda(device=device_id)
        label = label.type(torch.LongTensor)
        label = label.cuda(device=device_id)
        cls_label = cls_label.cuda(device=device_id)
        edge_label = edge_label.type(torch.LongTensor)
        edge_label = edge_label.cuda(device=device_id)
        seg, cls, recover, seg_edge, x_feature, x_aug_feature, x_same_patient_feature, x_othert_class_feature = net(ct_data, aug_ct_data, aug_same_id_ct_data, aug_diff_id_ct_data)

        seg_loss = criterion[0](seg, label)
        cls_loss = criterion[1](cls, cls_label)
        edge_loss = criterion[2](seg_edge, edge_label)
        recover_loss = criterion[3](recover, ct_data*seg.argmax(dim=1))
        sim_loss1 = 1 - criterion[4](x_feature, x_aug_feature).sum(axis=0)
        sim_loss2 = torch.min(1-criterion[4](x_feature, x_same_patient_feature), torch.tensor([0.2]*x_feature.shape[0]).cuda(device=device_id)).sum(axis=0)
        sim_loss3 = criterion[4](x_feature, x_othert_class_feature).sum(axis=0)
        supply_loss = edge_loss + recover_loss + sim_loss1 + sim_loss2 + sim_loss3

        loss = criterion[5](seg_loss, cls_loss, supply_loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        tbar.set_postfix({"loss":loss.item(), "seg_loss":seg_loss.item(), "cls_loss":cls_loss.item(), "supply_loss":supply_loss.item()})
        tbar.update(1)


def evaluate(data_loader, net, criterion, type):
    """
    验证数据集的方法
    :param data_loader:
    :param net:
    :param criterion:
    :param type:
    :return:
    """

    criterion_temp = AutomaticWeightedLoss(2).cuda(device=device_id)

    tbar = tqdm(data_loader, ascii=True, desc="[EVAL]{}".format(type), dynamic_ncols=True)
    anchor_case_id = -1
    predicts = []
    labels = []
    loss_list = []
    dice_list = []
    case_list = []
    recall_list = []
    precision_list = []

    cls_predict_list = []
    cls_label_list = []



    for batch_idx, (case_id,  ct_data, aug_ct_data, aug_same_id_ct_data, aug_diff_id_ct_data, label, cls_label, edge_label) in enumerate(tbar):
        ct_data = ct_data.cuda(device=device_id)
        aug_ct_data = aug_ct_data.cuda(device=device_id)
        aug_same_id_ct_data = aug_same_id_ct_data.cuda(device=device_id)
        aug_diff_id_ct_data = aug_diff_id_ct_data.cuda(device=device_id)
        label = label.type(torch.LongTensor)
        label = label.cuda(device=device_id)
        cls_label = cls_label.cuda(device=device_id)
        edge_label = edge_label.type(torch.LongTensor)
        edge_label = edge_label.cuda(device=device_id)
        seg, cls, recover, seg_edge, x_feature, x_aug_feature, x_same_patient_feature, x_othert_class_feature = net(
            ct_data, aug_ct_data, aug_same_id_ct_data, aug_diff_id_ct_data)

        seg_predict = seg.argmax(dim=1)
        cls_predict = cls.argmax(dim=1)

        seg_loss = criterion[0](seg, label)
        cls_loss = criterion[1](cls, cls_label)
        loss = criterion_temp(seg_loss, cls_loss)

        loss_list.append(loss.item())
        seg_predict = seg_predict.cpu().detach().numpy()
        cls_predict = cls_predict.cpu().detach().numpy()
        label = label.cpu().detach().numpy()
        cls_label = cls_label.cpu().detach().numpy()
        #predict[predict == 1] = 0
        #label[label == 1] = 0
        seg_predict[seg_predict >= 1] = 1
        cls_predict[cls_predict >= 1] = 1
        label[label >= 1] = 1
        cls_label[cls_label >= 1] = 1

        for i in range(len(case_id)):
            case_id_item = case_id[i]
            seg_predict_item = seg_predict[i]
            label_item = label[i]
            cls_predict_item = cls_predict[i]
            cls_label_item = cls_label[i]

            cls_predict_list.append(cls_predict_item)
            cls_label_list.append(cls_label_item)


            if anchor_case_id != -1 and anchor_case_id != case_id_item:
                predict_array = np.stack(predicts, axis=0)
                label_array = np.stack(labels, axis=0)
                dice = 2 * (predict_array * label_array).sum() / (predict_array.sum() + label_array.sum())
                recall = (predict_array[label_array == 1] == 1).sum() / (label_array == 1).sum()
                precision = (predict_array[label_array == 1] == 1).sum() / ((predict_array == 1).sum()+0.001)
                dice_list.append(dice)
                recall_list.append(recall)
                precision_list.append(precision)
                case_list.append(anchor_case_id)
                predicts.clear()
                labels.clear()
                # print(anchor_case_id, case_id_item, dice)
            anchor_case_id = case_id_item
            predicts.append(seg_predict_item)
            labels.append(label_item)

        tbar.set_postfix({"loss": loss.item()})
        tbar.update(1)

    predict_array = np.stack(predicts, axis=0)
    label_array = np.stack(labels, axis=0)
    dice = 2 * (predict_array * label_array).sum() / (predict_array.sum() + label_array.sum())
    recall = (predict_array[label_array == 1] == 1).sum() / (label_array == 1).sum()
    precision = (predict_array[label_array == 1] == 1).sum() / (predict_array == 1).sum()
    dice_list.append(dice)
    recall_list.append(recall)
    precision_list.append(precision)
    case_list.append(anchor_case_id)
    # print(anchor_case_id, case_id, dice)

    for index in range(len(case_list)):
        print("case_id:{}, dice:{}, recall:{}, precision:{}".format(
            case_list[index], round(dice_list[index],3), round(recall_list[index],3), round(precision_list[index], 3)))

    cls_predict_array = np.stack(cls_predict_list, axis=0)
    cls_label_array = np.stack(cls_label_list, axis=0)
    assert cls_predict_array.shape == cls_label_array.shape, str(cls_predict_array.shape) + "," + str(cls_label_array.shape)
    cls_acc = (cls_predict_array == cls_label_array).sum() / len(cls_label_array)
    cls_recall = (cls_label_array[cls_predict_array == 1] == 1).sum() / cls_label_array.sum()
    cls_precision = (cls_label_array[cls_predict_array == 1] == 1).sum() / cls_predict_array.sum()
    cls_f1 = (2 * cls_recall * cls_precision) / (cls_precision + cls_recall)
    print("[Cls] Acc:{}, F1:{}, Recall:{}".format(round(cls_acc,3), round(cls_f1,3), round(cls_recall,3)))


    seg_dice = np.mean(np.array(dice_list))
    seg_recall = np.mean(np.array(recall_list))
    seg_precision = np.mean(np.array(precision_list))

    loss = np.mean(np.array(loss_list))
    return (loss, seg_dice, seg_recall, seg_precision, cls_acc, cls_f1, cls_recall)



def main_shell(batch_size=1, num_gpu=1, lr=0.001, max_epoll=100):
    torch.cuda.empty_cache()

    net = InterpreStarJoinNet(input_channel=1, seg_class=2, cls_class=2)
    net = net.cuda(device=device_id)

    #net = net.cuda()

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=3, verbose=True,
        threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08
    )
    early_stopping = EarlyStopping(patience=30, verbose=True)

    seg_criterion = JI_and_Focal_loss({'batch_dice': batch_size, 'smooth': 1e-5, 'do_bg': False, 'square': False}).cuda(device=device_id)
    cls_criterion = torch.nn.CrossEntropyLoss().cuda(device=device_id)
    egde_criterion = torch.nn.CrossEntropyLoss().cuda(device=device_id)
    recover_criterion = torch.nn.L1Loss().cuda(device=device_id)
    sim_criterion = torch.nn.CosineSimilarity(dim=1).cuda(device=device_id)



    total_criterion = AutomaticWeightedLoss(3).cuda(device=device_id)
    criterion = [seg_criterion, cls_criterion, egde_criterion, recover_criterion, sim_criterion, total_criterion]


    transform = MedicalTransformCompose(output_size=(512, 512), roi_error_range=15, use_roi=False)
    train_data = DemoDataLoader(type="train", transform=transform)
    train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    valid_train_data = DemoDataLoader(type="train")
    valid_train_data_loader = DataLoader(valid_train_data, batch_size=batch_size, shuffle=False)
    valid_data = DemoDataLoader(type="valid")
    valid_data_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)

    best_dice = 0

    for epoll in range(max_epoll):
        epoch_str = f' Epoch {epoll + 1}/{max_epoll} '
        print(f'{epoch_str:-^40s}')
        print(f'Learning rate: {optimizer.param_groups[0]["lr"]}')

        net.train()
        transform.train()
        torch.set_grad_enabled(True)
        train(data_loader=train_data_loader, net=net, criterion=criterion, optimizer=optimizer)

        net.eval()
        transform.eval()
        torch.set_grad_enabled(False)
        (train_loss, train_seg_dice, train_seg_recall, train_seg_precision,
                     train_cls_acc, train_cls_f1, train_cls_recall) = evaluate(data_loader=valid_train_data_loader, net=net, criterion=criterion, type="train")
        (valid_loss, valid_seg_dice, valid_seg_recall, valid_seg_precision,
                     valid_cls_acc, valid_cls_f1, valid_cls_recall) = evaluate(data_loader=valid_data_loader, net=net, criterion=criterion, type="valid")
        scheduler.step(train_loss)
        early_stopping(valid_loss)

        print("Train loss : {}, Train Dice : {}, Train Acc:{}".format(round(train_loss, 6), round(train_seg_dice, 3), round(train_cls_acc, 3)))
        print("Valid loss : {}, Valid Dice : {}, Valid Acc:{}".format(round(valid_loss, 6), round(valid_seg_dice, 3), round(valid_cls_acc, 3)))

        with open("result/InterJoinNetwork/log.log", "a+") as file:
          file.writelines("Epoll:{}, T_Loss: {}, T_S_Dice: {}, T_S_Recall: {}, T_S_Pre: {}, "
                          "T_C_Acc: {}, T_C_F1:{}, T_C_Recall: {}, "
                          "V_Loss: {}, V_S_Dice: {}, V_S_Recall: {}, V_S_Pre:{},"
                          "V_C_Acc: {}, V_C_F1: {}, V_C_Recall:{} \n".format(
              epoll+1, round(train_loss, 6), round(train_seg_dice, 3), round(train_seg_recall, 3), round(train_seg_precision, 3),
                                             round(train_cls_acc, 3), round(train_cls_f1, 3), round(train_cls_recall, 3),
                       round(valid_loss, 6), round(valid_seg_dice, 3), round(valid_seg_recall, 3), round(valid_seg_precision, 3),
                                             round(valid_cls_acc, 3), round(valid_cls_f1, 3), round(valid_cls_recall, 3)
          ))

        if best_dice < valid_seg_dice + valid_cls_acc:
            best_dice = valid_seg_dice + valid_cls_acc
            torch.save(net.state_dict(), "result/InterJoinNetwork/epoll_{}.pkl".format(epoll+1))

        if early_stopping.early_stop:
            print("Early stopping")
            break



if __name__ == '__main__':
    main_shell(batch_size=2, num_gpu=2, lr=0.001)