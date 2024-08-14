import argparse
#import better_exceptions
from pathlib import Path
from collections import OrderedDict
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.optim.lr_scheduler import StepLR
import torch.utils.data
from torch.utils.data import DataLoader
import torch.nn.functional as Fun
from torch.utils.tensorboard import SummaryWriter
import torch.cuda as cuda
from GPUtil import showUtilization as gpu_usage
import pretrainedmodels
import pretrainedmodels.utils
from model import get_model
from dataset import FaceDataset
from defaults import _C as cfg
import ssl
from PIL import Image
import random
from BH import *
import cv2
ssl._create_default_https_context = ssl._create_unverified_context


def get_args():
    model_names = sorted(name for name in pretrainedmodels.__dict__
                         if not name.startswith("__")
                         and name.islower()
                         and callable(pretrainedmodels.__dict__[name]))
    parser = argparse.ArgumentParser(description=f"available models: {model_names}",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_dir", type=str, required=True, help="Data root directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint if any")
    parser.add_argument("--checkpoint", type=str, default="checkpoint", help="Checkpoint directory")
    parser.add_argument("--tensorboard", type=str, default=None, help="Tensorboard log directory")
    parser.add_argument('--multi_gpu', action="store_true", help="Use multi GPUs (data parallel)")
    parser.add_argument("opts", default=[], nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")
    args = parser.parse_args()
    return args


class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count


def train(train_loader, model, criterion, optimizer, epoch, device):
    model.train()
    loss_monitor = AverageMeter()
    accuracy_monitor = AverageMeter()

    with tqdm(train_loader) as _tqdm:
        for x, y in _tqdm:
            x = x.to(device)
            y = y.to(device)

            # compute output
            outputs = model(x)

            # calc loss
            loss = criterion(outputs, y)
            cur_loss = loss.item()

            # calc accuracy
            _, predicted = outputs.max(1)
            correct_num = predicted.eq(y).sum().item()

            # measure accuracy and record loss
            sample_num = x.size(0)
            loss_monitor.update(cur_loss, sample_num)
            accuracy_monitor.update(correct_num, sample_num)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _tqdm.set_postfix(OrderedDict(stage="train", epoch=epoch, loss=loss_monitor.avg),
                              acc=accuracy_monitor.avg, correct=correct_num, sample_num=sample_num)

    return loss_monitor.avg, accuracy_monitor.avg


def validate(validate_loader, model, criterion, epoch, device):
    model.eval()
    loss_monitor = AverageMeter()
    accuracy_monitor = AverageMeter()
    preds = []
    gt = []

    with torch.no_grad():
        with tqdm(validate_loader) as _tqdm:
            for i, (x, y, z) in enumerate(_tqdm):
                x = x.to(device)
                y = y.to(device)
                
                # compute output
                outputs = model(x)
                preds.append(Fun.softmax(outputs, dim=-1).cpu().numpy())
                gt.append(y.cpu().numpy())
                

                # valid for validation, not used for test
                if criterion is not None:
                    # calc loss
                    loss = criterion(outputs, y)
                    cur_loss = loss.item()

                    # calc accuracy
                    _, predicted = outputs.max(1)
                    correct_num = predicted.eq(y).sum().item()

                    # measure accuracy and record loss
                    sample_num = x.size(0)
                    loss_monitor.update(cur_loss, sample_num)
                    accuracy_monitor.update(correct_num, sample_num)
                    _tqdm.set_postfix(OrderedDict(stage="val", epoch=epoch, loss=loss_monitor.avg),
                                      acc=accuracy_monitor.avg, correct=correct_num, sample_num=sample_num)

    preds = np.concatenate(preds, axis=0)
    gt = np.concatenate(gt, axis=0)
    ages = np.arange(0, 101)
    ave_preds = (preds * ages).sum(axis=-1)
    
    diff = ave_preds - gt
    mae = np.abs(diff).mean()

    return loss_monitor.avg, accuracy_monitor.avg, mae


def free_gpu_cache():
    print("Initial GPU Usage")
    gpu_usage()                             

    torch.cuda.empty_cache()

    torch.cuda.set_device(0)

    print("GPU Usage after emptying the cache")
    gpu_usage()

def free_gpu_cache():
    print("Initial GPU Usage")
    gpu_usage()                             

    torch.cuda.empty_cache()

    torch.cuda.set_device(0)

    print("GPU Usage after emptying the cache")
    gpu_usage()




# ----------------------Part of BHE_Attack------------------------------------------


img_size = 224
img_transform = lambda i: i
threshold = 15   # 年龄差阈值

def pic_predict(model, img, path):
    path = path.split('/')[2]
    path = 'result/' + path
    img.save(path)
    
    img = cv2.imread(path)
    img = img_transform(img).astype(np.float32)
    img = torch.from_numpy(np.transpose(img, (2, 0, 1)))
    img = torch.tensor(img)
    img = img.unsqueeze(0)
    img = img.to(device)
    
    outputs = model(img)
    result = Fun.softmax(outputs, dim=-1).cpu().numpy()[0]
    pos = np.argmax(result)
    score = result[pos]
    
    return pos, score



def decide_function(model,path,age,xs):
    img = Image.open(path)
    attack_image = add_watermark_to_image(img, xs, watermark)
    attack_image = attack_image.convert('RGB')
    result, _ = pic_predict(model, attack_image, path)
    print("result: " + str(result) + ", age: " + str(age))
    if abs(result - age) > threshold:
        with open("run_log.txt",'a') as f:
            f.write(str(xs) + "  true age: " + str(age) + "  predict age: " + str(result) + '\n')
        return True
    else:
        return False

def object_function(model,path,age,xs):
    img = Image.open(path)
    attack_image = add_watermark_to_image(img, xs, watermark)
    attack_image = attack_image.convert('RGB')
    _, score = pic_predict(model, attack_image, path)
    
    print("score: " + str(score) + '\n')
    return score


# ------------------------------参数----------------------------------
def initpara():
    NP =50  # 种群数量
    F=0
    CR = 0.9  # 交叉概率
    generation = 3 # 遗传代数
    len_x = 5  # 将scale,angel作为参数进行测试，因此基因长度为5
    watermark_logo='ACMMM'
    watermark = Image.open("logo/ACMMM20.png")
    niter=3
    
    
    with open("run_log.txt",'a') as f:
        f.write("  alpha_max: " + str(255) + "  alpha_min: " + str(100) + "  NP: " + str(NP) + "  generation: " + str(generation) + "  CR: " + str(CR) + "  logo: " + str(watermark_logo) + "  threshold: " + str(threshold) + '\n')   # 记得修改这里
    
    return NP, F, CR, generation, len_x, watermark,watermark_logo,niter

def xy_sl_init(image_x, image_y):
    sl = 1.5 + random.random() * (4 - 1.5)
    watermark_x1, watermark_y1 = watermark.size
    watermark_scale = min(image_x / (sl * watermark_x1), image_y / (sl * watermark_y1)) 
    watermark_x1 = int(watermark_x1 * watermark_scale)
    watermark_y1 = int(watermark_y1 * watermark_scale)
    value_up_range = [255, image_x-watermark_x1, image_y-watermark_y1, 4,359]     # 第一位为透明度，最初为255
    value_down_range = [100, 0, 0, 1.5,0]     # 最初为100
    return sl, value_up_range, value_down_range


# 种群初始化
def initialtion(NP, image_x, image_y):
    np_list = []  # 种群，染色体
    
    for i in range(0, NP):
        x_list = []  # 个体，基因
        sl, value_up_range, value_down_range = xy_sl_init(image_x, image_y)
        for j in range(0, len_x-1):
            x_list.append(value_down_range[j] + random.random() * (value_up_range[j] - value_down_range[j]))
        x_list.append(sl)
        np_list.append(x_list)
    return np_list
# ------------------------------参数----------------------------------

## BH 变异
def mutation(model,path,img,label,watermark,np_list):
    v_list = []
    for i in range(0, NP):

        x_list, _= BH_Calculation(model,path,img, label, watermark, xs=np_list[i],niter=niter)
        # r1 = random.randint(0, NP - 1)
        # while r1 == i:
        #     r1 = random.randint(0, NP - 1)
        # r2 = random.randint(0, NP - 1)
        # while r2 == r1 | r2 == i:
        #     r2 = random.randint(0, NP - 1)
        # r3 = random.randint(0, NP - 1)
        # while r3 == r2 | r3 == r1 | r3 == i:
        #     r3 = random.randint(0, NP - 1)

        v_list.append(x_list)

    return v_list




#交叉
def crossover(np_list, v_list):
    u_list = []
    for i in range(0, NP):
        vv_list = []
        for j in range(0, len_x):
            if (random.random() <= CR) | (j == random.randint(0, len_x - 1)):
                vv_list.append(v_list[i][j])
            else:
                vv_list.append(np_list[i][j])
        u_list.append(vv_list)
    return u_list





# 选择
def selection(model,path,age,u_list, np_list):
    for i in range(0, NP):
        if object_function(model,path,age,u_list[i]) <= object_function(model,path,age,np_list[i]):
            np_list[i] = u_list[i]
        else:
            np_list[i] = np_list[i]
    return np_list



def BHE_Attack(model,path,label,np_list):

    min_x = []
    min_f = []
    xx = []
    for ii in range(0, NP):
        flag=False
        flag=decide_function(model,path,label,np_list[ii])
        if flag==True:
            image=Image.open(path)
            result_img=add_watermark_to_image(image, np_list[ii], watermark).convert('RGB')
            result_path = path.split('/')[2]
            result_path = 'result/' + result_path
            result_img.save(result_path)
            return True
        xx.append(object_function(model,path,label,np_list[ii]))


    min_f.append(min(xx))
    min_x.append(np_list[xx.index(min(xx))])
    
    
    for i in range(0, generation):
        print('第'+str(i+1)+'代')
        img=Image.open(path)
        v_list = mutation(model,path,img,label,watermark,np_list)
        u_list = crossover(np_list, v_list)
        np_list = selection(model,path,label,u_list, np_list)

        xx = []
        for j in range(0, NP):
            flag = False
            flag = decide_function(model,path,label,np_list[j])
            print(flag)
            if flag == True:
                with open("run_log.txt",'a') as f:
                    f.write("Success when iterating!" + '\n')
                image = Image.open(path)
                result_img=add_watermark_to_image(image, np_list[ii], watermark).convert('RGB')
                result_path = path.split('/')[2]
                result_path = 'result/' + result_path
                result_img.save(result_path)
                return  True
            xx.append(object_function(model,path,label,np_list[j]))
        # print('!!!!!!!')
        # print(xx)
        # print(min(xx))
        # print("%%%%%")
        min_f.append(min(xx))
        min_x.append(np_list[xx.index(min(xx))])
        image = Image.open(path)
        result_img = add_watermark_to_image(image, np_list[i], watermark).convert('RGB')
        result_path = path.split('/')[2]
        result_path = 'result/' + result_path
        result_img.save(result_path)
    return False



NP, F, CR, generation, len_x, watermark,watermark_logo,niter= initpara()

def attack(validate_loader, model, criterion, epoch, device):
    total = 0
    failed = 0
    success = 0
    noattack = 0
    
    model.eval()
    

    with torch.no_grad():
        for i, (x, y, z) in enumerate(validate_loader):
            for ii in range(0, 128):   # BATCH_SIZE is 128
                total = total + 1
                single = x[ii].unsqueeze(dim = 0)
                single = single.to(device)
                output = model(single)
                predict = np.argmax(Fun.softmax(output, dim=-1).cpu().numpy())
                age = y.cpu().numpy()[ii]
                img_path = z[ii]
                

                
                print("predicted age: " + str(predict))
                print("true age: " + str(age))
                
                img = Image.open(img_path)
                image_x, image_y = img.size
                
                if (abs(predict - age) > threshold) or (image_x > 500) or (image_y > 500):
                    noattack = noattack + 1
                    print("Don't satisfy init requirement! Failed! Skip this attack!")
                    print("total: " + str(total) + ", failed: " + str(failed) + ", success: " + str(success) + ", skip: " + str(noattack))
                    continue
                else:
                    np_list = initialtion(NP, image_x, image_y)   # 最后一个参数是是否存在约束条件
                    

                    result_flag = BHE_Attack(model,img_path, age, np_list)

                    if result_flag == True:
                        success = success + 1
                    else:
                        failed = failed + 1
                    
                    print("total: " + str(total) + ", failed: " + str(failed) + ", success: " + str(success) + ", skip: " + str(noattack))
                    with open("run_log.txt",'a') as f:
                        f.write(str(i * 128 + ii) + ":   total: " + str(total) + ", failed: " + str(failed) + ", success: " + str(success) + ", skip: " + str(noattack) + '\n\n')  
                        
                        
# ----------------------Part of BHE_Attack------------------------------------------

                
                    


def main():
    args = get_args()

    if args.opts:
        cfg.merge_from_list(args.opts)

    cfg.freeze()
    start_epoch = 0
    checkpoint_dir = Path(args.checkpoint)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # create model
    print("=> creating model '{}'".format(cfg.MODEL.ARCH))
    model = get_model(model_name=cfg.MODEL.ARCH)

    if cfg.TRAIN.OPT == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.TRAIN.LR,
                                    momentum=cfg.TRAIN.MOMENTUM,
                                    weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LR)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # optionally resume from a checkpoint
    resume_path = args.resume

    if resume_path:
        if Path(resume_path).is_file():
            print("=> loading checkpoint '{}'".format(resume_path))
            checkpoint = torch.load(resume_path, map_location="cpu")
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume_path, checkpoint['epoch']))
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            print("=> no checkpoint found at '{}'".format(resume_path))

    if args.multi_gpu:
        model = nn.DataParallel(model)

    if device == "cuda":
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss().to(device)
    train_dataset = FaceDataset(args.data_dir, "train", img_size=cfg.MODEL.IMG_SIZE, augment=True,
                                age_stddev=cfg.TRAIN.AGE_STDDEV)
    train_loader = DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True,
                              num_workers=cfg.TRAIN.WORKERS, drop_last=True)

    val_dataset = FaceDataset(args.data_dir, "valid", img_size=cfg.MODEL.IMG_SIZE, augment=False)
    val_loader = DataLoader(val_dataset, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False,
                            num_workers=cfg.TRAIN.WORKERS, drop_last=False)

    scheduler = StepLR(optimizer, step_size=cfg.TRAIN.LR_DECAY_STEP, gamma=cfg.TRAIN.LR_DECAY_RATE,
                       last_epoch=start_epoch - 1)
    best_val_mae = 10000.0
    train_writer = None

    if args.tensorboard is not None:
        opts_prefix = "_".join(args.opts)
        train_writer = SummaryWriter(log_dir=args.tensorboard + "/" + opts_prefix + "_train")
        val_writer = SummaryWriter(log_dir=args.tensorboard + "/" + opts_prefix + "_val")

    for epoch in range(start_epoch, cfg.TRAIN.EPOCHS):
        # train
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, device)

        # validate
        val_loss, val_acc, val_mae = validate(val_loader, model, criterion, epoch, device)

        if args.tensorboard is not None:
            train_writer.add_scalar("loss", train_loss, epoch)
            train_writer.add_scalar("acc", train_acc, epoch)
            val_writer.add_scalar("loss", val_loss, epoch)
            val_writer.add_scalar("acc", val_acc, epoch)
            val_writer.add_scalar("mae", val_mae, epoch)

        # checkpoint
        if val_mae < best_val_mae:
            print(f"=> [epoch {epoch:03d}] best val mae was improved from {best_val_mae:.3f} to {val_mae:.3f}")
            model_state_dict = model.module.state_dict() if args.multi_gpu else model.state_dict()
            torch.save(
                {
                    'epoch': epoch + 1,
                    'arch': cfg.MODEL.ARCH,
                    'state_dict': model_state_dict,
                    'optimizer_state_dict': optimizer.state_dict()
                },
                str(checkpoint_dir.joinpath("epoch{:03d}_{:.5f}_{:.4f}.pth".format(epoch, val_loss, val_mae)))
            )
            best_val_mae = val_mae
        else:
            print(f"=> [epoch {epoch:03d}] best val mae was not improved from {best_val_mae:.3f} ({val_mae:.3f})")

        # adjust learning rate
        scheduler.step()

    print("=> training finished")
    print(f"additional opts: {args.opts}")
    print(f"best val mae: {best_val_mae:.3f}")


if __name__ == '__main__':
    main()
    # 调用代码：python test.py --data_dir appa-real-release --resume checkpoint/epoch047_0.02328_3.8090.pth
