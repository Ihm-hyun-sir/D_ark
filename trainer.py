from utils import MetricLogger, ProgressLogger
import time
import torch
from tqdm import tqdm
# import wandb

def train_one_epoch(model, use_head_n, dataset, data_loader_train, device, criterion, optimizer, epoch, teacher):
    batch_time = MetricLogger('Time', ':6.3f')
    losses_cls = MetricLogger('Loss_'+dataset+' cls', ':.4e')
    losses_mse = MetricLogger('Loss_'+dataset+' mse', ':.4e')
    progress = ProgressLogger(
        len(data_loader_train),
        [batch_time, losses_cls, losses_mse],
        prefix="Epoch: [{}]".format(epoch))

    model.train()
    MSE = torch.nn.MSELoss()
    # coefficient scheduler from  0 to 0.5 
    #coff = coef_schedule[it]
    end = time.time()
    for i, (samples1, samples2, targets) in enumerate(data_loader_train):
        samples1, samples2, targets = samples1.float().to(device), samples2.float().to(device), targets.float().to(device)
        
        feat_t, pred_t = teacher(samples2, use_head_n)
        feat_s, pred_s = model(samples1, use_head_n)
        loss_cls = criterion(pred_s, targets)
        loss_const = MSE(feat_s, feat_t)
    
        
        #loss = coff[it] * loss_cls + coff[it] * loss_const
        loss = 0.8 * loss_cls + 0.2 * loss_const

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses_cls.update(loss_cls.item(), samples1.size(0))
        losses_mse.update(loss_const.item(), samples1.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 50 == 0:
            progress.display(i)
        
        ema_update_teacher(model, teacher)
       
        
    """
    ema_update_teacher(model, teacher, momentum_schedule, it)
    """
   

    # wandb.log({"train_loss_cls_{}".format(dataset): losses_cls.avg})
    # wandb.log({"train_loss_mse_{}".format(dataset): losses_mse.avg})


def ema_update_teacher(model, teacher):
    with torch.no_grad():
        #m = momentum_schedule[it]  # momentum parameter
        m = 0.95 
        for param_q, param_k in zip(model.parameters(), teacher.parameters()):
            param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)


def evaluate(model, use_head_n, data_loader_val, device, criterion, dataset):
    model.eval()

    with torch.no_grad():
        batch_time = MetricLogger('Time', ':6.3f')
        losses = MetricLogger('Loss', ':.4e')
        progress = ProgressLogger(
        len(data_loader_val),
        [batch_time, losses], prefix='Val_'+dataset+': ')

        end = time.time()
        for i, (samples, _, targets) in enumerate(data_loader_val):
            samples, targets = samples.float().to(device), targets.float().to(device)

            _, outputs = model(samples, use_head_n)
            loss = criterion(outputs, targets)

            losses.update(loss.item(), samples.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 50 == 0:
                progress.display(i)

    return losses.avg


def test_classification(model, use_head_n, data_loader_test, device, multiclass = False): 
    model.eval()

    y_test = torch.FloatTensor().to(device)
    p_test = torch.FloatTensor().to(device)

    with torch.no_grad():
        for i, (samples, _, targets) in enumerate(tqdm(data_loader_test)):
            targets = targets.cuda()
            y_test = torch.cat((y_test, targets), 0)
            samples = samples.to(device)
            """
            if len(samples.size()) == 4:
                bs, c, h, w = samples.size()
                n_crops = 1
            elif len(samples.size()) == 5:
                bs, n_crops, c, h, w = samples.size()
            """

            #varInput = torch.autograd.Variable(samples.view(-1, c, h, w).to(device))

            _, out = model(samples, use_head_n)
            if multiclass:
                out = torch.softmax(out,dim = 1)
            else:
                out = torch.sigmoid(out)
            #outMean = out.view(bs, n_crops, -1).mean(1)
            p_test = torch.cat((p_test,out),0)

    return y_test, p_test
    
