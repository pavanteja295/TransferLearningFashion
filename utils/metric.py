import time
import torch

def accuracy(output, target, topk=(1,), avg_meters=None):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        # import pdb; pdb.set_trace()
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))


        res = []
        for k in topk:
            correct_elem = correct[:k].sum(dim=0) # this should sum to one or zero
            correct_k = correct[:k].view(-1).float().sum().item()
            res.append(correct_k*100.0 / batch_size)
        
        
        if avg_meters is not None:
            sums_ = torch.zeros(len(avg_meters))
            cnts_ = torch.zeros(len(avg_meters))

            for i, tar in enumerate(target):
                sums_[tar.item()] += correct_elem[i].item()
                cnts_[tar.item()] += 1

            for i, cnt in enumerate(cnts_):
                if cnt.item():
                    # update only the non zero the average meter
                    avg_meters[i].update(sums_[i].item()/cnts_[i].item(), cnts_[i].item())
                
            if len(res) == 1:
                return res[0], avg_meters
            
        if len(res)==1:
            return res[0]
        else:
            return res


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = float(self.sum) / self.count


class Timer(object):
    """
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.interval = 0
        self.time = time.time()

    def value(self):
        return time.time() - self.time

    def tic(self):
        self.time = time.time()

    def toc(self):
        self.interval = time.time() - self.time
        self.time = time.time()
        return self.interval