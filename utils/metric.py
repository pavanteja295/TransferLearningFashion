import time
import torch
from textwrap import wrap
import re
import itertools
# import tfplot
import matplotlib
import numpy as np
from sklearn.metrics import confusion_matrix

# def plot_confusion_matrix(correct_labels, predict_labels, labels, title='Confusion matrix', tensor_name = 'MyFigure/image', normalize=False):
#     ''' 
#     Parameters:
#         correct_labels                  : These are your true classification categories.
#         predict_labels                  : These are you predicted classification categories
#         labels                          : This is a lit of labels which will be used to display the axix labels
#         title='Confusion matrix'        : Title for your matrix
#         tensor_name = 'MyFigure/image'  : Name for the output summay tensor

#     Returns:
#         summary: TensorFlow summary 

#     Other itema to note:
#         - Depending on the number of category and the data , you may have to modify the figzie, font sizes etc. 
#         - Currently, some of the ticks dont line up due to rotations.
#     '''
#     cm = confusion_matrix(correct_labels, predict_labels, labels=labels)
#     if normalize:
#         cm = cm.astype('float')*10 / cm.sum(axis=1)[:, np.newaxis]
#         cm = np.nan_to_num(cm, copy=True)
#         cm = cm.astype('int')

#     np.set_printoptions(precision=2)
#     ###fig, ax = matplotlib.figure.Figure()

#     fig = matplotlib.figure.Figure(figsize=(7, 7), dpi=320, facecolor='w', edgecolor='k')
#     ax = fig.add_subplot(1, 1, 1)
#     im = ax.imshow(cm, cmap='Oranges')

#     classes = [re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x) for x in labels]
#     classes = ['\n'.join(wrap(l, 40)) for l in classes]

#     tick_marks = np.arange(len(classes))

#     ax.set_xlabel('Predicted', fontsize=7)
#     ax.set_xticks(tick_marks)
#     c = ax.set_xticklabels(classes, fontsize=4, rotation=-90,  ha='center')
#     ax.xaxis.set_label_position('bottom')
#     ax.xaxis.tick_bottom()

#     ax.set_ylabel('True Label', fontsize=7)
#     ax.set_yticks(tick_marks)
#     ax.set_yticklabels(classes, fontsize=4, va ='center')
#     ax.yaxis.set_label_position('left')
#     ax.yaxis.tick_left()

#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         ax.text(j, i, format(cm[i, j], 'd') if cm[i,j]!=0 else '.', horizontalalignment="center", fontsize=6, verticalalignment='center', color= "black")
#     fig.set_tight_layout(True)
#     summary = tfplot.figure.to_summary(fig, tag=tensor_name)
#     return summary


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