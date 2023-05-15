from __future__ import print_function
import cv2
import math
from patchify import patchify
from sklearn.model_selection import train_test_split
from visdom import Visdom
import random
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix
import model
from Utils import *
from option import opt


if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)
cudnn.benchmark = False

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

dataset = opt.datasetname
X, y = loadData(dataset)
H = X.shape[0]
W = X.shape[1]
pca_components = opt.spectrumnum
print('Hyperspectral data shape:', X.shape)
print('Label shape:', y.shape)
sample_number = np.count_nonzero(y)
print('the number of sample:', sample_number)
X_pca = applyPCA(X, numComponents=pca_components)
print('Data shape after PCA :', X_pca.shape)
[nRow, nColumn, nBand] = X_pca.shape
num_class = int(np.max(y))
windowsize = opt.windowsize
Wid = opt.inputsize
halfsizeTL = int((Wid-1)/2)
halfsizeBR = int((Wid-1)/2)
paddedDatax = cv2.copyMakeBorder(X_pca, halfsizeTL, halfsizeBR, halfsizeTL, halfsizeBR, cv2.BORDER_CONSTANT, 0)  #cv2.BORDER_REPLICAT周围值
paddedDatay = cv2.copyMakeBorder(y, halfsizeTL, halfsizeBR, halfsizeTL, halfsizeBR, cv2.BORDER_CONSTANT, 0)
patchIndex = 0
X_patch = np.zeros((sample_number, Wid, Wid, pca_components))
y_patch = np.zeros(sample_number)
for h in range(0, paddedDatax.shape[0]):
    for w in range(0, paddedDatax.shape[1]):
        if paddedDatay[h, w] == 0:
            continue
        X_patch[patchIndex, :, :, :] = paddedDatax[h-halfsizeTL:h+halfsizeBR+1, w-halfsizeTL:w+halfsizeBR+1, :]
        X_patch[patchIndex] = paddedDatay[h, w]
        patchIndex = patchIndex + 1
X_train_p = patchify(paddedDatax, (Wid, Wid, pca_components), step=1)
if opt.input3D:
    X_train_p = X_train_p.reshape(-1, Wid, Wid, pca_components, 1)
else:
    X_train_p = X_train_p.reshape(-1, Wid, Wid, pca_components)
y_train_p = y.reshape(-1)
indices_0 = np.arange(y_train_p.size)
X_train_q = X_train_p[y_train_p > 0, :, :, :]
y_train_q = y_train_p[y_train_p > 0]
indices_1 = indices_0[y_train_p > 0]
y_train_q -= 1
X_train_q = X_train_q.transpose(0, 3, 1, 2)
Xtrain, Xtest, ytrain, ytest, idx1, idx2 = train_test_split(X_train_q, y_train_q, indices_1,
                                                            train_size=opt.numtrain, random_state=opt.random_seed,
                                                            stratify=y_train_q)
print('after Xtrain shape:', Xtrain.shape)
print('after Xtest shape:', Xtest.shape)

trainset = TrainDS(Xtrain, ytrain)
testset = TestDS(Xtest, ytest)
train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=opt.batchSize, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=opt.batchSize, shuffle=False, num_workers=0)
nz = int(opt.nz)
nc = pca_components
nb_label = num_class
print("label", nb_label)

def train(netD, train_loader, test_loader):
    viz = Visdom()
    viz.close()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for epoch in range(1, opt.epochs + 1):
        netD.train()
        right = 0
        for i, datas in enumerate(train_loader):
            netD.zero_grad()
            img, label = datas
            batch_size = img.size(0)
            input.resize_(img.size()).copy_(img)
            c_label.resize_(batch_size).copy_(label)
            c_output = netD(input)
            c_errD_real = c_criterion(c_output, c_label)
            errD_real = c_errD_real
            errD_real.backward()
            D_x = c_output.data.mean()
            correct, length = test(c_output, c_label)
            optimizerD.step()
            right += correct

        if epoch % 5 == 0:
            print('[%d/%d][%d/%d]   D(x): %.4f, errD_real: %.4f,  Accuracy: %.4f / %.4f = %.4f'
                  % (epoch, opt.epochs, i, len(train_loader),
                     D_x, errD_real,
                     right, len(train_loader.dataset), 100. * right / len(train_loader.dataset)))
        if epoch % 5 == 0:
            netD.eval()
            test_loss = 0
            right = 0
            all_Label = []
            all_target = []
            for data, target in test_loader:
                indx_target = target.clone()
                if opt.cuda:
                    data, target = data.cuda(), target.cuda()
                with torch.no_grad():
                    data, target = Variable(data), Variable(target)

                start.record(stream=torch.cuda.current_stream())
                output = netD(data)
                end.record(stream=torch.cuda.current_stream())
                end.synchronize()
                test_loss += c_criterion(output, target).item()
                pred = output.max(1)[1]  # get the index of the max log-probability
                all_Label.extend(pred)
                all_target.extend(target)
                right += pred.cpu().eq(indx_target).sum()

            test_loss = test_loss / len(test_loader)  # average over number of mini-batch
            acc = float(100. * float(right)) / float(len(test_loader.dataset))
            print('\tTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
                test_loss, right, len(test_loader.dataset), acc))

            AAA = torch.stack(all_target).data.cpu().numpy()
            BBB = torch.stack(all_Label).data.cpu().numpy()
            C = confusion_matrix(AAA, BBB)
            C = C[:num_class, :num_class]
            k = kappa(C, np.shape(C)[0])
            AA_ACC = np.diag(C) / np.sum(C, 1)
            AA = np.mean(AA_ACC, 0)

            if math.isnan(acc):
                acc = 0
            viz.line(
                X=np.array([epoch]),
                Y=np.array([acc]),
                win='window1',
                update='append',
                opts=dict(legend=["acc"],
                          showlegend=True,
                          markers=False,
                          title='precision',
                          xlabel='epoch',
                          ylabel='Volume',
                          fillarea=False),
            )
            viz.line(
                X=np.column_stack((epoch, epoch)),
                Y=np.column_stack((D_x.data.cpu().numpy(), (errD_real).data.cpu().numpy())),
                win='window2',
                update='append',
                opts=dict(legend=["D(X)", "errD_real"],
                          showlegend=True,
                          markers=False,
                          title='loss',
                          xlabel='epoch',
                          ylabel='Volume',
                          fillarea=False),
            )

            print('OA= %.5f AA= %.5f k= %.5f' % (acc, AA, k))

for index_iter in range(1):
    print('iter:', index_iter)
    netD = model.LSGAVIT(img_size=Wid,
                         patch_size=3,
                         in_chans=pca_components,
                         num_classes=num_class,
                         embed_dim=120,
                         depths=[2],
                         num_heads=[12, 12, 12, 24],
                         )
    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
    print(netD)
    c_criterion = nn.CrossEntropyLoss()
    input = torch.FloatTensor(opt.batchSize, nc, opt.inputsize, opt.inputsize)
    c_label = torch.LongTensor(opt.batchSize)
    if opt.cuda:
        netD.cuda()
        c_criterion.cuda()
        input = input.cuda()
        c_label = c_label.cuda()
    input = Variable(input)
    c_label = Variable(c_label)
    optimizerD = optim.Adam(netD.parameters(), lr=opt.D_lr)
    train(netD, train_loader, test_loader)