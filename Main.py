import os
from timeit import default_timer as timer
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary

import torch.nn.functional as F

import Model_I3D
from Dataset import UCF101
from Utils import build_paths, print_time, set_seed

from torch.utils.tensorboard import SummaryWriter

import warnings


print_time('START TIME')

#### Paths #############################################################################################################

class_idxs, train_split, test_split, frames_root, pretrained_path = build_paths()

#### Params ############################################################################################################

print('\n==> Initializing Hyperparameters...\n')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0                                                           # best test accuracy
start_epoch = 0                                                        # start from epoch 0 or last checkpoint epoch
num_epochs = 50                                                        # Default = 200
initial_lr = .00001
batch_size = 4
num_workers = 2
num_classes = 101
seed = 0
clip_len = 64
model_summary = False
resume = False
pretrain = False
print_batch = False
nTest = 1

set_seed(seed=seed)

print('GPU Support:', 'Yes' if device != 'cpu' else 'No')
print('Starting Epoch:', start_epoch)
print('Total Epochs:', num_epochs)
print('Batch Size:', batch_size)
print('Clip Length: ', clip_len)
print('Initial Learning Rate: %g' % initial_lr)
print('Random Seed:', seed)

### Data ###############################################################################################################

print('\n==> Preparing Data...\n')

trainset = UCF101(class_idxs=class_idxs, split=train_split, frames_root=frames_root, clip_len=clip_len, train=True)
# print("trainset-buffer:", type(trainset[0][0]))
# print("trainset-label:", type(trainset[0][1]))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
# print("trainset-buffer AFTER:", type(trainset[0][0]))
# print("trainset-label BEFORE:", type(trainset[0][1]))
# exit()
print("Buffer shape:", trainset[0][0].shape)
print("Label shape:", trainset[0][1].shape)
exit()

testset = UCF101(class_idxs=class_idxs, split=test_split, frames_root=frames_root, clip_len=clip_len, train=False)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

print('Number of Classes: %d' % num_classes)
print('Number of Training Videos: %d' % len(trainset))
print('Number of Testing Videos: %d' % len(testset))


### Model ##############################################################################################################

print('\n==> Building Model...\n')

model = Model_I3D.InceptionI3d(num_classes=num_classes)
model = model.to(device)

if model_summary:
    summary(model, input_size=(3, clip_len, 224, 224))

writer = SummaryWriter()


### Optimizer, Loss, initial_lr Scheduler ##############################################################################

# train_params = [{'params': Model_I3D.get_1x_lr_params(model), 'initial_lr': initial_lr},
#                 {'params': Model_I3D.get_10x_lr_params(model), 'initial_lr': initial_lr * 10}]

# criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=initial_lr, momentum=0.9, weight_decay=5e-4)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
# criterion.to(device)

print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

num_steps_per_update = 4 # accum gradient
global steps
steps = 0

### Training ###########################################################################################################

def train(epoch):
    # print('\n==> Training model...\n')
    start = timer()
    # scheduler.step()
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    tot_loss = 0.0
    tot_loc_loss = 0.0
    tot_cls_loss = 0.0

    for batch_idx, (inputs, targets) in enumerate(trainloader):

        # print("inputs:", type(inputs))
        # print("targets:", type(targets))
        # exit()

        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        # print("Input shape: ", inputs.shape)

        per_frame_logits = model(inputs)
        # print("logits shape 1:", per_frame_logits.shape)
        # upsample to input size
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            per_frame_logits = F.upsample(per_frame_logits, clip_len, mode='linear')
        # print("logits shape 2:", per_frame_logits.shape)
        # print("targets shape:", targets.shape)

        per_frame_logits = torch.mean(per_frame_logits, dim=1)
        # print("logits shape 3:", per_frame_logits.shape)
        # exit()

        # compute localization loss
        loc_loss = F.binary_cross_entropy_with_logits(per_frame_logits, targets)
        tot_loc_loss += loc_loss.data[0]

        # compute classification loss (with max-pooling along time B x C x T)
        cls_loss = F.binary_cross_entropy_with_logits(torch.max(per_frame_logits, dim=2)[0],
                                                      torch.max(targets, dim=2)[0])
        tot_cls_loss += cls_loss.data[0]

        loss = (0.5 * loc_loss + 0.5 * cls_loss) / num_steps_per_update

        tot_loss += loss.data[0]
        loss.backward()

        writer.add_scalar("Batch loss/train", loss, epoch)


        if num_iter == num_steps_per_update:
            steps += 1
            num_iter = 0
            optimizer.step()
            optimizer.zero_grad()
            # lr_sched.step()
            if steps % 10 == 0:
                # print
                # '{} Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f}'.format(phase, tot_loc_loss / (
                #             10 * num_steps_per_update), tot_cls_loss / (10 * num_steps_per_update), tot_loss / 10)
                # # save model
                # torch.save(i3d.module.state_dict(), save_model + str(steps).zfill(6) + '.pt')
                tot_loss = tot_loc_loss = tot_cls_loss = 0.


        # loss = criterion(outputs, targets)
        # loss.backward()
        # optimizer.step()

        # train_loss += loss.item()
        # probs = nn.Softmax(dim=1)(outputs)
        # predicted = torch.max(probs, 1)[1]
        # total += targets.size(0)
        # correct += predicted.eq(targets).sum().item()

        # if print_batch:
        #     print('Epoch: %d | Batch: %d/%d | Running Loss: %.3f | Running Acc: %.2f%% (%d/%d) [Train]'
        #         % (epoch+1, batch_idx+1, len(trainloader), train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    end = timer()
    optim_dict = optimizer.state_dict()
    current_lr = optim_dict['param_groups'][0]['lr']

    writer.add_scalar("Loss/train", tot_loss, epoch)

    # print('Epoch %d | Loss: %.3f | Acc: %.2f%% | Current lr: %f | Time: %.2f min [Train]'
    #             % (epoch+1, train_loss/len(trainloader), 100.*correct/total, current_lr, (end - start)/60))


### Testing ############################################################################################################

def test(epoch):
    # print('\n==> Testing model...\n')
    start = timer()
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    tot_loss = 0.0
    tot_loc_loss = 0.0
    tot_cls_loss = 0.0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)

            per_frame_logits = model(inputs)
            # upsample to input size
            per_frame_logits = F.upsample(per_frame_logits, clip_len, mode='linear')

            # compute localization loss
            loc_loss = F.binary_cross_entropy_with_logits(per_frame_logits, targets)
            tot_loc_loss += loc_loss.data[0]

            # compute classification loss (with max-pooling along time B x C x T)
            cls_loss = F.binary_cross_entropy_with_logits(torch.max(per_frame_logits, dim=2)[0],
                                                          torch.max(targets, dim=2)[0])
            tot_cls_loss += cls_loss.data[0]


            loss = (0.5 * loc_loss + 0.5 * cls_loss) / num_steps_per_update

            tot_loss += loss.data[0]

            # outputs = model(inputs)
            # loss = criterion(outputs, targets)
            #
            # test_loss += loss.item()
            # _, predicted = outputs.max(dim=1)
            # total += targets.size(0)
            # correct += predicted.eq(targets).sum().item()
            #
            # if print_batch:
            #     print('Epoch: %d | Batch: %d/%d | Running Loss: %.3f | Running Acc: %.2f%% (%d/%d) [Test]'
            #         % (epoch+1, batch_idx+1, len(testloader), test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    end = timer()
    optim_dict = optimizer.state_dict()
    current_lr = optim_dict['param_groups'][0]['lr']

    # print('Epoch %d | Loss: %.3f | Acc: %.2f%% | Current lr: %f | Time: %.2f min [Test]'
    #       % (epoch+1, test_loss/len(testloader), 100.*correct/total, current_lr, (end - start)/60))

    # Save checkpoint
    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving Checkpoint..')
        state = {'model_state': model.state_dict(), 'optim_state': optimizer.state_dict(), 'acc': acc, 'epoch': epoch}
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc


    writer.add_scalar("Loss/test", tot_loss, epoch)
    writer.add_scalar("Accuracy/test", acc, epoch)



for epoch in range(start_epoch, start_epoch+num_epochs):
    if epoch == start_epoch:
        print('\n==> Training model...\n')

    train(epoch)

    if (epoch + 1) % nTest == 0:
        test(epoch)

writer.close()