import torch

from torch.nn import Module
from torch.nn.utils.rnn import pack_padded_sequence
from thirdparty.meters import AverageMeter, accuracy


def train_epoch(
    train_loader, 
    encoder: Module, 
    decoder: Module, 
    criterion, 
    optimizer, 
    scheduler, 
    epoch, 
    device, 
    log):

    losses = AverageMeter()
    top1 = AverageMeter()
    
    # switch to train mode ???
    encoder.train()
    decoder.train()

    # number of training samples
    num_iter = len(train_loader)

    for i, (images, captions, lengths) in enumerate(train_loader):
        # Set mini-batch dataset
        images = images.to(device)
        captions = captions.to(device)
        targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

        # compute output
        features = encoder(images)
        outputs = decoder(features, captions, lengths)
        loss = criterion(outputs, targets)

        # measure accuracy ??? and record loss
        losses.update(loss, images.size(0))
        
        # prec1 = accuracy(output, target)
        # top1.update(prec1[0], images.size(0))

        # compute gradient and do optimizer step
        # optimizer.zero_grad() # ???
        decoder.zero_grad()
        encoder.zero_grad()
        loss.backward()
        optimizer.step()

        # logging
        time_string = log.timers['global'].current2str()  # get current time
        log.log_iter(i, epoch, num_iter, losses.val.to('cpu').item(), time_string)

    # adjust_learning_rate
    if scheduler is not None:
        scheduler.step()

    return top1.avg.to('cpu').item(), losses.avg.to('cpu').item()


def validate(val_loader, encoder, decoder, criterion, device):
    top1 = AverageMeter()
    losses = AverageMeter()

    # switch to evaluate mode
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        for i, (images, captions, lengths) in enumerate(val_loader):
            # Set mini-batch dataset
            images = images.to(device)
            captions = captions.to(device)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

            # compute output
            features = encoder(images)
            outputs = decoder(features, captions, lengths)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            losses.update(loss, images.size(0))

            # prec1 = accuracy(output, target)
            # top1.update(prec1[0], input_data.size(0))

    return top1.avg.to('cpu').item(), losses.avg.to('cpu').item()
