import torch

from torch.nn import Module
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data.dataloader import DataLoader
from logger import Logger
from datasets.vocab import Vocabulary
from thirdparty.meters import AverageMeter

from nltk.translate.bleu_score import corpus_bleu

def train_epoch(
    train_loader, 
    model: Module, 
    criterion, 
    optimizer, 
    scheduler, 
    epoch, 
    device, 
    log:Logger):

    losses = AverageMeter()
    
    # switch to train mode
    model.train()

    # number of training samples
    num_iter = len(train_loader)

    for i, (images, captions, lengths) in enumerate(train_loader):
        # Set mini-batch dataset
        images = images.to(device)
        captions = captions.to(device)
        targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

        # compute output
        outputs = model(images, captions, lengths)
        loss = criterion(outputs, targets)

        # record loss
        losses.update(loss, images.size(0))

        # Zero the gradients.
        optimizer.zero_grad()
        # Feed forward
        loss.backward()
        optimizer.step()

        # logging
        time_string = log.timers['global'].current2str()  # get current time
        log.log_iter(i+1, epoch, num_iter, losses.val.to('cpu').item(), time_string)

    # adjust_learning_rate
    if scheduler is not None:
        scheduler.step()

    return losses.avg.to('cpu').item()


def validate(val_loader: DataLoader, model:Module, vocab:Vocabulary, device, bleu_n:int = 4, ):

    list_of_references=[]
    hypotheses = []

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for images, captions in val_loader:
            # Set mini-batch dataset
            features = model.encoder(images) # [n, l, w, h]
            for i in range(len(captions)):
                feature = features[i].unsqueeze(0).to(device)

                # compute output
                sampled_ids = model.decoder.sample(feature)
                word_ids = sampled_ids.squeeze(0).cpu().tolist()

                hypothesis = []
                for word_id in word_ids:
                    word = vocab(word_id, return_index=False)
                    
                    if word == vocab.START:
                        continue
                    if word == vocab.END:
                        break

                    hypothesis.append(word)

                list_of_references.append(captions[i])
                hypotheses.append(hypothesis)

    bleu_weights = [1/bleu_n]*bleu_n
    return corpus_bleu(list_of_references=list_of_references, hypotheses=hypotheses, weights=bleu_weights)
