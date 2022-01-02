import os

from logger import Logger

from models.model_factory import ModelFactory
from datasets.ds_factory import DatasetFactory

from loss_function import set_loss_function
from optimizer_scheduler import set_optimizer_scheduler
from utils import save_checkpoint, handle_device, next_expr_name
from parse_args import parse_arguments
from train_validate import train_epoch, validate
from constants import TRAIN, VAL

def train(params: dict, log: Logger):
    # specify dataset
    data = DatasetFactory.create(params)

    # specify model
    model = ModelFactory.create(params, len(data.vocab))

    # define loss function (criterion)
    criterion = set_loss_function(params)

    # optimizer & scheduler & load from checkpoint
    optimizer, scheduler, start_epoch, best_metric = set_optimizer_scheduler(params, model, log)

    bleu_n = params['bleu_n']
    # log details
    log_string = "\n" + "==== NET MODEL:\n" + str(model)
    log_string += "\n" + "==== OPTIMIZER:\n" + str(optimizer) + "\n"
    log_string += "\n" + "==== SCHEDULER:\n" + str(scheduler) + "\n"
    log_string += "\n" + "==== DATASET (TRAIN):\n" + repr(data.dataset['train']) + "\n" 
    log_string += "\n" + "==== DATASET (VAL):\n" + repr(data.dataset['val']) + "\n"
    log_string += "\n" + "==== METRIC: BLEU-"+str(bleu_n)+"\n"
    log.log_global(log_string)
   
    # train
    device = params['device']
    for epoch in range(start_epoch, params['TRAIN']['epochs']):

        # train for one epoch
        loss_train = train_epoch(data.loader[TRAIN], model, criterion, optimizer, scheduler, epoch,
                           device, log)

        # ??? evaluate on train set

        # evaluate on validation set
        metric_val = validate(
            val_loader=data.loader[VAL],
            model=model, 
            vocab=data.vocab,
            bleu_n=bleu_n,
            device=device
        )

        # remember best metric
        is_best = metric_val > best_metric
        best_metric = max(metric_val, best_metric)

        # save checkpoint
        if params['LOG']['do_checkpoint']:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_metric': best_metric,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler
            }, model, params, is_best)

        # logging results
        time_string = log.timers['global'].current2str()  # get current time
        log.log_epoch(epoch + 1,
                      loss_train,
                      metric_val,
                      is_best, time_string)


if __name__ == "__main__":

    # parse argument string
    params = parse_arguments()

    # additional configuration
    if len(params['experiment_name']) == 0:  # experiment name
        params['experiment_name'] = next_expr_name(params['path_save'], "e", 4)
    params['device'] = handle_device(params['with_cuda'])  # manage gpu/cpu devices
    if params['new_folder']:
        params['path_save'] = os.path.join(params['path_save'], params['experiment_name'])
        os.mkdir(params['path_save'])

    # logging & timer
    logger = Logger(params)

    # train
    train(params, logger)

    # close all log files
    logger.close()
