import torch
from utils import load_checkpoint


def set_optimizer_scheduler(params, encoder, decoder, log):
    
    model_params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())

    # optimizer
    if params['TRAIN']['optim'] == "sgd":
        optimizer = torch.optim.SGD(
            model_params,
            lr=params['TRAIN']['lr'],
            momentum=params['TRAIN']['momentum'],
            weight_decay=params['TRAIN']['weight_decay']
        )
    elif params['TRAIN']['optim'] == "adam":
        optimizer = torch.optim.Adam(
            model_params, 
            lr=params['TRAIN']['lr']
        )
    elif params['TRAIN']['optim'] == "rms_prop":
        optimizer = torch.optim.RMSprop(model_params)
    else:
        raise ValueError('The settings for optimize are not recognized.')

    # resume from a checkpoint
    start_epoch, best_prec = 0, 0
    # if len(params['TRAIN']['resume']) > 0:
        # start_epoch, best_prec = load_checkpoint(log, model,
        #                                          params['TRAIN']['resume'],
        #                                          optimizer)

        

    # scheduler (if any)
    if params['TRAIN']['scheduler'] == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=params['TRAIN']['lr_schedule_step'],
                                                    gamma=params['TRAIN']['lr_schedule_gamma'])
    else:
        scheduler = None

    return optimizer, scheduler, start_epoch, best_prec
