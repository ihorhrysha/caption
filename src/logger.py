import os
from typing import Dict

from torch.nn.modules.module import Module
import wandb
from wandb.sdk.wandb_run import Run

from thirdparty.timer import Timer


def dict2str(d, start_n=0):
    res = ""
    prefix_val = " " * start_n
    for k in d:
        if type(d[k]) is dict:
            res += prefix_val + str(k) + ": " + "\n" + dict2str(d[k], start_n + 2)
        else:
            res += prefix_val + str(k) + ": " + str(d[k]) + "\n"
    return res


class Logger(object):
    def __init__(self, params):

        self.with_tensorboard = params['LOG']['tensorboard']
        self.with_wandb = params['LOG']['wandb']['enable']

        self.experiment_name = params['experiment_name']
        self.iter_interval = params['LOG']['iter_interval']
        self.num_epochs = params['TRAIN']['epochs']

        self.path_log_files = os.path.join(params['path_save'], self.experiment_name)

        # timer
        self.timers = {'global': Timer()}
        self.timers['global'].tic()
        
        self.params = params

    def __enter__(self):
        
        # log files names
        filename_log_epoch = self.path_log_files + "_log_epoch.txt"
        filename_log_iter = self.path_log_files + "_log_iter.txt"
        filename_global = self.path_log_files + "_log.txt"

        # create and open log files
        self.f_log_iter = open(filename_log_iter, "w+")
        self.f_log_epoch = open(filename_log_epoch, "w+")
        self.f_log_global = open(filename_global, "w+")

        # init all files
        self.f_log_iter.write("{:>6} {:>14} {:>14}\n".format('iter', 'loss', 'elapsed_time'))
        self.f_log_iter.flush()
        self.f_log_epoch.write("{:>6} {:>14} {:>14} {:>14}\n".
                               format('epoch',
                                      'avg_loss_train', 'avg_metric_val',
                                      'elapsed_time'))
        self.f_log_epoch.flush()

        if self.with_wandb:
            self.wandb_run:Run = wandb.init(
                project=self.params['project_name'], 
                entity=self.params['LOG']['wandb']['entity'], 
                config=self.params
            )


        # make TensorBoard logging
        if self.with_tensorboard:
            from torch.utils.tensorboard import SummaryWriter

            # set-up writer
            self.writer_tb = SummaryWriter(self.params['path_save'])

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # close log files
        if self.with_wandb:
            self.wandb_run.finish(exit_code=int(exc_val is not None))
        self.f_log_iter.close()
        self.f_log_epoch.close()
        self.f_log_global.close()

    def watch(self, model:Module, criterion=None) -> None:
        if self.with_wandb:
            self.wandb_run.watch(models=model, criterion=criterion)

    def log_iter(self, iter_current, epoch_current, num_iter, loss, time_str):

        if iter_current % self.iter_interval == 0:

            # log details
            log_string = "[{}] Epoch[{:^5}/{:^5}] Iteration[{:^5}/{:^5}] Loss: {:.4f} Time: {}" \
                         "".format(self.experiment_name, epoch_current, self.num_epochs,
                                   iter_current, num_iter, loss, time_str)
            self.log_global(log_string)

            globaliter = iter_current + epoch_current * num_iter

            self.f_log_iter.write(
                "{:6d} {:14.4f} {:>14}\n".format(globaliter, loss, time_str))
            self.f_log_iter.flush()

            # tb log
            if self.with_tensorboard:
                self.writer_tb.add_scalar('Train/RunningLoss', loss, globaliter)

            if self.with_wandb:
                self.wandb_run.log({
                    "loss_train_iter": loss, 
                })

    def log_epoch(self,
                  n_epoch,
                  loss_train, metric_val, example_table: Dict, is_best, time_str):

        # log details
        log_string = ("Epoch [{:^5}]: Train Avg loss: {:.4f} \n".format(n_epoch, loss_train) +
                      "{:14} Val Avg Metric: {:.4f};\n".format(" ", metric_val) +
                      "{:14} Time: {}".format(" ", time_str) +
                      ("\n{:14} BEST MODEL SAVED".format(" ") if is_best else ""))
        self.log_global(log_string)

        self.f_log_epoch.write("{:6d} {:14.4f} {:14.4f} {:>14}\n".
                               format(n_epoch, loss_train, metric_val, time_str))
        self.f_log_epoch.flush()

        if self.with_tensorboard:
            self.writer_tb.add_scalar('Train/Loss', loss_train, n_epoch)
            self.writer_tb.add_scalar('Val/Metric', metric_val, n_epoch)

        if self.with_wandb:
            if is_best:
                self.wandb_run.summary["best_metric"] = metric_val

            wandb_table = wandb.Table(
                columns=list(example_table.keys()), 
                data=[list(row) for row in zip(*list(example_table.values()))]
            )
            self.wandb_run.log({
                "loss_train": loss_train, 
                "metric_val": metric_val,
                "example_table": wandb_table,
                "epoch": n_epoch,
                "is_best": is_best
            })

    def log_global(self, log_str):
        self.f_log_global.write(log_str + "\n")
        self.f_log_global.flush()
        print(log_str)