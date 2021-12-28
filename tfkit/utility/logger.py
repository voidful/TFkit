import csv
import os
import json


class Logger:

    def __init__(self, savedir, logfilename="message.log", metricfilename="metric.log", tensorboard=False, wandb=False,
                 print_fn=print):
        self.savedir = savedir
        self.logfilepath = os.path.join(savedir, logfilename)
        self.metricfilepath = os.path.join(savedir, metricfilename)
        self.tensorboard_writer = None
        self.wandb_writer = None
        self.print_fn = print_fn
        if tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.tensorboard_writer = SummaryWriter()
        if wandb:
            import wandb
            project_name = savedir.replace("/", "_")
            self.wandb_writer = wandb.init(project=project_name)

    def write_config(self, config_dict):
        if self.wandb_writer:
            self.wandb_writer.config.update(config_dict)
        if self.tensorboard_writer:
            self.tensorboard_writer.add_hparams(config_dict)

        with open(self.metricfilepath, "a", encoding='utf8') as log_file:
            writer = csv.writer(log_file)
            writer.writerow([json.dumps(config_dict)])

    def write_log(self, *args):
        line = ' '.join([str(a) for a in args])
        with open(self.logfilepath, "a", encoding='utf8') as log_file:
            log_file.write(line + '\n')
        self.print_fn(line)

    def write_metric(self, tag, scalar_value, global_step):
        if self.wandb_writer:
            self.wandb_writer.log({tag: scalar_value, "global_step": global_step})
        if self.tensorboard_writer:
            self.tensorboard_writer.add_scalar(tag, scalar_value, global_step)
        with open(self.metricfilepath, "a", encoding='utf8') as log_file:
            writer = csv.writer(log_file)
            writer.writerow([tag, scalar_value, global_step])
