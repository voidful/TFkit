import csv
import os


class Logger:

    def __init__(self, savedir, logfilename="message.log", metricfilename="metric.log", tensorboard=False):
        self.savedir = savedir
        self.logfilepath = os.path.join(savedir, logfilename)
        self.metricfilepath = os.path.join(savedir, metricfilename)
        self.tensorboard_writer = tensorboard.SummaryWriter() if tensorboard else None

    def write_log(self, *args):
        line = ' '.join([str(a) for a in args])
        with open(self.logfilepath, "a", encoding='utf8') as log_file:
            log_file.write(line + '\n')
        print(line)

    def write_metric(self, *args):
        if self.tensorboard_writer:
            self.tensorboard_writer.add_scalar(*args)
        else:
            with open(self.metricfilepath, "a", encoding='utf8') as log_file:
                writer = csv.writer(log_file)
                writer.writerow(args)
