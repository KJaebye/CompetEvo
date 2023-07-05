# ------------------------------------------------------------------------------------------------------------------- #
#   @description: Class Logger
#   @author: Kangyao Huang
#   @created date: 26.Oct.2022
# ------------------------------------------------------------------------------------------------------------------- #

import logging
import sys
import os
from termcolor import colored
import platform
import socket
from datetime import datetime

NOTSET = 0
LOGGING_METHOD = ['info', 'warning', 'error', 'critical',
                  'warn', 'exception', 'debug']


class MyFormatter(logging.Formatter):
    """ A class to make preference format. """

    def format(self, record):
        date = colored('[%(asctime)s @%(filename)s:%(lineno)d]', 'green')
        msg = '%(message)s'

        if record.levelno == logging.WARNING:
            fmt = date + ' ' + colored('WRN', 'yellow', attrs=[]) + ' ' + msg
        elif record.levelno == logging.ERROR:
            fmt = date + ' ' + colored('ERR', 'red', attrs=['underline']) + ' ' + msg
        elif record.levelno == logging.CRITICAL:
            fmt = date + ' ' + colored('CRITICAL', 'blue', attrs=['underline']) + ' ' + msg
        else:
            fmt = date + ' ' + msg

        if hasattr(self, '_style'):
            # Python3 compatibilty
            self._style._fmt = fmt
        self._fmt = fmt

        return super(self.__class__, self).format(record)


class Logger(logging.Logger):
    """ Logger to record everything about training. """

    def __init__(self, name, args=None, cfg=None, level=NOTSET):
        super(Logger, self).__init__(name, level)
        self.file_path = None
        self.args = args
        self.cfg = cfg
        # log output dir
        self.output_dir = './tmp'
        self.sub_dir = '/%s' % (cfg.env_name)
        self.time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.target_dir = '/' + self.time_str
        self.run_dir = self.output_dir + self.sub_dir + self.target_dir
        self.model_dir = '%s/models' % self.run_dir
        self.log_dir = '%s/log' % self.run_dir
        self.tb_dir = '%s/tb' % self.run_dir

        # add paths to cfg
        setattr(cfg, 'model_dir', self.model_dir)
        setattr(cfg, 'log_dir', self.log_dir)
        setattr(cfg, 'tb_dir', self.tb_dir)
        setattr(cfg, 'run_dir', self.run_dir)

        # log output file
        self.prefix = cfg.env_name + '-'
        self.file_name = self.prefix + self.time_str + '.log'
        # training related attributes
        self.num_steps = 0
        self.episode_len = 0
        self.episode_reward = 0
        self.episode_c_reward = 0  # custom reward

    def print_system_info(self):
        # print necessary info
        self.info('Hardware info: {}'.format(platform.machine()))
        self.info('Device info: {}'.format(socket.gethostname()))
        self.info('Platform info: {}'.format(platform.platform()))
        self.info('System info: {}'.format(platform.system()))
        self.info('Current Python version: {}'.format(platform.python_version()))
        return

    def set_output_handler(self):
        """ Set the console output handler. """
        con_handler = logging.StreamHandler(sys.stdout)
        con_handler.setFormatter(MyFormatter(datefmt='%Y%m%d %H:%M:%S'))
        self.addHandler(con_handler)
        return

    def set_file_handler(self):
        """ Create and save log file. Set the log file handler. """
        # create directories
        os.makedirs(self.model_dir, exist_ok=False)
        os.makedirs(self.log_dir, exist_ok=False)
        os.makedirs(self.tb_dir, exist_ok=False)

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.file_path = os.path.join(self.log_dir, self.file_name)
        file_handler = logging.FileHandler(
            filename=self.file_path, encoding='utf-8', mode='w')
        file_handler.setFormatter(MyFormatter(datefmt='%Y%m%d %H:%M:%S'))
        self.addHandler(file_handler)
        self.info('Log file set to {}'.format(self.file_path))
        return
