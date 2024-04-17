'''
 @FileName    : cmd_parser.py
 @EditTime    : 2022-09-27 14:29:39
 @Author      : Buzhen Huang
 @Email       : hbz@seu.edu.cn
 @Description : Parser YAML config file
'''

# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os
import configargparse

def parse_config(argv=None):
    arg_formatter = configargparse.ArgumentDefaultsHelpFormatter
    cfg_parser = configargparse.YAMLConfigFileParser
    description = 'DL project template'
    parser = configargparse.ArgParser(formatter_class=arg_formatter,
                                      config_file_parser_class=cfg_parser,
                                      description=description,
                                      prog='template')

    parser.add_argument('--trainset',
                        default='',
                        type=str,
                        help='trainset.')
    parser.add_argument('--testset',
                        default='',
                        type=str,
                        help='testset.')
    parser.add_argument('--data_folder',
                        default='',
                        help='The directory that contains the data.')
    parser.add_argument('-c', '--config',
                        required=True, is_config_file=True,
                        help='config file path')
    parser.add_argument('--note',
                        default='test',
                        type=str,
                        help='code note')
    parser.add_argument('--lr',
                        default=0.001,
                        type=float,
                        help='learning rate.')
    parser.add_argument('--batchsize',
                        default=10,
                        type=int,
                        help='batch size.')
    parser.add_argument('--epoch',
                        default=500,
                        type=int,
                        help='num epoch.')
    parser.add_argument('--worker',
                        default=0,
                        type=int,
                        help='workers for dataloader.')
    parser.add_argument('--mode',
                        default='',
                        type=str,
                        help='running mode.')        
    parser.add_argument('--pretrain',
                        default=False,
                        type=lambda x: x.lower() in ['true', '1'],
                        help='True for use pretrain parameters.')
    parser.add_argument('--use_sch',
                        default=False,
                        type=lambda x: x.lower() in ['true', '1'],
                        help='.')
    parser.add_argument('--pretrain_dir',
                        default='',
                        type=str,
                        help='The directory that contains the pretrain model.')
    parser.add_argument('--model',
                        default='',
                        type=str,
                        help='the model used for this project.')
    parser.add_argument('--output',
                        default='output',
                        type=str,
                        help='the output')
    parser.add_argument('--train_loss',
                        default='L1 partloss',
                        type=str,
                        help='training loss type.')
    parser.add_argument('--test_loss',
                        default='L1',
                        type=str,
                        help='testing loss type.')
    parser.add_argument('--viz',
                        default=False,
                        type=lambda x: x.lower() in ['true', '1'],
                        help='True for visualize input.')
    parser.add_argument('--use_prior',
                        default=False,
                        type=lambda x: x.lower() in ['true', '1'],
                        help='True for visualize input.')
    parser.add_argument('--task',
                        default='ed_train',
                        type=str,
                        help='ee_train: encoder-encoder only, else ed_train.')
    parser.add_argument('--gpu_index',
                        default=0,
                        type=int,
                        help='gpu index.')
    parser.add_argument('--frame_length',
                        default=16,
                        type=int,
                        help='frame_length.')
    parser.add_argument('--model_type',
                        default='smpl',
                        type=str,
                        help='ee_train: encoder-encoder only, else ed_train.')
    args = parser.parse_args()
    args_dict = vars(args)
    return args_dict
