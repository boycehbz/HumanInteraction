'''
 @FileName    : train.py
 @EditTime    : 2024-04-01 16:05:48
 @Author      : Buzhen Huang
 @Email       : buzhenhuang@outlook.com
 @Description : 
'''

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from torch.utils.data import DataLoader
from cmd_parser import parse_config
from utils.module_utils import seed_worker, set_seed
from modules import init, LossLoader, ModelLoader, DatasetLoader

###########Load config file in debug mode#########
# import sys
# sys.argv = ['','--config=cfg_files/config.yaml'] # interVAE config


def main(**args):
    seed = 7
    g = set_seed(seed)

    # Global setting
    dtype = torch.float32
    batchsize = args.get('batchsize')
    num_epoch = args.get('epoch')
    workers = args.get('worker')
    device = torch.device(index=args.get('gpu_index'), type='cuda')
    mode = args.get('mode')

    # Initialize project setting, e.g., create output folder, load SMPL model
    out_dir, logger, smpl = init(dtype=dtype, **args)

    # Load loss function
    loss = LossLoader(smpl, device=device, **args)

    # Load model
    model = ModelLoader(dtype=dtype, device=device, out_dir=out_dir, **args)

    # create data loader
    dataset = DatasetLoader(dtype=dtype, smpl=smpl, **args)
    if mode == 'train':
        train_dataset = dataset.load_trainset()
        train_loader = DataLoader(
            train_dataset,
            batch_size=batchsize, shuffle=True,
            num_workers=workers, pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g,
        )
        if args.get('use_sch'):
            model.load_scheduler(train_dataset.cumulative_sizes[-1])
    test_dataset = dataset.load_testset()
    test_loader = DataLoader(
        test_dataset,
        batch_size=batchsize, shuffle=False,
        num_workers=workers, pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    # Load handle function with the task name
    task = args.get('task')
    exec('from process import %s_train' %task)
    exec('from process import %s_test' %task)

    for epoch in range(num_epoch):
        # training mode
        if mode == 'train':
            training_loss = eval('%s_train' %task)(model, loss, train_loader, epoch, num_epoch, device=device)

            # # save trained model
            # if (epoch) % 1 == 0:
            #     model.save_model(epoch, task)

            if (epoch) % 1 == 0:
                testing_loss = eval('%s_test' %task)(model, loss, test_loader, epoch, device=device)
                lr = model.optimizer.state_dict()['param_groups'][0]['lr']
                logger.append([int(epoch + 1), lr, training_loss, testing_loss])
                # logger.plot(['Train Loss', 'Test Loss'])
                # savefig(os.path.join(out_dir, 'log.jpg'))
            else:
                testing_loss = -1.

            # save trained model
            model.save_best_model(testing_loss, epoch, task)

        # testing mode
        elif epoch == 0 and mode == 'test':
            training_loss = -1.
            testing_loss = eval('%s_test' %task)(model, loss, test_loader, epoch, device=device)

            lr = model.optimizer.state_dict()['param_groups'][0]['lr']
            logger.append([int(epoch + 1), lr, training_loss, testing_loss])

    logger.close()


if __name__ == "__main__":
    args = parse_config()

    main(**args)




