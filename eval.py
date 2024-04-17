'''
 @FileName    : eval.py
 @EditTime    : 2024-04-03 14:41:12
 @Author      : Buzhen Huang
 @Email       : buzhenhuang@outlook.com
 @Description : 
'''
import torch
from cmd_parser import parse_config
from utils.module_utils import seed_worker, set_seed
from modules import init, LossLoader, ModelLoader, DatasetLoader
from utils.eval_utils import HumanEval

# ###########Load config file in debug mode#########
# import sys
# sys.argv = ['','--config=cfg_files/eval.yaml'] 

def main(**args):
    seed = 7
    g = set_seed(seed)

    # Global setting
    is_seq = False
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
    eval_dataset = dataset.load_evalset()

    # Load handle function with the task name
    task = args.get('task')
    exec('from process import %s_eval' %task)
    
    for i, (name, dataset) in enumerate(zip(dataset.testset, eval_dataset)):
        eval_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.get('batchsize'), shuffle=False,
            num_workers=args.get('worker'), pin_memory=True, drop_last=False
        )
        pred, gt = eval('%s_eval' %task)(model, eval_loader, loss, device=device)

        evaluator = HumanEval(name)
        evaluator(pred, gt)
        evaluator.report()


if __name__ == "__main__":
    args = parse_config()
    main(**args)





