from argparse import ArgumentParser, Namespace
from pathlib import Path
import json
import os
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from model_weight_split.backborn.fcn import FCN
from src.task.pipeline import DataPipeline
from src.task.runner import Runner
from src.metric import multilabel_recall
from src.net.linear_cls import LinearCls
import wandb

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_config(args: Namespace) -> DictConfig:
    parent_config_dir = Path("conf/")
    dataset_config_dir = parent_config_dir / "dataset"
    backborn_config_dir = parent_config_dir / "backborn"
    config = OmegaConf.create()
    dataset_config = OmegaConf.load(dataset_config_dir / f"{args.dataset}.yaml")
    backborn_config = OmegaConf.load(backborn_config_dir / f"{args.backborn}.yaml")
    config.update(data=dataset_config, backborn=backborn_config, hparams=vars(args))
    return config

def get_tensorboard_logger(args: Namespace) -> TensorBoardLogger:
    logger = TensorBoardLogger(
        save_dir=f"exp/{args.dataset}", name=f"backborn:{args.backborn}_trained:{args.backborn_data}", version=args.eval_type
    )
    return logger

def get_wandb_logger(model):
    logger = WandbLogger()
    logger.watch(model)
    return logger 

def get_checkpoint_callback(save_path) -> ModelCheckpoint:
    prefix = save_path
    suffix = "Best-{epoch:02d}-{val_loss:.4f}"
    checkpoint_callback = ModelCheckpoint(
        dirpath=prefix,
        filename=suffix,
        save_top_k=1,
        save_last= True,
        monitor="val_loss",
        mode="min",
        save_weights_only=True,
        verbose=True,
    )
    return checkpoint_callback

def get_best_performance(runner, save_path):
    for fnames in os.listdir(save_path):
        if "Best" in fnames:
            checkpoint_path = Path(save_path,fnames)
    state_dict = torch.load(checkpoint_path)
    runner.load_state_dict(state_dict.get("state_dict"))
    return runner

def nearest_neighbor_eval(dataloader, backborn, args):
    embeddings = []
    labels = []
    backborn = backborn.to(args.gpus)
    for batch in dataloader:
        audio, label = batch
        with torch.no_grad():
            embedding = backborn.get_embedding(audio.squeeze(0).to(args.gpus))
        embeddings.append(embedding.detach().cpu())
        labels.append(label)
    embeddings = torch.stack(embeddings).squeeze(1)
    labels = torch.stack(labels).squeeze(1)
    embedding_norm = embeddings / embeddings.norm(dim=-1, keepdim=True)
    sim_matrix = embedding_norm @ embedding_norm.T
    sim_matrix = sim_matrix.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    return  {
        "R@1" : multilabel_recall(sim_matrix, labels, top_k=1),
        "R@2" : multilabel_recall(sim_matrix, labels, top_k=2),
        "R@4" : multilabel_recall(sim_matrix, labels, top_k=4),
        "R@8" : multilabel_recall(sim_matrix, labels, top_k=8),
    }

def main(args) -> None:
    if args.reproduce:
        seed_everything(42)

    save_path = f"exp/{args.dataset}/backborn:{args.backborn}_trained:{args.backborn_data}/{args.eval_type}/"
    config = get_config(args)
    pipeline =  DataPipeline(
        dataset_type = args.dataset, 
        audio_path = config.data.audio_path, 
        split_path = config.data.split_path,
        sr = config.backborn.sample_rate,
        duration = config.backborn.duration,
        batch_size = args.batch_size,
        num_workers = args.num_workers
    )
    backborn = FCN(
        sample_rate = config.backborn.sample_rate,
        n_fft = config.backborn.n_fft,
        f_min = config.backborn.f_min,
        f_max = config.backborn.f_max,
        n_mels = config.backborn.n_mels,
        n_class = config.backborn.n_class
    )
    backborn.load_state_dict(torch.load(f"./model_weight_split/weights/{args.backborn_data}/{args.backborn}/best_model.pth"))
    results = {}
    if args.eval_type == "nn":
        dataset = pipeline.get_dataset(
            dataset_builder = pipeline.dataset_builder,
            audio_path = config.data.audio_path, 
            split_path = config.data.split_path,
            split = "TEST", 
            sr = config.backborn.sample_rate,
            duration = config.backborn.duration
        )
        dataloader = pipeline.get_dataloader(
            dataset = dataset,
            batch_size = 1,
            num_workers = args.num_workers,
            shuffle = False,
            drop_last = False,
        )
        results.update({f"{args.eval_type}" : nearest_neighbor_eval(dataloader, backborn, args)})
    else:
        model = LinearCls(
            backborn = backborn, 
            feature_dim = config.backborn.feature_dim,
            n_calss = config.data.n_class,
            prediction_type = config.data.prediction_type,
            eval_type = args.eval_type
        )
        runner = Runner(
            model = model, 
            lr = args.lr,
            weight_decay = args.weight_decay, 
            T_0 = args.T_0, 
            prediction_type = config.data.prediction_type,
            eval_type = args.eval_type,
            test_case="last"
        )
        logger = get_tensorboard_logger(args)
        checkpoint_callback = get_checkpoint_callback(save_path)
        trainer = Trainer(
            max_epochs= args.max_epochs,
            gpus= [args.gpus],
            distributed_backend= args.distributed_backend,
            benchmark= args.benchmark,
            deterministic= args.deterministic,
            logger=logger,
            callbacks=[
                checkpoint_callback
            ]
        )

        trainer.fit(runner, datamodule=pipeline)
        trainer.test(runner, datamodule=pipeline)
        results.update({f"last_{args.eval_type}": runner.test_results})
        best_runner = Runner(
            model = model, 
            lr = args.lr,
            weight_decay = args.weight_decay, 
            T_0 = args.T_0, 
            prediction_type = config.data.prediction_type,
            eval_type = args.eval_type,
            test_case="best"
        )
        best_runner = get_best_performance(best_runner, save_path)
        trainer.test(best_runner, datamodule=pipeline)
        results.update({f"best_{args.eval_type}": best_runner.test_results})

    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    with open(Path(save_path, f"results.json"), mode="w") as io:
        json.dump(results, io, indent=4)
    OmegaConf.save(config=config, f= Path(save_path, f"hparams.yaml"))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", default="mtat", type=str)
    parser.add_argument("--backborn", default="fcn", type=str)
    parser.add_argument("--backborn_data", default="mtat", type=str)
    parser.add_argument("--eval_type", default="nn", type=str)
    # pipeline
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    # runner
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--weight_decay", default=0.0001, type=float)
    parser.add_argument("--T_0", default=16, type=int)
    parser.add_argument("--max_epochs", default=5, type=int)
    parser.add_argument("--gpus", default=0, type=int)
    parser.add_argument("--distributed_backend", default="dp", type=str)
    parser.add_argument("--deterministic", default=True, type=str2bool)
    parser.add_argument("--benchmark", default=False, type=str2bool)
    parser.add_argument("--reproduce", default=True, type=str2bool)

    args = parser.parse_args()
    main(args)
 