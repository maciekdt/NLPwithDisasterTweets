from custom_trainer import CustomTrainer
import optuna
import argparse
import json
import torch
import gc

parser = argparse.ArgumentParser()
parser.add_argument("--trials", type = int, default = 30)
parser.add_argument("--epochs", type = int, default = 10)
parser.add_argument("--patience", type = int, default = 3)
args = parser.parse_args()

def optimize_bert(trial):
    lr = trial.suggest_float('lr', 1e-6, 1e-1, log=True)
    lr_pre = trial.suggest_float('lr_pre', 1e-6, 1e-1, log=True)
    batch_size = trial.suggest_int("batch_size", 8, 64, log=True)
    model_id = trial.suggest_categorical('model_id', ['bert-base-uncased', 'bert-large-uncased'])
    
    trainer = CustomTrainer()
    
    max_f1 = trainer.train(
        model_id=model_id,
        lr=lr,
        lr_pre=lr_pre,
        batch_size=batch_size,
        optim_mode=True,
        max_epochs=args.epochs,
        patience=args.patience,
    )
    
    del trainer
    torch.cuda.empty_cache()
    gc.collect()  
    
    return max_f1


log_path = "optuna_logs/"
study = optuna.create_study(
    direction="maximize",
    storage = "sqlite:///" + log_path  + "study.db",
    load_if_exists = False,
    study_name = "BERT_study"
)

study.optimize(
    optimize_bert,
    n_trials = args.trials, 
    show_progress_bar = True
)

best_params = study.best_params
best_params_path = log_path + "best_params_dqn.json"

with open(best_params_path, "w") as f:
    json.dump(best_params, f, indent=4)