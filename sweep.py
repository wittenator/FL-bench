import hydra
from omegaconf import OmegaConf
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
import sys

from generate_data import main as generate_data__main
from main import main as train_main

@hydra.main(config_path="config", config_name="sweep.yaml", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    split_argument_hash = generate_data__main(cfg.datasets)

    cfg.experiment.dataset.name = cfg.datasets.dataset
    cfg.experiment.dataset.split_args_hash = split_argument_hash

    train_main(cfg.experiment)

if __name__ == "__main__":
    # For gather the Fl-bench logs and hydra logs
    # Otherwise the hydra logs are stored in ./outputs/...
    sys.argv.append(
        "hydra.run.dir=./out/${experiment.method}/${datasets.dataset}/${now:%Y-%m-%d-%H-%M-%S}"
    )
    main()
