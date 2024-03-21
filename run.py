# import hydra
from importlib import import_module
from configs.configs import BaseConfig, build_config
from omegaconf import OmegaConf, DictConfig
from hydra.core.config_store import ConfigStore
from summarize import summarize_results

PERSPECTIVES = {
    "stereotype": "perspectives.stereotype.bias_generation",
    # "advglue": "perspectives.advglue.gpt_eval",
    # "toxicity": "perspectives.toxicity.text_generation_hydra",
    # "fairness": "perspectives.fairness.fairness_evaluation",
    # "privacy": "perspectives.privacy.privacy_evaluation",
    # # "adv_demonstration": "perspectives.adv_demonstration.adv_demonstration_hydra",     # NOT Interested
    # "machine_ethics": "perspectives.machine_ethics.test_machine_ethics",
    # # "ood": "perspectives.ood.evaluation_ood"                                           # NOT Interested
}


# cs = ConfigStore.instance()
# cs.store(name="config", node=BaseConfig)
# cs.store(name="slurm_config", node=BaseConfig)


# @hydra.main(config_path="configs", config_name="config", version_base="1.2")
def run(config: DictConfig) -> None:
    # The 'validator' methods will be called when you run the line below
    config: BaseConfig = OmegaConf.to_object(config)
    assert isinstance(config, BaseConfig)
    print(config)

    for name, module_name in PERSPECTIVES.items():
        if getattr(config, name) is not None:
            perspective_module = import_module(module_name)
            perspective_module.main(config)

    summarize_results()


if __name__ == "__main__":
    config = build_config(perspectives=PERSPECTIVES)
    run(config=config)
