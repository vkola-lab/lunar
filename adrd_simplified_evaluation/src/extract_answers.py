from omegaconf import OmegaConf
from pathlib import Path
from answer_extractor import AnswerExtractor


if __name__ == "__main__":

    llm_extractor_config_path = "/projectnb/vkolagrp/bellitti/adrd-foundation-model/adrd_simplified_evaluation/src/extractor_config.yml"

    extractor = AnswerExtractor(llm_extractor_config_path=llm_extractor_config_path)

    ans_path = Path('/projectnb/vkolagrp/bellitti/adrd-foundation-model/adrd_simplified_evaluation/results_sub/test_cog/')

    extractor.extract_from_dir(ans_path)