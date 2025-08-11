from omegaconf import OmegaConf
from pathlib import Path
from answer_extractor import AnswerExtractor


if __name__ == "__main__":


    extractor = AnswerExtractor()

    ans_path = Path('/projectnb/vkolagrp/bellitti/adrd-foundation-model/adrd_simplified_evaluation/results_sub/test_cog/')

    extractor.extract_from_dir(ans_path)