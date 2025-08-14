from omegaconf import OmegaConf
from pathlib import Path
from answer_extractor import AnswerExtractor
import sys

if __name__ == "__main__":

    llm_extractor_config_path = sys.argv[2]

    extractor = AnswerExtractor(llm_extractor_config_path=llm_extractor_config_path)

    ans_path = Path(sys.argv[1])

    extractor.extract_from_dir(ans_path)