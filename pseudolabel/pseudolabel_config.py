from dataclasses import dataclass
import pandas as pd

from pseudolabel import utils


@dataclass
class PseudolabelConfig:
    t0_melloddy_path: str
    t1_melloddy_path: str
    t2_melloddy_path: str

    t2_images_path: str
    t_images_features: str

    t8c: str

    key_json: str
    parameters_json: str
    ref_hash_json: str

    def __check_files_exist(self):
        utils.check_file_exists(self.t0_melloddy_path)
        utils.check_file_exists(self.t1_melloddy_path)
        utils.check_file_exists(self.t2_melloddy_path)

        utils.check_file_exists(self.t2_images_path)
        utils.check_file_exists(self.t_images_features)

        utils.check_file_exists(self.t8c)

        utils.check_file_exists(self.key_json)
        utils.check_file_exists(self.parameters_json)
        utils.check_file_exists(self.ref_hash_json)

    def __t_images_check(self):
        t_images = pd.read_csv(self.t_images_features)
        assert (
            sum(t_images.input_compound_id.duplicated()) == 0
        ), f"{self.t_images_features} can't have duplicates"

    def check_data(self):
        self.__check_files_exist()
        self.__t_images_check()
