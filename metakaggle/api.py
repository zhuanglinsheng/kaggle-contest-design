"""
"""
import os

# third-party packages
from kaggle.api import kaggle_api_extended  # type: ignore

def download_rawdata(
		download_path: str,
		force: bool = False,
		quiet: bool = False,
		licenses: list[str] = []
) -> None:
	"""Download rawdata from https://www.kaggle.com/datasets/kaggle/meta-kaggle

	Note: kaggle api is required. Hence,
	1. install Python package `kaggle`
	2. [create token](https://www.kaggle.com/docs/api) and
	3. put the `kaggle.json` file to the `.kaggle` folder of your home directory
	"""
	os.makedirs(download_path, exist_ok=True)
	api = kaggle_api_extended.KaggleApi()
	api.authenticate()
	api.dataset_download_files(  # type: ignore
		'kaggle/meta-kaggle',
		download_path, force, quiet, False, licenses)
