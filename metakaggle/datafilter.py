"""
"""
import zipfile
import csv
import io
import os
from collections import defaultdict

# local
from utils import TableName

# third-party
import pandas as pd

def _load_zipfile(download_path: str) -> zipfile.ZipFile:
	"""Internal use for loading zip file
	"""
	zip_path = os.path.join(download_path, 'meta-kaggle.zip')
	zipf = zipfile.ZipFile(zip_path, 'r')
	return zipf

def tables(download_path: str) -> list[str]:
	"""Get the name of all tables
	"""
	table_names: list[str] = []
	zipf = _load_zipfile(download_path)
	for tmp in zipf.namelist():
		table_names.append(tmp.rsplit('.', 1)[0])
	return table_names

def table_header(
		dataset: TableName,
		download_path: str,
		n: int = 1
	) -> dict[str, list[str]]:
	"""Get the header of designated dataset
	"""
	# check input
	assert n >= 0, '`n` should be nonnegative'
	assert n < 10, '`n` should not be too large'

	header: defaultdict[str, list[str]] = defaultdict(list)
	idxrow = int(0)

	zipf = _load_zipfile(download_path)
	with zipf.open(dataset + '.csv') as f:
		io_input = io.TextIOWrapper(f, encoding='utf-8')
		reader = csv.DictReader(io_input)
		for row in reader:
			if n == 0:
				for key in row:
					header[key] = []
				break
			elif n > 0:
				idxrow += 1
				if idxrow > n:
					break
			for key, value in row.items():
				header[key].append(value)
	return dict(header)

def table_filter(
		dataset: TableName,
		download_path: str,
		fields: list[str],
		fields_index: list[str] | None = None,
		fields_datetime: list[str] | None = None,
		n: int = -1
) -> pd.DataFrame:
	"""
	"""
	zipf = _load_zipfile(download_path)
	with zipf.open(dataset + '.csv') as f:
		if n >= 0:
			tbl = pd.read_csv(f,  # type: ignore
				usecols = fields,
				low_memory = False,
				nrows = n)
		else:
			tbl = pd.read_csv(f,  # type: ignore
				usecols = fields,
				low_memory = False)
	# Set datetime
	if fields_datetime is not None:
		for field in fields_datetime:
			tbl[field] = pd.to_datetime(tbl[field])  # type: ignore
	# To Do: set index
	if fields_index is not None:
		for field in fields_index:
			tbl.set_index(fields_index)  # type: ignore
	return tbl
