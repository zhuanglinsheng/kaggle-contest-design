"""
"""
import zipfile
import csv
import io
import os
from collections import defaultdict

# local
from .utils import TableName

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
		n: int=1
	) -> dict[str, list[str]]:
	"""Get the header of designated dataset
	"""
	# check input
	assert n > 0, '`n` should be positive'
	assert n < 10, '`n` should not be too large'

	header: defaultdict[str, list[str]] = defaultdict(list)
	idxrow = int(0)

	zipf = _load_zipfile(download_path)
	with zipf.open(dataset + '.csv') as f:
		reader = csv.DictReader(io.TextIOWrapper(f, encoding='utf-8'))
		for row in reader:
			for key, value in row.items():
				header[key].append(value)
			idxrow += 1
			if idxrow >= n:
				break
	return dict(header)



def table_filter(
		dataset: TableName,
		download_path: str,
		columns: list[TableName],
		conditions: list[str]
):
	"""
	"""
