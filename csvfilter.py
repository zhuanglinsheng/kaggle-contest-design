"""
Handle csv file of large scale
"""
import csv
from io import StringIO
from collections import defaultdict
from typing import Iterable

CsvParsedValue = bool | int | float | str | None

class DataView:
	"""
	"""
	data: dict[str, list[CsvParsedValue]]
	__abbrsize: int = 5

	def __init__(self,
			data: dict[str, list[CsvParsedValue]],
	) -> None:
		self.data = data

	def __repr__(self) -> str:
		buffer = StringIO()
		for key, value in self.data.items():
			buffer.write(key)
			buffer.write(': ')
			if len(value) > self.__abbrsize:
				head_items = value[:self.__abbrsize - 1]
				tail_item = value[-1]
				buffer.write(f"[{', '.join(map(str, head_items))}, ..., {str(tail_item)}]")
			else:
				buffer.write(str(value))
			buffer.write('\n')
		return buffer.getvalue()

def csv_filter(
		io_input: Iterable[str],
		fields: list[str],
		fields_type: list[type],
		n: int
) -> DataView:
	"""
	Args:
		- n: `-1` means real all
	"""
	reader = csv.DictReader(io_input)
	idxrow = 0
	data: defaultdict[str, list[CsvParsedValue]] = defaultdict(list)
	for idxrow, row in enumerate(reader):
		if 0 <= n <= idxrow:
			break
		for field, field_type in zip(fields, fields_type):
			row_value: str = row[field]
			match field_type:
				case _ if field_type is int:
					try:
						value = int(row_value)
					except:
						value = None
					data[field].append(value)
				case _ if field_type is float:
					try:
						value = float(row_value)
					except:
						value = None
					data[field].append(value)
				case _ if field_type is str:
					value = str(row_value)
				case _:
					ValueError(f'type `{field_type}` is not supported')
	return DataView(dict(data))
