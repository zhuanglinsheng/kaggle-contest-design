"""
Handle csv file of large scale
"""

from typing import Literal
from typing import Iterable, Sequence


class FilterCondition:
	"""Conditions for Filtering

	Attributes:
		left
		right
		cond
	"""
	_left: str
	_right: str
	_cond: Literal['<=', '==', '>=', '<', '>']

	def __init__(self,
			left: str,
			cond: Literal['<=', '==', '>=', '<', '>'],
			right: str
	) -> None:
		self._left = left
		self._right = right
		self._cond = cond

	def __repr__(self) -> str:
		return self._left + ' ' + self._cond + ' ' + self._right


def csv_filter(
		f: Iterable[str],
		fieldnames: Sequence[str] | None = None,
		restkey: str | None = None,
		restval: str | None = None,
		dialect: str = "excel",
		*,
		delimiter: str = ",",
		quotechar: str | None = '"',
		escapechar: str | None = None,
		doublequote: bool = True,
		skipinitialspace: bool = False,
		lineterminator: str = "\r\n",
		quoting: int = 0,
		strict: bool = False
) -> None:
	pass

