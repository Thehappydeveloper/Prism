import sys
from pathlib import Path
from typing import Optional
from collections import defaultdict
import numpy as np
import pandas as pd

try:
    import polars as pl
    _HAS_POLARS = True
except ImportError:
    _HAS_POLARS = False

class PlyReader:
    """
    Optimized PLY file reader that parses vertex data into a pandas DataFrame.
    Supports ASCII (fast via Polars/Pandas) and binary PLY (via memmap).
    """

    _sys_byteorder = ('>', '<')[sys.byteorder == 'little']
    _ply_dtypes = {
        b'int8': 'i1', b'char': 'i1',
        b'uint8': 'u1', b'uchar': 'u1',
        b'int16': 'i2', b'short': 'i2',
        b'uint16': 'u2', b'ushort': 'u2',
        b'int32': 'i4', b'int': 'i4',
        b'uint32': 'u4', b'uint': 'u4',
        b'float32': 'f4', b'float': 'f4',
        b'float64': 'f8', b'double': 'f8'
    }
    _valid_formats = {'ascii': '', 'binary_big_endian': '>', 'binary_little_endian': '<'}

    def __init__(self, allow_bool: bool = False):
        if allow_bool:
            self._ply_dtypes[b'bool'] = '?'

    def read(self, filename: str | Path) -> pd.DataFrame:
        """
        Read a PLY file and return its vertex data as a pandas DataFrame.
        """
        filename = Path(filename)
        if not filename.exists():
            raise FileNotFoundError(f"PLY file not found: {filename}")

        fmt, end_header, vertex_count, dtypes = self._parse_header(filename)
        return self._read_data(filename, fmt, end_header, vertex_count, dtypes)

    def _parse_header(self, filename: Path) -> tuple[str, int, int, dict]:
        """
        Parse the PLY header and return format, header end position, vertex count, and dtypes.
        """
        with open(filename, 'rb') as ply:
            if b'ply' not in ply.readline():
                raise ValueError(f"{filename} is not a valid PLY file")

            fmt = ply.readline().decode().strip().split()[1]
            ext = self._valid_formats.get(fmt)
            if ext is None:
                raise ValueError(f"Unsupported PLY format: {fmt}")

            dtypes = defaultdict(list)
            vertex_count = None

            while True:
                line = ply.readline()
                if not line:
                    break
                tokens = line.decode().strip().split()
                if not tokens:
                    continue
                if tokens[0] == 'element' and tokens[1] == 'vertex':
                    vertex_count = int(tokens[2])
                elif tokens[0] == 'property' and vertex_count is not None:
                    dtype_name = tokens[1].encode()
                    name = tokens[2]
                    dtypes['vertex'].append((name, ext + self._ply_dtypes[dtype_name]))
                elif tokens[0] == 'end_header':
                    break

            if vertex_count is None:
                raise ValueError(f"Vertex count not found in {filename}")

            end_header = ply.tell()

        return fmt, end_header, vertex_count, dtypes

    def _read_data(
        self,
        filename: Path,
        fmt: str,
        end_header: int,
        vertex_count: int,
        dtypes: dict
    ) -> pd.DataFrame:
        """
        Read vertex data based on header info.
        """
        with open(filename, 'rb') as ply:
            ply.seek(end_header)

            if fmt == 'ascii':
                # ✅ Fast ASCII parsing
                if _HAS_POLARS:
                    df = pl.read_csv(
                        ply,
                        separator=" ",
                        has_header=False,
                        n_rows=vertex_count,
                        ignore_errors=True
                    )
                    df = df.to_pandas()
                else:
                    df = pd.read_csv(
                        ply,
                        sep=r"\s+",
                        header=None,
                        nrows=vertex_count,
                        engine="c"
                    )

                df.columns = [name for name, _ in dtypes['vertex']]
                return df

            else:
                # ✅ Faster binary reading with memmap
                dt = np.dtype(dtypes['vertex'])
                arr = np.memmap(
                    filename,
                    dtype=dt,
                    mode="r",
                    offset=end_header,
                    shape=(vertex_count,)
                )
                if self._valid_formats[fmt] != self._sys_byteorder:
                    arr = arr.byteswap().newbyteorder()
                return pd.DataFrame(arr)
