import pandas as pd
from pathlib import Path
from typing import Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from fov_predictors.base import FovPredictor


class Fov:
    def __init__(self, csv_path: str, predictor: "FovPredictor", start_frame_index:int = 0,) -> None:
        """
        Initialize Fov object with a CSV file path.
        Loads the CSV into memory.
        """
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
        
        self.df: pd.DataFrame = pd.read_csv(self.csv_path)
        self.df.set_index("FrameNumber", inplace=True)
        self.df.index = self.df.index.astype(int)
        self.predictor = predictor
        self.start_frame_index = start_frame_index

    def get_frame(self, frame_number: int) -> Optional[pd.Series]:
        """
        Get a specific frame row by frame_number.
        Returns a pandas Series if found, else None.
        """
        try:
            return self.df.loc[frame_number]
        except KeyError:
            print(f"Frame {frame_number} not found in {self.csv_path.name}.")
            return None

    def get_total_frames(self) -> int:
        """
        Return total number of frames in this CSV.
        """
        return len(self.df)

    def get_frames_range(self, start_frame: int, end_frame: int) -> pd.DataFrame:
        """
        Return all rows whose FrameNumber index is in [start_frame, end_frame].
        Missing frames are skipped.
        """
        if end_frame < start_frame:
            return self.df.iloc[0:0].copy()
        mask = (self.df.index >= int(start_frame)) & (self.df.index <= int(end_frame))
        return self.df.loc[mask].copy()

    def get_history(self, frame_number: int, count_back: int, include_current: bool = True) -> pd.DataFrame:
        """
        Return a backward window ending at frame_number.
        If include_current=False, history ends at frame_number-1.
        """
        if count_back <= 0:
            return self.df.iloc[0:0].copy()
        end = int(frame_number) if include_current else int(frame_number) - 1
        start = end - int(count_back) + 1
        return self.get_frames_range(start, end)

    def get_all_frames(self) -> pd.DataFrame:
        """
        Return the full dataframe (all frames).
        """
        return self.df.copy()
    
    def predict(self, frame_number: int):
        """
        Use the predictor to get position and forward vector for a given frame.
        """
        return self.predictor.predict(frame_number, self)
