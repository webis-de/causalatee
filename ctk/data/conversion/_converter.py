from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


class FormatConverter(ABC):
    def __init__(self, target: Path, subdirs_for_tasks: bool = True) -> None:
        super().__init__()
        self._target = target
        self._subdirs_for_tasks = subdirs_for_tasks

    def __get_file_path(self, task: str, split: str) -> Path:
        task = task.replace(" ", "-")
        if self._subdirs_for_tasks:
            return self._target / task / f"{split}.parquet"
        return self._target / f"{task}-{split}.parquet"

    @abstractmethod
    def _convert(self, task: str, split: str) -> pd.DataFrame: ...

    def convert(self, task: str, split: str) -> None:
        df = self._convert(task, split)
        # TODO: task-specific sanity checks
        path = self.__get_file_path(task, split)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, engine="pyarrow")
