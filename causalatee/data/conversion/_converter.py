from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from ..constants import Task
from ..utils import verify_dataset


class FormatConverter(ABC):
    def __init__(self, target: Path, subdirs_for_tasks: bool = True) -> None:
        super().__init__()
        self._target = target
        self._subdirs_for_tasks = subdirs_for_tasks

    def __get_file_path(self, task: Task, split: str) -> Path:
        task_dir = task.replace(" ", "-")
        if self._subdirs_for_tasks:
            return self._target / task_dir / f"{split}.parquet"
        return self._target / f"{task_dir}-{split}.parquet"

    @abstractmethod
    def _convert(self, task: Task, split: str) -> pd.DataFrame: ...

    def convert(self, task: Task, split: str) -> None:
        df = self._convert(task, split)
        for error in verify_dataset(df, task):
            print(f"WARNING [{type(self).__name__} {task}/{split}]: {error}")
        path = self.__get_file_path(task, split)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, engine="pyarrow")
