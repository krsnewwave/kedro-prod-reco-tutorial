from annoy import AnnoyIndex
from kedro.io import AbstractDataSet
from kedro.io.core import get_filepath_str, get_protocol_and_path
from typing import Dict, Any
from pathlib import Path


class KedroAnnoyIndex(AbstractDataSet):
    """Wrap ANNOY so it can be included in Kedro data catalog

    Args:
        AbstractDataSet (AbstractDataset): Kedro abstract class
    """

    def __init__(self, filepath, embedding_length, metric) -> None:
        """Creates new instance of wrapper

        Args:
            filepath (str): _description_
            embedding_length (int): _description_
            metric (str): _description_
        """
        # protocol, path = get_protocol_and_path(filepath)
        # self._protocol = protocol
        # self._filepath = PurePosixPath(path)
        self._filepath = Path(filepath)

        self.embedding_length = embedding_length
        self.metric = metric

    def _load(self) -> AnnoyIndex:
        """Loads annoy index

        Returns:
            AnnoyIndex: _description_
        """
        annoy_index = AnnoyIndex(self.embedding_length, self.metric)
        annoy_index.load(self._filepath.as_posix())
        return annoy_index

    def _save(self, annoy_idx: AnnoyIndex) -> None:
        """Saves annoy index

        Args:
            annoy_idx (AnnoyIndex): _description_
        """
        annoy_idx.save(self._filepath.as_posix())

    def _describe(self) -> Dict[str, Any]:
        """Describes dataset`

        Returns:
            Dict[str, Any]: _description_
        """
        return dict(filepath=self._filepath, embedding_length=self.embedding_length, metric=self.metric)
