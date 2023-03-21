import os
from typing import Dict, List, Optional

import torch

from sentiment_classification.module_utils import PreTrainedModule
from sentiment_classification.sentence_level.model import Model
from sentiment_classification.sentence_level.utils import convert_to_unicode
from sentiment_classification.tokenization.tokenization import (
    Tokenizer as BertTokenizer,
)


class Classifier(PreTrainedModule):
    """sentiment classification .


    Attributes:
        config (Optional[Dict],optional):
            Modeling config.
            Defaults to None.
    """

    def __init__(
        self,
        *,
        config: Optional[Dict] = None,
    ) -> None:
        super().__init__()

        if config is None:
            config = {}
        self.config = config

        self._tokenizer = BertTokenizer(maxlen=self._maxlen)
        self._model = Model(self.config)

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, config: Dict):
        self._config = config
        self._vocab_size = config.get("vocab_size", 50000)
        self._hidden_size = config.get("hidden_size", 512)
        self._num_heads = config.get("num_heads", 8)
        self._maxlen = config.get("maxlen", 512)
        self._n_layers = config.get("n_layers", 8)
        self._pad_idx = config.get("pad_idx", 0)

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        return self._model

    def load(self, model: str) -> None:
        """Load  state dict from huggingface repo or local model path or dict.
        Args:
            model (str):
                Model file need to be loaded.
                Can be either:
                    path of a pretrained model.
                    model repo.
        Raises:
            ValueError: str model should be a path!
        """
        if model in self._PRETRAINED_LIST:
            model = self.download(model)
        if isinstance(model, str):
            if os.path.isdir(model):
                self._load_from_dir(model)
            elif os.path.isfile(model):
                dir = os.path.join(self._tmpdir.name, "sentiment_classification")
                if os.path.exists(dir):
                    pass
                else:
                    os.mkdir(dir)
                self._unzip2dir(model, dir)
                self._load_from_dir(dir)
            else:
                raise ValueError("""str model should be a path!""")

        else:
            raise ValueError("""str model should be a path!""")

    def _load_from_dir(self, model_dir: str) -> None:
        """Set model params from `model_file`.

        Args:
            model_dir (str):
                Dir containing model params.
        """
        model_files = os.listdir(model_dir)

        # config
        if "config.pkl" not in model_files:
            raise FileNotFoundError("""config should in model dir!""")

        config = self._load_pkl(os.path.join(model_dir, "config.pkl"))
        self.config = config

        # model
        if "model.pkl" not in model_files:
            raise FileNotFoundError("""classifier should in model dir!""")

        self._model = Model(self._config)
        self._model.load_state_dict(
            torch.load(os.path.join(model_dir, "model.pkl"), map_location="cpu")
        )
        self._model.eval()

        # bert_tokenizer
        if "bert_tokenizer.pkl" in model_files:
            self._tokenizer.load(os.path.join(model_dir, "bert_tokenizer.pkl"))
        else:
            raise FileNotFoundError("""bert_tokenizer should in model dir!""")

    def _get_rank_from_score(self, score: List[float]) -> float:
        rank = 1
        for s in score:
            if s >= 0.75:
                rank += 1
                continue
            elif s <= 0.25:
                return rank
            else:
                rank += (s - 0.25) / 0.5

        return rank

    def rank(self, text: str) -> Dict[str, float]:
        """Scoring the input text.

        Args:
            input (str):
                Text input.

        Returns:
            Dict[str,float]:
                The toxic score of the input .
        """
        text = convert_to_unicode(text)
        input = self._tokenizer.encode_tensor(
            text, maxlen=self.config.get("maxlen", 512)
        ).view(1, -1)
        score = self._model.score(input).view(-1).tolist()
        rank = self._get_rank_from_score(score)
        return rank

    def batch_rank(self, texts: List[str]) -> List[Dict[str, float]]:
        """Scoring the input text.

        Args:
            input (List[str]):
                Text input.

        Returns:
            List[Dict[str, float]]:
                The toxic score of the input .
        """
        texts = [convert_to_unicode(text) for text in texts]

        input = [
            self._tokenizer.encode_tensor(
                text, maxlen=self.config.get("maxlen", 512)
            ).view(1, -1)
            for text in texts
        ]
        input = torch.cat(input, dim=0)
        scores = self._model.score(input).tolist()
        res = [self._get_rank_from_score(score) for score in scores]
        return res
