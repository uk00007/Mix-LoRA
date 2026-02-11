import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import datasets as hf_datasets
import evaluate as hf_evaluate
import torch

from moe_peft.common import InputData, Prompt


class BasicMetric:
    def __init__(self) -> None:
        pass

    def add_batch(self, predictions: torch.Tensor, references: torch.Tensor):
        pass

    def compute(self) -> Dict[str, Any]:
        pass


class AutoMetric(BasicMetric):
    def __init__(self, task_name: str) -> None:
        super().__init__()
        path_prefix = os.getenv("MOE_PEFT_METRIC_PATH")
        if path_prefix is None:
            path_prefix = ""
        elif not path_prefix.endswith(os.sep):
            path_prefix += os.sep

        if ":" in task_name:
            split = task_name.split(":")
            self.metric_ = hf_evaluate.load(path_prefix + split[0], split[1])
        else:
            self.metric_ = hf_evaluate.load(path_prefix + task_name)

    def add_batch(self, predictions: torch.Tensor, references: torch.Tensor):
        self.metric_.add_batch(predictions=predictions, references=references)

    def compute(self) -> Dict[str, Any]:
        return self.metric_.compute()


class BasicTask:
    def __init__(self) -> None:
        pass

    @property
    def peft_task_type(self) -> str:
        pass

    def loading_data(
        self, is_train: bool = True, path: Optional[str] = None
    ) -> List[InputData]:
        pass

    def loading_metric(self) -> BasicMetric:
        pass

    def init_kwargs(self) -> Dict:
        return {}


# Casual Fine-tuning Tasks
# Instant-Created Class
class CasualTask(BasicTask):
    @property
    def peft_task_type(self) -> str:
        return "CAUSAL_LM"

    def loading_data(
        self, is_train: bool = True, path: Optional[str] = None
    ) -> List[InputData]:
        assert path is not None, "Casual supervised fine-tuning requires data path."
        assert is_train, "Casual supervised fine-tuning task only supports training."
        # Loading dataset
        if path.endswith(".json") or path.endswith(".jsonl"):
            data = hf_datasets.load_dataset("json", data_files=path)
        elif ":" in path:
            split = path.split(":")
            data = hf_datasets.load_dataset(split[0], split[1])
        else:
            data = hf_datasets.load_dataset(path)
        ret: List[InputData] = []
        for data_point in data["train"]:
            ret.append(
                InputData(
                    inputs=Prompt(
                        instruction=data_point["instruction"],
                        input=data_point.get("input", None),
                        label=data_point.get("output", None),
                    )
                )
            )

        return ret


# Sequence Classification
class SequenceClassificationTask(BasicTask):
    def __init__(
        self,
        task_name: str,
        task_type: str,
        label_dtype: torch.dtype,
        num_labels: int,
        dataload_function: Callable,
        # Setting to `None` corresponds to the task name.
        metric_name: Optional[str] = None,
        # The default values are "train" and "validation".
        subset_map: Optional[Tuple[str, str]] = ("train", "validation"),
    ) -> None:
        super().__init__()
        self.task_name_ = task_name
        self.task_type_ = task_type
        self.label_dtype_ = label_dtype
        self.num_labels_ = num_labels
        self.dataload_function_ = dataload_function
        if metric_name is None:
            self.metric_name_ = task_name
        else:
            self.metric_name_ = metric_name
        self.subset_map_ = subset_map

    @property
    def peft_task_type(self) -> str:
        return "SEQ_CLS"

    def loading_data(
        self, is_train: bool = True, path: Optional[str] = None
    ) -> List[InputData]:
        if ":" in self.task_name_:
            split = self.task_name_.split(":")
            data = hf_datasets.load_dataset(
                split[0] if path is None else path, split[1]
            )
        else:
            data = hf_datasets.load_dataset(self.task_name_ if path is None else path)
        data = data[self.subset_map_[0] if is_train else self.subset_map_[1]]
        logging.info(f"Preparing data for {self.task_name_.upper()}")
        ret: List[InputData] = []
        for data_point in data:
            inputs, labels = self.dataload_function_(data_point)
            assert isinstance(labels, List)
            ret.append(InputData(inputs=inputs, labels=labels))

        return ret

    def loading_metric(self) -> BasicMetric:
        return AutoMetric(self.metric_name_)

    def init_kwargs(self) -> Dict:
        return {
            "task_type": self.task_type_,
            "num_labels": self.num_labels_,
            "label_dtype": self.label_dtype_,
        }


# Common Sense
class CommonSenseTask(BasicTask):
    def __init__(self) -> None:
        super().__init__()
        self.task_type_ = "common_sense"
        self.label_dtype_ = None

    @property
    def peft_task_type(self) -> str:
        return "QUESTION_ANS"

    def label_list(self) -> List[str]:
        pass


task_dict = {}


# Multi-Task (Only for train)
class MultiTask(BasicTask):
    def __init__(self, task_names: str) -> None:
        super().__init__()
        self.task_type_ = "multi_task"
        self.label_dtype_ = None
        self.task_list_: List[BasicTask] = []
        task_names = task_names.split(";")
        for name in task_names:
            self.task_list_.append(task_dict[name])

    def loading_data(
        self, is_train: bool = True, path: Optional[str] = None
    ) -> List[InputData]:
        logging.info(f"Preparing data for {len(self.task_list_)} tasks")
        path_list = None if path is None else path.split(";")
        data: List[InputData] = []
        assert is_train
        for idx, task in enumerate(self.task_list_):
            path: str = "" if path_list is None else path_list[idx].strip()
            data.extend(task.loading_data(is_train, None if len(path) == 0 else path))
        return data

# ... (Keep all existing code above unchanged) ...

# --- FIX START: Add Custom Task Definitions ---

# 1. Define Sentiment Analysis (SST-2) as a Generative Task
class SST2GenTask(CasualTask):
    def loading_data(self, is_train: bool = True, path: Optional[str] = None) -> List[InputData]:
        if not is_train:
            return []
        # Load the official GLUE/SST2 dataset
        logging.info("Loading GLUE SST-2 dataset...")
        dataset = hf_datasets.load_dataset("glue", "sst2")
        # Use a subset for speed if needed, otherwise use full 'train'
        split = dataset["train"] if is_train else dataset["validation"]


        if is_train:
            split = split.select(range(5000))
        
        ret: List[InputData] = []
        for item in split:
            # Convert label 0/1 to text "negative"/"positive"
            label_text = "positive" if item["label"] == 1 else "negative"
            
            # Format as an instruction for the model
            ret.append(
                InputData(
                    inputs=Prompt(
                        instruction="Analyze the sentiment of the following sentence. Answer positive or negative.",
                        input=item["sentence"],
                        label=label_text
                    )
                )
            )
        return ret

# 2. Define Summarization (CNN/DailyMail) as a Generative Task
# 2. Define Summarization (CNN/DailyMail) as a Generative Task
class CNNDMTask(CasualTask):
    def loading_data(self, is_train: bool = True, path: Optional[str] = None) -> List[InputData]:
        if not is_train:
            return []
        logging.info("Loading CNN/DailyMail dataset...")
        dataset = hf_datasets.load_dataset("cnn_dailymail", "3.0.0")
        split = dataset["train"] if is_train else dataset["validation"]
        if is_train:
            split = split.select(range(10000))
        ret: List[InputData] = []
        for item in split:
            # --- CRITICAL FIX: TRUNCATION ---
            # We cut the article to 4000 characters (approx 1000 tokens).
            # This leaves plenty of room for the instruction + summary 
            # within the model's 2048 token limit.
            article_text = item["article"]
            if len(article_text) > 4000:
                article_text = article_text[:4000]

            ret.append(
                InputData(
                    inputs=Prompt(
                        instruction="Summarize the following article.",
                        input=article_text, 
                        label=item["highlights"]
                    )
                )
            )
        return ret

# 3. Register these tasks into the global dictionary so launch.py can find them
task_dict["glue/sst2"] = SST2GenTask()
task_dict["cnn_dailymail"] = CNNDMTask()

# --- FIX END ---