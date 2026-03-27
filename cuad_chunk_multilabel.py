from __future__ import annotations

import copy
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup


def build_clause_mappings(cuad_df: pd.DataFrame) -> tuple[dict[str, int], dict[int, str]]:
    clause_names = sorted(cuad_df["clause_type"].unique().tolist())
    clause_to_id = {clause_name: idx for idx, clause_name in enumerate(clause_names)}
    id_to_clause = {idx: clause_name for clause_name, idx in clause_to_id.items()}
    return clause_to_id, id_to_clause


def build_contract_records(cuad_df: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for contract_title, group in cuad_df.groupby("contract_title", sort=False):
        contract_text = group["contract_text"].iloc[0]
        clause_spans: dict[str, list[tuple[int, int]]] = defaultdict(list)

        for row in group.itertuples(index=False):
            for answer_text, answer_start in zip(row.answer_texts, row.answer_starts):
                answer_end = int(answer_start) + len(answer_text)
                clause_spans[row.clause_type].append((int(answer_start), answer_end))

        records.append(
            {
                "contract_title": contract_title,
                "contract_text": contract_text,
                "clause_spans": dict(clause_spans),
            }
        )

    return records


def split_contract_records(
    contract_records: list[dict[str, Any]],
    train_size: float = 0.8,
    val_size: float = 0.1,
    test_size: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("train_size + val_size + test_size must sum to 1.0")

    titles = [record["contract_title"] for record in contract_records]
    title_to_record = {record["contract_title"]: record for record in contract_records}

    train_titles, temp_titles = train_test_split(
        titles,
        train_size=train_size,
        random_state=seed,
        shuffle=True,
    )

    val_ratio_within_temp = val_size / (val_size + test_size)
    val_titles, test_titles = train_test_split(
        temp_titles,
        train_size=val_ratio_within_temp,
        random_state=seed,
        shuffle=True,
    )

    train_records = [title_to_record[title] for title in train_titles]
    val_records = [title_to_record[title] for title in val_titles]
    test_records = [title_to_record[title] for title in test_titles]
    return train_records, val_records, test_records


def _span_overlaps_chunk(span_start: int, span_end: int, chunk_start: int, chunk_end: int) -> bool:
    return max(span_start, chunk_start) < min(span_end, chunk_end)


def build_chunk_examples(
    contract_records: list[dict[str, Any]],
    clause_to_id: dict[str, int],
    tokenizer: Any,
    max_length: int = 256,
    stride: int = 64,
) -> list[dict[str, Any]]:
    chunk_examples: list[dict[str, Any]] = []
    num_labels = len(clause_to_id)

    for record in contract_records:
        contract_text = record["contract_text"]
        chunked = tokenizer(
            contract_text,
            max_length=max_length,
            stride=stride,
            truncation=True,
            padding="max_length",
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
        )

        num_chunks = len(chunked["input_ids"])

        for chunk_index in range(num_chunks):
            offset_mapping = chunked["offset_mapping"][chunk_index]
            valid_offsets = [(start, end) for start, end in offset_mapping if end > start]

            if not valid_offsets:
                continue

            chunk_char_start = valid_offsets[0][0]
            chunk_char_end = valid_offsets[-1][1]
            labels = np.zeros(num_labels, dtype=np.float32)

            for clause_name, spans in record["clause_spans"].items():
                clause_id = clause_to_id[clause_name]
                if any(_span_overlaps_chunk(span_start, span_end, chunk_char_start, chunk_char_end) for span_start, span_end in spans):
                    labels[clause_id] = 1.0

            chunk_example = {
                "contract_title": record["contract_title"],
                "chunk_index": chunk_index,
                "chunk_char_start": chunk_char_start,
                "chunk_char_end": chunk_char_end,
                "chunk_text": contract_text[chunk_char_start:chunk_char_end],
                "input_ids": chunked["input_ids"][chunk_index],
                "attention_mask": chunked["attention_mask"][chunk_index],
                "labels": labels.tolist(),
            }

            if "token_type_ids" in chunked:
                chunk_example["token_type_ids"] = chunked["token_type_ids"][chunk_index]

            chunk_examples.append(chunk_example)

    return chunk_examples


class MultiLabelChunkDataset(Dataset):
    def __init__(self, chunk_examples: list[dict[str, Any]]) -> None:
        self.chunk_examples = chunk_examples

    def __len__(self) -> int:
        return len(self.chunk_examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        example = self.chunk_examples[index]

        item = {
            "input_ids": torch.tensor(example["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(example["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(example["labels"], dtype=torch.float32),
        }

        if "token_type_ids" in example:
            item["token_type_ids"] = torch.tensor(example["token_type_ids"], dtype=torch.long)

        return item


def make_chunk_dataframe(chunk_examples: list[dict[str, Any]], id_to_clause: dict[int, str]) -> pd.DataFrame:
    rows = []

    for example in chunk_examples:
        active_labels = [
            id_to_clause[label_id]
            for label_id, is_active in enumerate(example["labels"])
            if is_active
        ]
        rows.append(
            {
                "contract_title": example["contract_title"],
                "chunk_index": example["chunk_index"],
                "chunk_char_start": example["chunk_char_start"],
                "chunk_char_end": example["chunk_char_end"],
                "chunk_text": example["chunk_text"],
                "num_active_labels": int(sum(example["labels"])),
                "active_labels": active_labels,
            }
        )

    return pd.DataFrame(rows)


def prepare_chunked_splits(
    cuad_df: pd.DataFrame,
    model_name: str = "distilbert-base-uncased",
    max_length: int = 256,
    stride: int = 64,
    seed: int = 42,
) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    clause_to_id, id_to_clause = build_clause_mappings(cuad_df)
    contract_records = build_contract_records(cuad_df)
    train_records, val_records, test_records = split_contract_records(contract_records, seed=seed)

    train_examples = build_chunk_examples(train_records, clause_to_id, tokenizer, max_length=max_length, stride=stride)
    val_examples = build_chunk_examples(val_records, clause_to_id, tokenizer, max_length=max_length, stride=stride)
    test_examples = build_chunk_examples(test_records, clause_to_id, tokenizer, max_length=max_length, stride=stride)

    return {
        "tokenizer": tokenizer,
        "clause_to_id": clause_to_id,
        "id_to_clause": id_to_clause,
        "train_records": train_records,
        "val_records": val_records,
        "test_records": test_records,
        "train_examples": train_examples,
        "val_examples": val_examples,
        "test_examples": test_examples,
        "train_dataset": MultiLabelChunkDataset(train_examples),
        "val_dataset": MultiLabelChunkDataset(val_examples),
        "test_dataset": MultiLabelChunkDataset(test_examples),
    }


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def create_model(model_name: str, id_to_clause: dict[int, str]) -> Any:
    label2id = {label_name: label_id for label_id, label_name in id_to_clause.items()}
    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(id_to_clause),
        id2label=id_to_clause,
        label2id=label2id,
        problem_type="multi_label_classification",
    )


def compute_pos_weight(chunk_examples: list[dict[str, Any]]) -> torch.Tensor:
    label_matrix = np.asarray([example["labels"] for example in chunk_examples], dtype=np.float32)
    positive_counts = label_matrix.sum(axis=0)
    negative_counts = len(label_matrix) - positive_counts
    weights = np.where(positive_counts > 0, negative_counts / np.maximum(positive_counts, 1.0), 1.0)
    return torch.tensor(weights, dtype=torch.float32)


def sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits))


def compute_multilabel_metrics(logits: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    probabilities = sigmoid(logits)
    predictions = (probabilities >= threshold).astype(int)
    labels = labels.astype(int)

    return {
        "micro_f1": f1_score(labels, predictions, average="micro", zero_division=0),
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
        "samples_f1": f1_score(labels, predictions, average="samples", zero_division=0),
        "micro_precision": precision_score(labels, predictions, average="micro", zero_division=0),
        "micro_recall": recall_score(labels, predictions, average="micro", zero_division=0),
        "subset_accuracy": accuracy_score(labels, predictions),
        "hamming_loss": hamming_loss(labels, predictions),
    }


def tune_global_threshold(
    logits: np.ndarray,
    labels: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> tuple[float, dict[str, float]]:
    if thresholds is None:
        thresholds = np.arange(0.1, 0.91, 0.05)

    best_threshold = 0.5
    best_metrics = compute_multilabel_metrics(logits, labels, threshold=best_threshold)

    for threshold in thresholds:
        metrics = compute_multilabel_metrics(logits, labels, threshold=float(threshold))
        if metrics["micro_f1"] > best_metrics["micro_f1"]:
            best_threshold = float(threshold)
            best_metrics = metrics

    return best_threshold, best_metrics


def collect_logits_and_labels(
    model: Any,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader):
            if max_batches is not None and batch_index >= max_batches:
                break
            labels = batch["labels"].cpu().numpy()
            batch_inputs = {key: value.to(device) for key, value in batch.items() if key != "labels"}
            logits = model(**batch_inputs).logits.cpu().numpy()
            all_logits.append(logits)
            all_labels.append(labels)

    if not all_logits:
        raise ValueError("No batches were collected. Increase max_batches or use a non-empty dataset.")

    return np.vstack(all_logits), np.vstack(all_labels)


@dataclass
class TrainingArtifacts:
    model: Any
    history: pd.DataFrame
    best_threshold: float
    best_val_metrics: dict[str, float]
    tokenizer: Any
    id_to_clause: dict[int, str]


def train_chunk_classifier(
    train_dataset: MultiLabelChunkDataset,
    val_dataset: MultiLabelChunkDataset,
    train_examples: list[dict[str, Any]],
    model_name: str,
    tokenizer: Any,
    id_to_clause: dict[int, str],
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
    device: torch.device | None = None,
) -> TrainingArtifacts:
    device = device or choose_device()
    model = create_model(model_name, id_to_clause).to(device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    effective_train_batches = len(train_loader)
    if max_train_batches is not None:
        effective_train_batches = min(effective_train_batches, max_train_batches)

    total_steps = max(1, epochs * max(1, effective_train_batches))
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    pos_weight = compute_pos_weight(train_examples).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_state_dict = None
    best_threshold = 0.5
    best_val_metrics = {"micro_f1": -1.0}
    history_rows = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        seen_train_batches = 0

        for batch_index, batch in enumerate(train_loader):
            if max_train_batches is not None and batch_index >= max_train_batches:
                break
            labels = batch["labels"].to(device)
            batch_inputs = {key: value.to(device) for key, value in batch.items() if key != "labels"}

            optimizer.zero_grad(set_to_none=True)
            logits = model(**batch_inputs).logits
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += float(loss.item())
            seen_train_batches += 1

        val_logits, val_labels = collect_logits_and_labels(model, val_loader, device, max_batches=max_val_batches)
        threshold, val_metrics = tune_global_threshold(val_logits, val_labels)
        average_train_loss = total_loss / max(1, seen_train_batches)

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": average_train_loss,
                "val_threshold": threshold,
                **val_metrics,
            }
        )

        if val_metrics["micro_f1"] > best_val_metrics["micro_f1"]:
            best_val_metrics = val_metrics
            best_threshold = threshold
            best_state_dict = copy.deepcopy(model.state_dict())

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return TrainingArtifacts(
        model=model,
        history=pd.DataFrame(history_rows),
        best_threshold=best_threshold,
        best_val_metrics=best_val_metrics,
        tokenizer=tokenizer,
        id_to_clause=id_to_clause,
    )


def evaluate_chunk_classifier(
    model: Any,
    dataset: MultiLabelChunkDataset,
    threshold: float,
    batch_size: int = 16,
    max_batches: int | None = None,
    device: torch.device | None = None,
) -> dict[str, float]:
    device = device or choose_device()
    model = model.to(device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    logits, labels = collect_logits_and_labels(model, dataloader, device, max_batches=max_batches)
    return compute_multilabel_metrics(logits, labels, threshold=threshold)


def predict_contract_chunks(
    contract_text: str,
    model: Any,
    tokenizer: Any,
    id_to_clause: dict[int, str],
    threshold: float = 0.5,
    max_length: int = 256,
    stride: int = 64,
    batch_size: int = 16,
    device: torch.device | None = None,
) -> pd.DataFrame:
    device = device or choose_device()
    model = model.to(device)
    model.eval()

    tokenized = tokenizer(
        contract_text,
        max_length=max_length,
        stride=stride,
        truncation=True,
        padding="max_length",
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        return_tensors="pt",
    )

    offset_mapping = tokenized.pop("offset_mapping")
    num_chunks = tokenized["input_ids"].shape[0]

    rows = []
    with torch.no_grad():
        for start_idx in range(0, num_chunks, batch_size):
            end_idx = start_idx + batch_size
            batch_inputs = {
                key: value[start_idx:end_idx].to(device)
                for key, value in tokenized.items()
            }
            batch_logits = model(**batch_inputs).logits
            batch_probs = torch.sigmoid(batch_logits).cpu().numpy()

            for relative_idx, probabilities in enumerate(batch_probs):
                chunk_idx = start_idx + relative_idx
                chunk_offsets = offset_mapping[chunk_idx].tolist()
                valid_offsets = [(start, end) for start, end in chunk_offsets if end > start]
                chunk_char_start = valid_offsets[0][0]
                chunk_char_end = valid_offsets[-1][1]
                predicted_labels = [
                    id_to_clause[label_id]
                    for label_id, score in enumerate(probabilities)
                    if score >= threshold
                ]
                label_scores = {
                    id_to_clause[label_id]: float(score)
                    for label_id, score in enumerate(probabilities)
                }
                rows.append(
                    {
                        "chunk_index": chunk_idx,
                        "chunk_char_start": chunk_char_start,
                        "chunk_char_end": chunk_char_end,
                        "chunk_text": contract_text[chunk_char_start:chunk_char_end],
                        "predicted_labels": predicted_labels,
                        "label_scores": label_scores,
                    }
                )

    return pd.DataFrame(rows)
