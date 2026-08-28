# -*- coding: utf-8 -*-
"""PyTorch tabular MLP with Apple Metal/MPS acceleration."""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
from sklearn.base import clone
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn


def torch_runtime():
    mps_built = bool(torch.backends.mps.is_built())
    mps_available = bool(torch.backends.mps.is_available())
    return {
        "available": True,
        "version": str(torch.__version__),
        "mps_built": mps_built,
        "mps_available": mps_available,
    }


def resolve_torch_device(requested="auto"):
    requested = str(requested or "auto").strip().lower()
    if requested not in {"auto", "cpu", "mps"}:
        raise ValueError("Вычислитель должен быть Auto, CPU или GPU · MPS.")
    available = bool(torch.backends.mps.is_available())
    if requested == "mps" and not available:
        raise ValueError(
            "PyTorch не обнаружил Metal/MPS. Выберите Auto или CPU."
        )
    resolved = "mps" if available and requested in {"auto", "mps"} else "cpu"
    return {
        "requested": requested,
        "resolved": "MPS" if resolved == "mps" else "CPU",
        "device": resolved,
        "mps_available": available,
        "torch_version": str(torch.__version__),
    }


def _activation(name):
    name = str(name or "relu")
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "logistic":
        return nn.Sigmoid()
    if name == "identity":
        return nn.Identity()
    raise ValueError("Выбрана неподдерживаемая функция активации.")


class _TorchMLP(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size, activation):
        super().__init__()
        layers = []
        previous = int(input_size)
        for width in hidden_layers:
            layers.extend([nn.Linear(previous, int(width)), _activation(activation)])
            previous = int(width)
        layers.append(nn.Linear(previous, int(output_size)))
        self.layers = nn.Sequential(*layers)

    def forward(self, values):
        return self.layers(values)


class TorchTabularModel:
    """Serializable fitted preprocessor + CPU-stored torch network."""

    def __init__(self, *, preprocessor, network, task, classes=None,
                 target_scaler=None, preferred_device="cpu", batch_size=8192):
        self.preprocessor = preprocessor
        self.network = network.to("cpu")
        self.task = str(task)
        self.classes_ = np.asarray([] if classes is None else classes, dtype=object)
        self.target_scaler = target_scaler
        self.preferred_device = str(preferred_device or "cpu")
        self.batch_size = max(1, int(batch_size or 8192))

    def _resolved_inference_device(self):
        if self.preferred_device == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _logits_on_device(self, values, device):
        self.network.to(device)
        self.network.eval()
        chunks = []
        with torch.inference_mode():
            for start in range(0, len(values), self.batch_size):
                batch = torch.as_tensor(
                    values[start:start + self.batch_size],
                    dtype=torch.float32, device=device,
                )
                chunks.append(self.network(batch).detach().cpu().numpy())
        if not chunks:
            output_size = len(self.classes_) if self.task == "classification" else 1
            return np.empty((0, output_size), dtype=np.float32)
        return np.concatenate(chunks, axis=0)

    def _logits(self, frame):
        values = np.asarray(self.preprocessor.transform(frame), dtype=np.float32)
        device = self._resolved_inference_device()
        try:
            return self._logits_on_device(values, device)
        except (RuntimeError, NotImplementedError):
            if device.type != "mps":
                raise
            self.network.to("cpu")
            torch.mps.empty_cache()
            return self._logits_on_device(values, torch.device("cpu"))
        finally:
            self.network.to("cpu")
            if device.type == "mps":
                torch.mps.empty_cache()

    def predict_proba(self, frame):
        if self.task != "classification":
            raise AttributeError("predict_proba доступен только для классификации.")
        logits = self._logits(frame)
        shifted = logits - logits.max(axis=1, keepdims=True)
        values = np.exp(shifted)
        return values / values.sum(axis=1, keepdims=True)

    def predict(self, frame):
        logits = self._logits(frame)
        if self.task == "classification":
            return self.classes_[np.argmax(logits, axis=1)].astype(str)
        scaled = logits.reshape(-1, 1)
        return self.target_scaler.inverse_transform(scaled).reshape(-1)


def _internal_split(y, *, task, fraction, random_seed, enabled):
    indices = np.arange(len(y))
    if not enabled:
        return indices, np.asarray([], dtype=int)
    stratify = np.asarray(y) if task == "classification" else None
    train_indices, valid_indices = train_test_split(
        indices, test_size=float(fraction), random_state=int(random_seed),
        stratify=stratify,
    )
    return np.asarray(train_indices), np.asarray(valid_indices)


def _state_on_cpu(network):
    return {
        key: value.detach().cpu().clone()
        for key, value in network.state_dict().items()
    }


def _fit_on_device(
    x_values, y_values, *, task, classes, hidden_layers, activation, solver,
    max_iter, learning_rate, alpha, batch_size, early_stopping,
    validation_fraction, patience, tolerance, class_balance, random_seed,
    device_name, progress_callback, cancel_event, progress_start, progress_span,
):
    if solver not in {"adam", "sgd"}:
        raise ValueError("PyTorch backend поддерживает оптимизаторы Adam и SGD.")
    torch.manual_seed(int(random_seed))
    device = torch.device(device_name)
    if device.type == "mps":
        torch.mps.manual_seed(int(random_seed))
    train_indices, valid_indices = _internal_split(
        y_values, task=task, fraction=validation_fraction,
        random_seed=random_seed, enabled=bool(early_stopping),
    )
    x_tensor = torch.as_tensor(x_values, dtype=torch.float32, device=device)
    if task == "classification":
        class_map = {str(value): index for index, value in enumerate(classes)}
        y_encoded = np.asarray([class_map[str(value)] for value in y_values], dtype=np.int64)
        y_tensor = torch.as_tensor(y_encoded, dtype=torch.long, device=device)
        weights = None
        if class_balance == "balanced":
            counts = np.bincount(y_encoded, minlength=len(classes)).astype(float)
            weights_array = np.ones(len(classes), dtype=np.float32)
            present = counts > 0
            weights_array[present] = len(y_encoded) / (len(classes) * counts[present])
            weights = torch.as_tensor(weights_array, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        output_size = len(classes)
        target_scaler = None
    else:
        target_scaler = StandardScaler()
        scaled = target_scaler.fit_transform(
            np.asarray(y_values, dtype=float).reshape(-1, 1)
        ).astype(np.float32).reshape(-1)
        y_tensor = torch.as_tensor(scaled, dtype=torch.float32, device=device)
        criterion = nn.MSELoss()
        output_size = 1
    network = _TorchMLP(
        x_values.shape[1], hidden_layers, output_size, activation
    ).to(device)
    if solver == "sgd":
        optimizer = torch.optim.SGD(
            network.parameters(), lr=float(learning_rate), momentum=.9,
            weight_decay=float(alpha),
        )
    else:
        optimizer = torch.optim.Adam(
            network.parameters(), lr=float(learning_rate),
            weight_decay=float(alpha),
        )
    train_index_tensor = torch.as_tensor(train_indices, dtype=torch.long, device=device)
    valid_index_tensor = torch.as_tensor(valid_indices, dtype=torch.long, device=device)
    resolved_batch = min(max(1, int(batch_size)), max(1, len(train_indices)))
    losses, validation_scores = [], []
    best_loss = float("inf")
    best_state = None
    stale_epochs = 0
    total_epochs = int(max_iter)
    report_every = max(1, total_epochs // 100)
    for epoch in range(total_epochs):
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("Обучение отменено.")
        network.train()
        permutation = train_index_tensor[
            torch.randperm(len(train_index_tensor), device=device)
        ]
        epoch_loss = 0.0
        seen = 0
        for start in range(0, len(permutation), resolved_batch):
            batch_indices = permutation[start:start + resolved_batch]
            optimizer.zero_grad(set_to_none=True)
            output = network(x_tensor[batch_indices])
            if task == "classification":
                loss = criterion(output, y_tensor[batch_indices])
            else:
                loss = criterion(output.reshape(-1), y_tensor[batch_indices])
            loss.backward()
            optimizer.step()
            size = len(batch_indices)
            epoch_loss += float(loss.detach().cpu()) * size
            seen += size
        losses.append(epoch_loss / max(1, seen))

        if len(valid_indices):
            network.eval()
            with torch.inference_mode():
                validation_output = network(x_tensor[valid_index_tensor])
                if task == "classification":
                    validation_loss = float(
                        criterion(
                            validation_output, y_tensor[valid_index_tensor]
                        ).detach().cpu()
                    )
                    validation_prediction = validation_output.argmax(dim=1)
                    score = float(
                        (validation_prediction == y_tensor[valid_index_tensor])
                        .float().mean().detach().cpu()
                    )
                else:
                    validation_loss = float(
                        criterion(
                            validation_output.reshape(-1),
                            y_tensor[valid_index_tensor],
                        ).detach().cpu()
                    )
                    predicted = validation_output.detach().cpu().numpy().reshape(-1, 1)
                    predicted = target_scaler.inverse_transform(predicted).reshape(-1)
                    score = float(r2_score(np.asarray(y_values)[valid_indices], predicted))
            validation_scores.append(score)
            if best_loss - validation_loss > float(tolerance):
                best_loss = validation_loss
                best_state = _state_on_cpu(network)
                stale_epochs = 0
            else:
                stale_epochs += 1
            if stale_epochs >= int(patience):
                break
        if progress_callback and (epoch == 0 or (epoch + 1) % report_every == 0):
            fraction = (epoch + 1) / total_epochs
            progress_callback(
                progress_start + progress_span * fraction,
                f"PyTorch · {device_name.upper()} · эпоха {epoch + 1}/{total_epochs}",
            )
    if best_state is not None:
        network.load_state_dict(best_state)
    network.to("cpu")
    if device.type == "mps":
        torch.mps.empty_cache()
    history = {
        "label": "PyTorch",
        "epochs": list(range(1, len(losses) + 1)),
        "loss": losses,
        "validation": validation_scores,
        "n_iter": len(losses),
        "early_stopping": bool(len(valid_indices)),
        "best_validation_score": max(validation_scores) if validation_scores else None,
    }
    return network, target_scaler, history


def fit_torch_tabular(
    x, y, *, preprocessor, task, classes=None, hidden_layers=(64, 32),
    activation="relu", solver="adam", max_iter=500, learning_rate=.001,
    alpha=.0001, batch_size=64, early_stopping=True,
    validation_fraction=.15, patience=30, tolerance=.0001,
    class_balance="none", random_seed=42, compute_device="auto",
    progress_callback=None, cancel_event=None, progress_start=0, progress_span=1,
):
    resolved = resolve_torch_device(compute_device)
    fitted_preprocessor = clone(preprocessor)
    x_values = np.asarray(fitted_preprocessor.fit_transform(x), dtype=np.float32)
    if x_values.ndim != 2 or x_values.shape[1] < 1:
        raise ValueError("После подготовки не осталось входных признаков.")
    y_values = np.asarray(y)
    class_labels = [str(value) for value in ([] if classes is None else classes)]

    def execute(device_name):
        return _fit_on_device(
            x_values, y_values, task=task, classes=class_labels,
            hidden_layers=hidden_layers, activation=activation, solver=solver,
            max_iter=max_iter, learning_rate=learning_rate, alpha=alpha,
            batch_size=batch_size, early_stopping=early_stopping,
            validation_fraction=validation_fraction, patience=patience,
            tolerance=tolerance, class_balance=class_balance,
            random_seed=random_seed, device_name=device_name,
            progress_callback=progress_callback, cancel_event=cancel_event,
            progress_start=progress_start, progress_span=progress_span,
        )

    try:
        network, target_scaler, history = execute(resolved["device"])
    except (RuntimeError, NotImplementedError) as error:
        if resolved["device"] != "mps" or resolved["requested"] != "auto":
            raise
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        resolved["fallback_reason"] = str(error)
        resolved["resolved"] = "CPU"
        resolved["device"] = "cpu"
        if progress_callback:
            progress_callback(progress_start, "MPS недоступен для операции · переход на CPU")
        network, target_scaler, history = execute("cpu")
    model = TorchTabularModel(
        preprocessor=fitted_preprocessor, network=network, task=task,
        classes=class_labels, target_scaler=target_scaler,
        preferred_device=resolved["device"], batch_size=max(1024, int(batch_size)),
    )
    return model, history, resolved
