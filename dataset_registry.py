# -*- coding: utf-8 -*-
"""Pure helpers for source, working and derived dataset records."""

from __future__ import annotations

from copy import deepcopy
from uuid import uuid4


SOURCE_DATASET_ID = "source"
_PAYLOAD_CACHE: dict[str, str] = {}


def _store_payload(payload):
    if payload is None:
        return None
    reference = uuid4().hex
    _PAYLOAD_CACHE[reference] = payload
    return reference


def _replace_payload(record, reference_key, payload):
    old_reference = record.get(reference_key)
    if old_reference and _PAYLOAD_CACHE.get(old_reference) == payload:
        return old_reference
    new_reference = _store_payload(payload)
    shared_elsewhere = old_reference in {
        value
        for key, value in record.items()
        if key.endswith("_ref") and key != reference_key
    }
    if old_reference:
        if not shared_elsewhere:
            _PAYLOAD_CACHE.pop(old_reference, None)
    record[reference_key] = new_reference
    return new_reference


def payload_for_record(record, *, filtered=False):
    if not record:
        return None
    reference_key = "filtered_ref" if filtered else "data_ref"
    payload = _PAYLOAD_CACHE.get(record.get(reference_key))
    if filtered and payload is None:
        payload = _PAYLOAD_CACHE.get(record.get("data_ref"))
    return payload


def _empty_filter_state():
    return {
        "filters_state": {},
        "filters_applied_state": {},
        "filter_logic": "and",
        "filter_applied_logic": "and",
    }


def create_source_registry(payload, meta, source_name=None):
    """Create a fresh registry for a newly loaded file."""
    if not payload:
        return {}
    _PAYLOAD_CACHE.clear()
    data_reference = _store_payload(payload)
    record = {
        "id": SOURCE_DATASET_ID,
        "name": source_name or "Исходный",
        "kind": "source",
        "parent_id": None,
        "data_ref": data_reference,
        "filtered_ref": data_reference,
        "meta": dict(meta or {}),
        "steps": [],
        **_empty_filter_state(),
    }
    return {SOURCE_DATASET_ID: record}


def dataset_options(registry):
    return [
        {"label": str(record.get("name") or dataset_id), "value": dataset_id}
        for dataset_id, record in (registry or {}).items()
    ]


def summarize_transformation_steps(steps):
    """Return short human-readable descriptions of committed transformations."""
    summaries = []
    for step in steps or []:
        step = step or {}
        operation = step.get("type") or step.get("operation")
        params = step.get("params") or {}
        inputs = [str(value) for value in (step.get("inputs") or [])]
        outputs = [str(value) for value in (step.get("outputs") or [])]

        if operation == "binning":
            source = inputs[0] if inputs else "канал"
            target = outputs[0] if outputs else "новый канал"
            groups = params.get("groups")
            detail = f" ({groups} групп)" if groups else ""
            summary = f"Биннинг: {source} → {target}{detail}"
        elif operation == "text_copy":
            source = ", ".join(inputs[:2]) or "каналы"
            if len(inputs) > 2:
                source += f" +{len(inputs) - 2}"
            target = ", ".join(outputs[:2]) or "текстовые копии"
            if len(outputs) > 2:
                target += f" +{len(outputs) - 2}"
            summary = f"Текст: {source} → {target}"
        elif operation == "group_aggregates":
            keys = [str(value) for value in (params.get("keys") or [])]
            metrics = [str(value) for value in (params.get("metrics") or [])]
            key_text = ", ".join(keys[:2]) or "группам"
            metric_text = ", ".join(metrics[:3]) or "метрики"
            summary = f"Агрегация: по {key_text} · {metric_text}"
        elif operation == "clustering":
            algorithm = str(params.get("algorithm") or "kmeans")
            clusters = params.get("k")
            target = outputs[0] if outputs else "кластер"
            feature_text = ", ".join(inputs[:3]) or "признаки"
            if len(inputs) > 3:
                feature_text += f" +{len(inputs) - 3}"
            k_text = f" · K={clusters}" if clusters else ""
            summary = (
                f"Кластеризация: {algorithm}{k_text} · {feature_text} → {target}"
            )
        else:
            summary = str(step.get("label") or "Преобразование")

        if step.get("scope") == "filtered":
            summary += " · после фильтров"
        summaries.append(summary)
    return summaries


def get_record(registry, dataset_id):
    return (registry or {}).get(str(dataset_id or ""))


def input_payload(registry, dataset_id, scope="base", active_id=None, active_filtered=None):
    record = get_record(registry, dataset_id)
    if not record:
        return None, None
    if scope == "filtered":
        payload = (
            active_filtered
            if str(dataset_id) == str(active_id) and active_filtered
            else payload_for_record(record, filtered=True)
        )
    else:
        payload = payload_for_record(record)
    return payload, record.get("meta") or {}


def _next_dataset_id(registry):
    used = set((registry or {}).keys())
    index = 1
    while f"dataset-{index}" in used:
        index += 1
    return f"dataset-{index}"


def suggest_dataset_name(registry, queued_steps, scope="base"):
    """Build a compact editable name from the queued pipeline and its input layer."""
    method_labels = {
        "binning": "Биннинг",
        "text_copy": "Текст",
        "group_aggregates": "Агрегат",
        "clustering": "Кластеризация",
    }
    methods = []
    for step in queued_steps or []:
        label = method_labels.get((step or {}).get("operation"), "Операция")
        if label not in methods:
            methods.append(label)
    if not methods:
        methods.append("Результат")

    scope_label = "После фильтров" if scope == "filtered" else "До фильтров"
    derived_count = sum(
        1
        for dataset_id, record in (registry or {}).items()
        if dataset_id != SOURCE_DATASET_ID and record.get("kind") == "derived"
    )
    return f"{'_'.join(methods)}_{scope_label}_{derived_count + 1}"


def commit_result(
    registry,
    input_id,
    payload,
    meta,
    step,
    output_mode="current",
    output_name=None,
):
    """Store one operation or an atomic list of operations."""
    updated = deepcopy(registry or {})
    parent = get_record(updated, input_id)
    if not parent:
        raise KeyError(f"Unknown dataset: {input_id}")

    committed_steps = (
        [dict(item or {}) for item in step]
        if isinstance(step, (list, tuple))
        else [dict(step or {})]
    )

    if output_mode == "new":
        dataset_id = _next_dataset_id(updated)
        default_name = f"Dataset {len(updated)}"
        updated[dataset_id] = {
            "id": dataset_id,
            "name": (output_name or "").strip() or default_name,
            "kind": "derived",
            "parent_id": str(input_id),
            "data_ref": _store_payload(payload),
            "filtered_ref": None,
            "meta": dict(meta or {}),
            "steps": [*(parent.get("steps") or []), *committed_steps],
            **_empty_filter_state(),
        }
        updated[dataset_id]["filtered_ref"] = updated[dataset_id]["data_ref"]
        return updated, dataset_id

    dataset_id = str(input_id)
    record = updated[dataset_id]
    old_data_reference = record.get("data_ref")
    old_filtered_reference = record.get("filtered_ref")
    data_reference = _store_payload(payload)
    record["data_ref"] = data_reference
    record["filtered_ref"] = data_reference
    for reference in {old_data_reference, old_filtered_reference}:
        if reference and reference != data_reference:
            _PAYLOAD_CACHE.pop(reference, None)
    record["meta"] = dict(meta or {})
    record["steps"] = [*(record.get("steps") or []), *committed_steps]
    return updated, dataset_id


def save_runtime_state(
    registry,
    dataset_id,
    *,
    filtered_data=None,
    filters_state=None,
    filters_applied_state=None,
    filter_logic=None,
    filter_applied_logic=None,
):
    updated = deepcopy(registry or {})
    record = get_record(updated, dataset_id)
    if not record:
        return updated
    if filtered_data is not None:
        _replace_payload(record, "filtered_ref", filtered_data)
    if filters_state is not None:
        record["filters_state"] = dict(filters_state or {})
    if filters_applied_state is not None:
        record["filters_applied_state"] = dict(filters_applied_state or {})
    if filter_logic is not None:
        record["filter_logic"] = filter_logic or "and"
    if filter_applied_logic is not None:
        record["filter_applied_logic"] = filter_applied_logic or "and"
    return updated
