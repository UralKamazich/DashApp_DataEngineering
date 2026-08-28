# -*- coding: utf-8 -*-
"""Delimited/ZIP local and remote dataset import plus sample catalog."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from contextlib import contextmanager
import csv
import re
from zipfile import BadZipFile, ZipFile
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen

import pandas as pd
from charset_normalizer import from_bytes


MAX_REMOTE_BYTES = 512 * 1024 * 1024
MAX_ARCHIVE_MEMBER_BYTES = 1024 * 1024 * 1024
MAX_ARCHIVE_TOTAL_BYTES = 2 * 1024 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 5000
MAX_ARCHIVE_TABLES = 200
MAX_COMPRESSION_RATIO = 250
SAMPLE_BYTES = 256 * 1024
DELIMITED_EXTENSIONS = {".csv", ".txt", ".tsv"}
ARCHIVE_TABLE_EXTENSIONS = DELIMITED_EXTENSIONS | {".xlsx"}

_SEABORN_RAW = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master"

POPULAR_DATASETS = [
    {
        "label": "Iris · 150 × 5",
        "value": f"{_SEABORN_RAW}/iris.csv",
        "name": "iris.csv",
        "description": "Ирисы Фишера: размеры чашелистиков и лепестков, 3 вида. Классификация.",
    },
    {
        "label": "Penguins · 344 × 7",
        "value": f"{_SEABORN_RAW}/penguins.csv",
        "name": "penguins.csv",
        "description": "Пингвины Палмера: вид, остров, размеры тела и пол. Классификация и визуальный анализ.",
    },
    {
        "label": "Titanic · 891 × 15",
        "value": f"{_SEABORN_RAW}/titanic.csv",
        "name": "titanic.csv",
        "description": "Пассажиры Titanic: выживание, класс, пол, возраст и тариф. Классификация.",
    },
    {
        "label": "Tips · 244 × 7",
        "value": f"{_SEABORN_RAW}/tips.csv",
        "name": "tips.csv",
        "description": "Счета и чаевые ресторана. Компактный набор для регрессии и графиков.",
    },
    {
        "label": "Flights · 144 × 3",
        "value": f"{_SEABORN_RAW}/flights.csv",
        "name": "flights.csv",
        "description": "Помесячное число авиапассажиров за 1949–1960 годы. Временные ряды.",
    },
    {
        "label": "Diamonds · 53 940 × 10",
        "value": f"{_SEABORN_RAW}/diamonds.csv",
        "name": "diamonds.csv",
        "description": "Характеристики и цена бриллиантов. Регрессия на более крупном наборе.",
    },
    {
        "label": "Power consumption · 2 075 259 × 9 · ZIP",
        "value": (
            "https://archive.ics.uci.edu/static/public/235/"
            "individual%2Bhousehold%2Belectric%2Bpower%2Bconsumption.zip"
        ),
        "name": "individual_household_power_consumption.zip",
        "description": (
            "UCI: энергопотребление по минутам почти за 4 года. "
            "19,7 МБ ZIP → 126,8 МБ TXT; загрузка может занять время."
        ),
    },
]


@dataclass(frozen=True)
class DelimitedImportInfo:
    format: str
    encoding: str | None
    delimiter: str | None
    decimal: str | None
    remote: bool
    archive: str | None = None
    archive_member: str | None = None
    compressed_bytes: int | None = None
    uncompressed_bytes: int | None = None

    def as_meta(self):
        return asdict(self)


def is_remote_source(value):
    return urlparse(str(value or "")).scheme.lower() in {"http", "https"}


def validate_remote_url(value):
    text = str(value or "").strip()
    parsed = urlparse(text)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Нужна прямая ссылка http:// или https:// на dataset.")
    return text


def source_name_from_location(value, fallback="dataset.csv"):
    text = str(value or "").strip()
    parsed = urlparse(text)
    raw_name = Path(unquote(parsed.path if parsed.scheme else text)).name
    return raw_name or fallback


def popular_dataset_by_url(value):
    return next(
        (item for item in POPULAR_DATASETS if item["value"] == value),
        None,
    )


def _remote_bytes(url):
    request = Request(
        validate_remote_url(url),
        headers={"User-Agent": "DataAnalize/2.0 dataset importer"},
    )
    with urlopen(request, timeout=45) as response:
        declared = response.headers.get("Content-Length")
        try:
            declared_size = int(declared) if declared else 0
        except (TypeError, ValueError):
            declared_size = 0
        if declared_size > MAX_REMOTE_BYTES:
            raise ValueError("Удалённый dataset больше 512 МБ.")
        content_type = str(response.headers.get("Content-Type") or "").lower()
        payload = response.read(MAX_REMOTE_BYTES + 1)
    if len(payload) > MAX_REMOTE_BYTES:
        raise ValueError("Удалённый dataset больше 512 МБ.")
    if "text/html" in content_type or re.match(
        rb"\s*(?:<!doctype\s+html|<html)", payload[:2048], re.IGNORECASE
    ):
        raise ValueError(
            "Ссылка открывает HTML-страницу. Используйте прямую Raw/Download "
            "ссылку на CSV или TXT."
        )
    return payload


@contextmanager
def _open_zip(source):
    try:
        if is_remote_source(source):
            buffer = BytesIO(_remote_bytes(source))
            with ZipFile(buffer) as archive:
                yield archive
        else:
            with ZipFile(Path(str(source)).expanduser()) as archive:
                yield archive
    except BadZipFile as error:
        raise ValueError("Файл повреждён или не является ZIP-архивом.") from error


def _safe_archive_members(archive):
    members = archive.infolist()
    if len(members) > MAX_ARCHIVE_MEMBERS:
        raise ValueError(f"В ZIP больше {MAX_ARCHIVE_MEMBERS} файлов.")
    total_size = 0
    tables = []
    for member in members:
        if member.is_dir():
            continue
        normalized = str(member.filename or "").replace("\\", "/")
        parts = [part for part in normalized.split("/") if part not in {"", "."}]
        if normalized.startswith("/") or ".." in parts:
            raise ValueError("ZIP содержит небезопасный путь к файлу.")
        if member.flag_bits & 0x1:
            raise ValueError("ZIP с зашифрованными файлами не поддерживается.")
        total_size += int(member.file_size)
        if total_size > MAX_ARCHIVE_TOTAL_BYTES:
            raise ValueError("Распакованный ZIP больше 2 ГБ.")
        if member.file_size > MAX_ARCHIVE_MEMBER_BYTES:
            raise ValueError("Таблица внутри ZIP больше 1 ГБ.")
        ratio = member.file_size / max(1, member.compress_size)
        if member.file_size > 10 * 1024 * 1024 and ratio > MAX_COMPRESSION_RATIO:
            raise ValueError("ZIP имеет подозрительно высокий коэффициент сжатия.")
        if (
            Path(normalized).suffix.lower() in ARCHIVE_TABLE_EXTENSIONS
            and not normalized.startswith("__MACOSX/")
            and not Path(normalized).name.startswith(".")
        ):
            tables.append({
                "name": normalized,
                "size": int(member.file_size),
                "compressed_size": int(member.compress_size),
            })
    if len(tables) > MAX_ARCHIVE_TABLES:
        raise ValueError(f"В ZIP больше {MAX_ARCHIVE_TABLES} табличных файлов.")
    if not tables:
        raise ValueError("В ZIP не найдено CSV, TXT, TSV или XLSX.")
    return tables


def list_archive_tables(source):
    """Return safe supported table members without extracting the archive."""
    with _open_zip(source) as archive:
        return _safe_archive_members(archive)


def _encoding(sample):
    if sample.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    if sample.startswith((b"\xff\xfe", b"\xfe\xff")):
        return "utf-16"
    try:
        sample.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        pass
    match = from_bytes(sample).best()
    if match and match.encoding:
        candidate = str(match.encoding).lower()
        aliases = {
            "windows-1251": "cp1251", "windows-1252": "cp1252",
            "mac-cyrillic": "mac_cyrillic",
        }
        return aliases.get(candidate, candidate)
    return "cp1251"


def _delimiter(text, extension):
    sample_lines = [line for line in text.splitlines()[:80] if line.strip()]
    sample = "\n".join(sample_lines)
    if not sample:
        raise ValueError("Файл не содержит данных.")
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        pass

    candidates = ["\t", ";", ",", "|"]
    counts = {item: [line.count(item) for line in sample_lines] for item in candidates}
    stable = [
        (sum(values), candidate)
        for candidate, values in counts.items()
        if values and min(values) > 0 and len(set(values)) <= 2
    ]
    if stable:
        return max(stable)[1]
    if extension in {".txt", ".tsv"}:
        fields = [len(re.split(r"\s+", line.strip())) for line in sample_lines]
        if fields and min(fields) > 1 and len(set(fields)) <= 2:
            return r"\s+"
    return "\0"


def _decimal(text, delimiter):
    if delimiter == ",":
        return "."
    comma = len(re.findall(r"(?<!\d)\-?\d+,\d+(?!\d)", text))
    point = len(re.findall(r"(?<!\d)\-?\d+\.\d+(?!\d)", text))
    return "," if comma > point else "."


def _read_delimited_input(input_value, sample, name, *, remote):
    extension = Path(name).suffix.lower()
    if extension not in DELIMITED_EXTENSIONS:
        raise ValueError("Для табличного текста поддерживаются .csv, .txt и .tsv.")
    encoding = _encoding(sample)
    try:
        sample_text = sample.decode(encoding, errors="strict")
    except (LookupError, UnicodeDecodeError):
        encoding = "cp1251"
        sample_text = sample.decode(encoding, errors="replace")
    delimiter = _delimiter(sample_text, extension)
    decimal = _decimal(sample_text, delimiter)
    engine = "python" if delimiter in {r"\s+", "\0"} else "c"

    frame = pd.read_csv(
        input_value,
        sep=delimiter,
        encoding=encoding,
        decimal=decimal,
        engine=engine,
        na_values=["?", "NA", "N/A", "null", "NULL"],
    )
    if frame.columns.empty:
        raise ValueError("Не удалось определить столбцы dataset.")
    info = DelimitedImportInfo(
        format=extension.lstrip(".").upper(),
        encoding=encoding,
        delimiter=delimiter,
        decimal=decimal,
        remote=remote,
    )
    return frame, info


def read_delimited_dataset(source, source_name=None):
    """Read CSV/TXT/TSV with inferred encoding, delimiter and decimal mark."""
    name = source_name or source_name_from_location(source)
    remote = is_remote_source(source)
    if remote:
        payload = _remote_bytes(source)
        return _read_delimited_input(
            BytesIO(payload), payload[:SAMPLE_BYTES], name, remote=True
        )
    path = Path(str(source)).expanduser()
    with path.open("rb") as stream:
        sample = stream.read(SAMPLE_BYTES)
    return _read_delimited_input(path, sample, name, remote=False)


def _archive_payload_to_frame(
    payload, selected, source, member_name, archive_label,
):
    extension = Path(member_name).suffix.lower()
    if extension in DELIMITED_EXTENSIONS:
        frame, info = _read_delimited_input(
            BytesIO(payload), payload[:SAMPLE_BYTES], member_name,
            remote=is_remote_source(source),
        )
        values = info.as_meta()
        values.update({
            "format": f"ZIP/{info.format}",
            "archive": archive_label,
            "archive_member": member_name,
            "compressed_bytes": selected["compressed_size"],
            "uncompressed_bytes": selected["size"],
        })
        return frame, DelimitedImportInfo(**values)
    if extension == ".xlsx":
        frame = pd.read_excel(BytesIO(payload), engine="openpyxl")
        info = DelimitedImportInfo(
            format="ZIP/XLSX", encoding=None, delimiter=None, decimal=None,
            remote=is_remote_source(source), archive=archive_label,
            archive_member=member_name,
            compressed_bytes=selected["compressed_size"],
            uncompressed_bytes=selected["size"],
        )
        return frame, info
    raise ValueError("Выбранный файл внутри ZIP не поддерживается.")


def inspect_archive(source, archive_name=None):
    """Inspect a ZIP and read its table immediately when it is the only one."""
    archive_label = archive_name or source_name_from_location(source, "dataset.zip")
    with _open_zip(source) as archive:
        tables = _safe_archive_members(archive)
        if len(tables) != 1:
            return tables, None, None
        selected = tables[0]
        payload = archive.read(selected["name"])
    frame, info = _archive_payload_to_frame(
        payload, selected, source, selected["name"], archive_label
    )
    return tables, frame, info


def read_archive_table(source, member_name, archive_name=None):
    """Read one validated table member from a local or remote ZIP."""
    archive_label = archive_name or source_name_from_location(source, "dataset.zip")
    with _open_zip(source) as archive:
        tables = _safe_archive_members(archive)
        selected = next((item for item in tables if item["name"] == member_name), None)
        if not selected:
            raise ValueError("Выбранная таблица больше недоступна внутри ZIP.")
        payload = archive.read(member_name)
    return _archive_payload_to_frame(
        payload, selected, source, member_name, archive_label
    )
