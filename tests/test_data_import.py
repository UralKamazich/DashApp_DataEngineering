import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd

from data_import import (
    POPULAR_DATASETS,
    inspect_archive,
    list_archive_tables,
    read_archive_table,
    read_delimited_dataset,
    source_name_from_location,
    validate_remote_url,
)


class DataImportTests(unittest.TestCase):
    def _write(self, folder, name, payload):
        path = Path(folder) / name
        path.write_bytes(payload)
        return path

    def test_utf8_csv(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self._write(
                folder, "sample.csv", b"name,value\nalpha,1.5\nbeta,2.5\n"
            )
            frame, info = read_delimited_dataset(path, path.name)
        self.assertEqual(frame.shape, (2, 2))
        self.assertEqual(frame["value"].tolist(), [1.5, 2.5])
        self.assertEqual(info.delimiter, ",")
        self.assertEqual(info.encoding, "utf-8")

    def test_cp1251_semicolon_and_decimal_comma(self):
        payload = "Имя;Значение\nАльфа;1,5\nБета;2,5\n".encode("cp1251")
        with tempfile.TemporaryDirectory() as folder:
            path = self._write(folder, "russian.csv", payload)
            frame, info = read_delimited_dataset(path, path.name)
        self.assertEqual(frame["Имя"].tolist(), ["Альфа", "Бета"])
        self.assertEqual(frame["Значение"].tolist(), [1.5, 2.5])
        self.assertEqual(info.encoding, "cp1251")
        self.assertEqual(info.decimal, ",")

    def test_tab_and_whitespace_text_tables(self):
        with tempfile.TemporaryDirectory() as folder:
            tab = self._write(folder, "tab.txt", b"x\ty\n1\t2\n3\t4\n")
            space = self._write(folder, "space.txt", b"x y z\n1 2 3\n4 5 6\n")
            tab_frame, tab_info = read_delimited_dataset(tab, tab.name)
            space_frame, space_info = read_delimited_dataset(space, space.name)
        self.assertEqual(tab_info.delimiter, "\t")
        self.assertEqual(tab_frame.shape, (2, 2))
        self.assertEqual(space_info.delimiter, r"\s+")
        self.assertEqual(space_frame.shape, (2, 3))

    def test_plain_text_can_remain_one_column(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self._write(folder, "labels.txt", b"label\nalpha\nbeta\n")
            frame, info = read_delimited_dataset(path, path.name)
        self.assertEqual(frame.columns.tolist(), ["label"])
        self.assertEqual(frame["label"].tolist(), ["alpha", "beta"])
        self.assertEqual(info.delimiter, "\0")

    def test_remote_csv_uses_same_parser(self):
        payload = b"feature,target\n1,yes\n2,no\n"
        with patch("data_import._remote_bytes", return_value=payload):
            frame, info = read_delimited_dataset(
                "https://example.test/sample.csv", "sample.csv"
            )
        self.assertEqual(frame.shape, (2, 2))
        self.assertTrue(info.remote)

    def test_catalog_and_url_validation(self):
        self.assertGreaterEqual(len(POPULAR_DATASETS), 5)
        self.assertEqual(
            source_name_from_location("https://example.test/data/iris.csv?raw=1"),
            "iris.csv",
        )
        self.assertEqual(
            validate_remote_url("https://example.test/data.csv"),
            "https://example.test/data.csv",
        )
        with self.assertRaisesRegex(ValueError, "http"):
            validate_remote_url("file:///tmp/private.csv")

    def test_zip_with_one_delimited_table(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "dataset.zip"
            with ZipFile(path, "w", ZIP_DEFLATED) as archive:
                archive.writestr(
                    "data/russian.csv",
                    "Имя;Значение\nАльфа;1,5\nБета;2,5\n".encode("cp1251"),
                )
            tables = list_archive_tables(path)
            frame, info = read_archive_table(path, tables[0]["name"], path.name)
            inspected, auto_frame, auto_info = inspect_archive(path, path.name)
        self.assertEqual([item["name"] for item in tables], ["data/russian.csv"])
        self.assertEqual(frame["Значение"].tolist(), [1.5, 2.5])
        self.assertEqual(info.format, "ZIP/CSV")
        self.assertEqual(info.archive_member, "data/russian.csv")
        self.assertEqual(inspected, tables)
        self.assertEqual(auto_frame.shape, frame.shape)
        self.assertEqual(auto_info.format, "ZIP/CSV")

    def test_zip_lists_multiple_tables_and_ignores_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "multiple.zip"
            with ZipFile(path, "w", ZIP_DEFLATED) as archive:
                archive.writestr("first.csv", "x\n1\n")
                archive.writestr("nested/second.tsv", "y\tlabel\n2\tb\n")
                archive.writestr("__MACOSX/._first.csv", "metadata")
                archive.writestr("readme.md", "notes")
            tables = list_archive_tables(path)
        self.assertEqual(
            [item["name"] for item in tables],
            ["first.csv", "nested/second.tsv"],
        )

    def test_xlsx_inside_zip(self):
        excel = BytesIO()
        pd.DataFrame({"x": [1, 2], "label": ["a", "b"]}).to_excel(
            excel, index=False, engine="openpyxl"
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "excel.zip"
            with ZipFile(path, "w", ZIP_DEFLATED) as archive:
                archive.writestr("table.xlsx", excel.getvalue())
            frame, info = read_archive_table(path, "table.xlsx", path.name)
        self.assertEqual(frame.shape, (2, 2))
        self.assertEqual(info.format, "ZIP/XLSX")

    def test_remote_zip_and_unsafe_member(self):
        safe_buffer = BytesIO()
        with ZipFile(safe_buffer, "w", ZIP_DEFLATED) as archive:
            archive.writestr("remote.csv", "x,y\n1,2\n")
        with patch("data_import._remote_bytes", return_value=safe_buffer.getvalue()):
            tables = list_archive_tables("https://example.test/data.zip")
            frame, info = read_archive_table(
                "https://example.test/data.zip", tables[0]["name"]
            )
        self.assertEqual(frame.shape, (1, 2))
        self.assertTrue(info.remote)
        with patch(
            "data_import._remote_bytes", return_value=safe_buffer.getvalue()
        ) as remote_read:
            _, auto_frame, _ = inspect_archive("https://example.test/data.zip")
        self.assertEqual(auto_frame.shape, (1, 2))
        remote_read.assert_called_once()

        with tempfile.TemporaryDirectory() as folder:
            unsafe = Path(folder) / "unsafe.zip"
            with ZipFile(unsafe, "w") as archive:
                archive.writestr("../outside.csv", "x\n1\n")
            with self.assertRaisesRegex(ValueError, "небезопасный"):
                list_archive_tables(unsafe)


if __name__ == "__main__":
    unittest.main()


class StoreFrameCacheTests(unittest.TestCase):
    def test_cached_read_returns_independent_copies(self):
        from utils import meta_from_df, read_df_from_store

        frame = pd.DataFrame({"code": ["001", "002"], "value": [1.5, 2.5]})
        frame["code"] = frame["code"].astype("string")
        meta = meta_from_df(frame)
        payload = frame.to_json(date_format="iso", orient="split")

        first = read_df_from_store(payload, meta)
        first.loc[0, "value"] = 999.0
        second = read_df_from_store(payload, meta)

        self.assertEqual(second.loc[0, "value"], 1.5)
        self.assertEqual(list(second["code"]), ["001", "002"])


class ServerFrameStoreTests(unittest.TestCase):
    def test_reference_round_trip_keeps_dtypes_and_isolates_copies(self):
        from utils import frame_to_store, read_df_from_store, store_row_count

        frame = pd.DataFrame({
            "code": pd.array(["001", "002", "003"], dtype="string"),
            "when": pd.to_datetime(["2026-01-02", "2026-03-04", "2026-05-06"]),
            "value": [1.5, 2.5, 3.5],
        })
        reference = frame_to_store(frame)
        self.assertLess(len(reference), 64)
        self.assertEqual(store_row_count(reference), 3)

        first = read_df_from_store(reference)
        first.loc[0, "value"] = -1.0
        frame.loc[1, "value"] = -2.0
        second = read_df_from_store(reference)

        pd.testing.assert_frame_equal(
            second,
            pd.DataFrame({
                "code": pd.array(["001", "002", "003"], dtype="string"),
                "when": pd.to_datetime(["2026-01-02", "2026-03-04", "2026-05-06"]),
                "value": [1.5, 2.5, 3.5],
            }),
        )

    def test_unknown_reference_returns_empty_frame(self):
        from utils import read_df_from_store, store_row_count

        self.assertTrue(read_df_from_store("dfref:missing").empty)
        self.assertEqual(store_row_count("dfref:missing"), 0)
