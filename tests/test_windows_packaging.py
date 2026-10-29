"""Static contracts for the Windows Electron clone and CI build."""

from pathlib import Path
import json
import unittest


ROOT = Path(__file__).resolve().parent.parent


class WindowsPackagingTests(unittest.TestCase):
    def test_electron_uses_windows_venv_in_development_and_sidecar_in_build(self):
        main = (ROOT / "electron" / "main.js").read_text(encoding="utf-8")
        self.assertIn(".venv', 'Scripts', 'python.exe", main)
        self.assertIn("server', 'dataanalize-server.exe", main)
        self.assertIn("preload.js", main)

    def test_native_file_dialog_is_exposed_without_node_integration(self):
        preload = (ROOT / "electron" / "preload.js").read_text(encoding="utf-8")
        bridge = (ROOT / "assets" / "electron_file_dialog.js").read_text(encoding="utf-8")
        main = (ROOT / "electron" / "main.js").read_text(encoding="utf-8")

        self.assertIn("contextBridge.exposeInMainWorld", preload)
        self.assertIn("dataset:pick-file", preload)
        self.assertIn("ipcMain.handle('dataset:pick-file'", main)
        self.assertIn('set_props("dataset-file-drop-store"', bridge)
        self.assertIn("contextIsolation: true", main)
        self.assertIn("nodeIntegration: false", main)
        self.assertIn("DATAANALIZE_ELECTRON", main)

        callback = (ROOT / "callbacks" / "file_handling.py").read_text(encoding="utf-8")
        self.assertIn('os.environ.get("DATAANALIZE_ELECTRON")', callback)

    def test_package_builds_x64_nsis_and_zip_with_python_sidecar(self):
        package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))
        windows = package["build"]["win"]
        targets = {target["target"] for target in windows["target"]}

        self.assertEqual(targets, {"nsis", "zip"})
        self.assertTrue(all(target["arch"] == ["x64"] for target in windows["target"]))
        self.assertEqual(
            windows["extraResources"][0]["from"],
            "python-dist/dataanalize-server",
        )
        self.assertIn("--win", package["scripts"]["build:win"])

    def test_github_action_builds_server_tests_and_electron_artifacts(self):
        workflow = (
            ROOT / ".github" / "workflows" / "build-windows.yml"
        ).read_text(encoding="utf-8")

        self.assertIn("runs-on: windows-latest", workflow)
        self.assertIn("unittest discover", workflow)
        self.assertIn("build_windows_server.py", workflow)
        self.assertIn("npm run build:win", workflow)
        self.assertIn("actions/upload-artifact@v4", workflow)
        self.assertIn("workflow_dispatch", workflow)
        self.assertNotIn("push:", workflow)

    def test_frozen_server_enables_windows_multiprocessing_support(self):
        server = (ROOT / "run_server.py").read_text(encoding="utf-8")
        self.assertLess(server.index("multiprocessing.freeze_support()"), server.index("import app"))

    def test_windows_debug_script_uses_native_venv_and_electron(self):
        script = (ROOT / "scripts" / "dev_windows.ps1").read_text(encoding="utf-8")

        self.assertIn(".venv\\Scripts\\python.exe", script)
        self.assertIn("py -3.14 -m venv .venv", script)
        self.assertIn("npm run dev", script)


if __name__ == "__main__":
    unittest.main()
