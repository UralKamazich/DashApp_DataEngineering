const { app, BrowserWindow, dialog, ipcMain, session } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const http = require('http');

let mainWindow;
let pythonProcess;

const DASH_URL = 'http://127.0.0.1:8090';
const APP_ROOT = app.isPackaged
    ? path.join(process.resourcesPath, 'app')
    : path.join(__dirname, '..');
const APP_ICON = path.join(APP_ROOT, 'assets', 'icon.png');

function pythonLaunch() {
    if (app.isPackaged && process.platform === 'win32') {
        return {
            command: path.join(process.resourcesPath, 'server', 'dataanalize-server.exe'),
            args: [],
            cwd: path.join(process.resourcesPath, 'app'),
        };
    }
    const venvPython = process.platform === 'win32'
        ? path.join(APP_ROOT, '.venv', 'Scripts', 'python.exe')
        : path.join(APP_ROOT, '.venv', 'bin', 'python');
    return {
        command: venvPython,
        args: [path.join(APP_ROOT, 'run_server.py')],
        cwd: APP_ROOT,
    };
}

// Проверка готовности сервера
function waitForServer(url, callback, retries = 30) {
    http.get(url, (res) => {
        if (res.statusCode === 200) {
            callback(true);
        } else if (retries > 0) {
            setTimeout(() => waitForServer(url, callback, retries - 1), 1000);
        } else {
            callback(false);
        }
    }).on('error', () => {
        if (retries > 0) {
            setTimeout(() => waitForServer(url, callback, retries - 1), 1000);
        } else {
            callback(false);
        }
    });
}

// Запуск Python Dash сервера
function startPythonServer() {
    // Development uses the platform-specific venv. Packaged Windows uses a
    // self-contained PyInstaller sidecar; packaged macOS keeps its venv in
    // Contents/Resources/app. None of these runtimes live inside app.asar.
    const launch = pythonLaunch();
    
    console.log('Starting Dash server...');
    
    pythonProcess = spawn(launch.command, launch.args, {
        cwd: launch.cwd,
        env: {
            ...process.env,
            PYTHONUNBUFFERED: '1',
            DATAANALIZE_ELECTRON: '1',
        },
        stdio: ['ignore', 'pipe', 'pipe']
    });
    
    pythonProcess.stdout.on('data', (data) => {
        console.log(`[Dash] ${data.toString().trim()}`);
    });
    
    pythonProcess.stderr.on('data', (data) => {
        console.error(`[Dash] ${data.toString().trim()}`);
    });
    
    pythonProcess.on('close', (code) => {
        console.log(`Python process exited with code ${code}`);
    });
    
    pythonProcess.on('error', (err) => {
        console.error('Failed to start Python process:', err);
        dialog.showErrorBox('Ошибка запуска', `Не удалось запустить сервер:\n${err.message}`);
    });
}

// Создание главного окна
function createWindow() {
    // Очистка кэша для загрузки свежих assets
    session.defaultSession.clearCache();

    mainWindow = new BrowserWindow({
        width: 1400,
        height: 900,
        minWidth: 800,
        minHeight: 600,
        title: `DataAnalize v${app.getVersion()}`,
        icon: APP_ICON,
        backgroundColor: '#1a1b1e',
        webPreferences: {
            nodeIntegration: false,
            contextIsolation: true,
            preload: path.join(__dirname, 'preload.js'),
        },
    });
    
    mainWindow.loadURL(DASH_URL);
    
    mainWindow.on('closed', () => {
        mainWindow = null;
    });
}

app.whenReady().then(() => {
    if (process.platform === 'darwin') {
        app.dock.setIcon(APP_ICON);
    }

    startPythonServer();
    
    waitForServer(DASH_URL, (ready) => {
        if (ready) {
            console.log('Server ready, creating window...');
            createWindow();
        } else {
            dialog.showErrorBox('Ошибка', 'Не удалось запустить сервер');
            app.quit();
        }
    });
    
    app.on('activate', () => {
        if (BrowserWindow.getAllWindows().length === 0) {
            createWindow();
        }
    });
});

ipcMain.handle('dataset:pick-file', async () => {
    const result = await dialog.showOpenDialog(mainWindow, {
        title: 'Выберите файл данных',
        properties: ['openFile'],
        filters: [
            { name: 'Поддерживаемые datasets', extensions: ['xlsx', 'csv', 'txt', 'tsv', 'zip', 'pkl'] },
            { name: 'Excel', extensions: ['xlsx'] },
            { name: 'Текстовые таблицы', extensions: ['csv', 'txt', 'tsv'] },
            { name: 'Архивы ZIP', extensions: ['zip'] },
            { name: 'Pickle', extensions: ['pkl'] },
            { name: 'Все файлы', extensions: ['*'] },
        ],
    });
    if (result.canceled || !result.filePaths.length) return null;
    const filePath = result.filePaths[0];
    return { path: filePath, name: path.basename(filePath) };
});

app.on('window-all-closed', () => {
    if (pythonProcess) pythonProcess.kill();
    if (process.platform !== 'darwin') app.quit();
});

app.on('before-quit', () => {
    if (pythonProcess) pythonProcess.kill();
});
