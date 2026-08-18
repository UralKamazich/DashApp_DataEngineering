const { app, BrowserWindow, dialog, session } = require('electron');
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
    const scriptPath = path.join(__dirname, '..', 'run_server.py');
    const venvPython = path.join(__dirname, '..', '.venv', 'bin', 'python');
    
    console.log('Starting Dash server...');
    
    pythonProcess = spawn(venvPython, [scriptPath], {
        cwd: path.join(__dirname, '..'),
        env: { ...process.env, PYTHONUNBUFFERED: '1' },
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

app.on('window-all-closed', () => {
    if (pythonProcess) pythonProcess.kill();
    if (process.platform !== 'darwin') app.quit();
});

app.on('before-quit', () => {
    if (pythonProcess) pythonProcess.kill();
});
