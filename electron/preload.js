const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('dataAnalizeDesktop', {
    pickDataset: () => ipcRenderer.invoke('dataset:pick-file'),
});
