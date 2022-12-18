/* eslint-disable no-undef */
const {
    contextBridge,
    ipcRenderer
} = require('electron');

const {
    getCurrentWindow,
    minimizeWindow,
    unmaximizeWindow,
    maxUnmaxWindow,
    isWindowMaximized,
    closeWindow,
    getVersion
} = require('./menu-functions');

contextBridge.exposeInMainWorld(
    'api',
    {
        getCkpts: (channel, data) => ipcRenderer.invoke(
            channel,
            data
        ),
        reinstallArtroom: (channel) => ipcRenderer.invoke(channel),
        getSettings: (channel) => ipcRenderer.invoke(channel),
        uploadSettings: (channel) => ipcRenderer.invoke(channel),
        chooseImages: (channel) => ipcRenderer.invoke(channel),
        getImageDir: (channel) => ipcRenderer.invoke(channel),
        openDiscord: (channel) => ipcRenderer.invoke(channel),
        openEquilibrium: (channel) => ipcRenderer.invoke(channel),
        uploadInitImage: (channel, data) => ipcRenderer.invoke(
            channel,
            data
        ),
        getImageFromPath: (channel, data) => ipcRenderer.invoke(
            channel,
            data
        ),
        copyToClipboard: (channel, data) => ipcRenderer.invoke(
            channel,
            data
        ),
        chooseUploadPath: (channel, data) => ipcRenderer.invoke(
            channel,
            data
        ),
        runPyTests: (channel, data) => ipcRenderer.invoke(
            channel,
            data
        ),
        restartServer: (channel, isDebug) => ipcRenderer.invoke(
            channel,
            isDebug
        ),
        mergeModels: (channel, data) => ipcRenderer.invoke(
            channel,
            data
        ),
        getCurrentWindow: () => {
            getCurrentWindow();
        },
        minimizeWindow: () => {
            minimizeWindow();
        },
        unmaximizeWindow: () => {
            unmaximizeWindow();
        },
        maxUnmaxWindow: () => {
            maxUnmaxWindow();
        },
        isWindowMaximized: () => {
            isWindowMaximized();
        },
        closeWindow: () => {
            closeWindow();
        },
        getVersion: () => {
            getVersion();
        }
    }
);
