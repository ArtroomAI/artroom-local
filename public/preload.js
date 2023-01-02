/* eslint-disable no-undef */
const {
    contextBridge,
    ipcRenderer
} = require("electron");

const api = {
    getCkpts: async (channel,data) => {return await ipcRenderer.invoke(channel,data);},
    getVaes: async (channel,data) => {return await ipcRenderer.invoke(channel,data);},
    getImages: async (channel,data) => {return await ipcRenderer.invoke(channel,data);},
    mergeModels: async (channel,data) => {return await ipcRenderer.invoke(channel,data);},
    reinstallArtroom: async (channel) => {return await ipcRenderer.invoke(channel);},
    getSettings: async (channel) => {return await ipcRenderer.invoke(channel);},
    uploadSettings: async (channel) => {return await ipcRenderer.invoke(channel);},
    chooseImages: async (channel) => {return await ipcRenderer.invoke(channel);},
    getImageDir: async (channel) => {return await ipcRenderer.invoke(channel);},
    openDiscord: async (channel) => {return await ipcRenderer.invoke(channel);},
    openEquilibrium: async (channel) => {return await ipcRenderer.invoke(channel);},
    uploadInitImage: async (channel,data) => {return await ipcRenderer.invoke(channel,data);},
    getImageFromPath: async (channel,data) => {return await ipcRenderer.invoke(channel,data);},
    copyToClipboard: async (channel,data) => {return await ipcRenderer.invoke(channel,data);},
    chooseUploadPath: async (channel,data) => {return await ipcRenderer.invoke(channel,data);},
    runPyTests: async (channel,data) => {return await ipcRenderer.invoke(channel,data);},
    restartServer: async (channel,isDebug) => {return await ipcRenderer.invoke(channel,isDebug);},
    minimizeWindow: async (channel) => {return await ipcRenderer.invoke(channel);},
    unmaximizeWindow: async (channel) => {return await ipcRenderer.invoke(channel);},
    maxUnmaxWindow: async (channel) => {return await ipcRenderer.invoke(channel);},
    closeWindow: async (channel) => {return await ipcRenderer.invoke(channel);},
    getVersion: async (channel) => {return await ipcRenderer.invoke(channel);},
}

if (process.env.NODE_ENV !== 'production') {
    window.api = api;
} else {
    contextBridge.exposeInMainWorld(
        'api', api
    );
}
