const {
    contextBridge,
    ipcRenderer
} = require("electron");

const {
    getCurrentWindow,
    minimizeWindow,
    unmaximizeWindow,
    maxUnmaxWindow,
    isWindowMaximized,
    closeWindow,
    getVersion
  } = require("./menu-functions");

contextBridge.exposeInMainWorld(
    'api', {
        getCkpts: async (channel,data) => {return await ipcRenderer.invoke(channel,data);},
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
        getCurrentWindow: () => {getCurrentWindow()},
        getCurrentWindow : () => {getCurrentWindow()},
        minimizeWindow : () => {minimizeWindow()},
        unmaximizeWindow : () => {unmaximizeWindow()},
        maxUnmaxWindow : () => {maxUnmaxWindow()},
        isWindowMaximized : () => {isWindowMaximized()},
        closeWindow : () => {closeWindow()},
        getVersion : () => {getVersion()},
    }
);