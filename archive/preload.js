const {
    contextBridge,
    ipcRenderer,
} = require("electron");

contextBridge.exposeInMainWorld(
    'api', {
        startSD: async (channel) => {return await ipcRenderer.invoke(channel);},
        paintSD: async (channel,data) => {return await ipcRenderer.invoke(channel,data);},
        upscale: async (channel,data) => {return await ipcRenderer.invoke(channel,data);},
        updateConfig: async (channel,data) => {return await ipcRenderer.invoke(channel,data);},
        updateSettings: async (channel,data) => {return await ipcRenderer.invoke(channel,data);},
        uploadInitImage: async (channel,data) => {return await ipcRenderer.invoke(channel,data);},
        chooseUploadPath: async (channel,data) => {return await ipcRenderer.invoke(channel,data);},
        chooseImages: async (channel) => {return await ipcRenderer.invoke(channel);},
        getImage: async (channel) => {return await ipcRenderer.invoke(channel);},
        getImageFromPath: async (channel,data) => {return await ipcRenderer.invoke(channel,data);},
        getImages: async (channel,data) => {return await ipcRenderer.invoke(channel,data);},
        getModelCkpts: async (channel,data) => {return await ipcRenderer.invoke(channel,data);},
        updateSettings: async (channel,data) => {return await ipcRenderer.invoke(channel,data);},
        getSettings: async (channel) => {return await ipcRenderer.invoke(channel);},
        uploadSettings: async (channel) => {return await ipcRenderer.invoke(channel);},
        getQueue: async (channel) => {return await ipcRenderer.invoke(channel);},
        writeQueue: async (channel,data) => {return await ipcRenderer.invoke(channel,data);},
        getUpscaleSettings: async (channel) => {return await ipcRenderer.invoke(channel);},
        getImageDir: async (channel) => {return await ipcRenderer.invoke(channel);},
    },

);


