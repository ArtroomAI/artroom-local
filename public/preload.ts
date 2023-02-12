import { contextBridge, ipcRenderer } from "electron";
import { ProgressInfo } from "electron-updater";

const api = {
    getCkpts: async (data: any) => {return await ipcRenderer.invoke('getCkpts',data);},
    getVaes: async (data: any) => {return await ipcRenderer.invoke('getVaes',data);},
    getImages: async (data: any): Promise<string[]> => {return await ipcRenderer.invoke('getImages',data);},
    mergeModels: async (data: any) => {return await ipcRenderer.invoke('mergeModels',data);},
    reinstallArtroom: async () => {return await ipcRenderer.invoke('reinstallArtroom');},
    getSettings: async () => {return await ipcRenderer.invoke('getSettings');},
    uploadSettings: async () => {return await ipcRenderer.invoke('uploadSettings');},
    chooseImages: async (): Promise<string[]> => {return await ipcRenderer.invoke('chooseImages');},
    getImageDir: async () => {return await ipcRenderer.invoke('getImageDir');},
    openDiscord: async () => {return await ipcRenderer.invoke('openDiscord');},
    openEquilibrium: async () => {return await ipcRenderer.invoke('openEquilibrium');},
    uploadInitImage: async (data: any) => {return await ipcRenderer.invoke('uploadInitImage',data);},
    getImageFromPath: async (data: any) => {return await ipcRenderer.invoke('getImageFromPath',data);},
    copyToClipboard: async (data: string) => {return await ipcRenderer.invoke('copyToClipboard',data);},
    saveFromDataURL: async (data: string) => {return await ipcRenderer.invoke('saveFromDataURL',data);},
    chooseUploadPath: async (): Promise<string> => {return await ipcRenderer.invoke('chooseUploadPath');},
    runPyTests: async () => {return await ipcRenderer.invoke('runPyTests');},
    restartServer: async (isDebug: boolean) => {return await ipcRenderer.invoke('restartServer',isDebug);},
    showInExplorer: async (path: string) => {return await ipcRenderer.invoke('showInExplorer', path);},
    minimizeWindow: async () => {return await ipcRenderer.invoke('minimizeWindow');},
    unmaximizeWindow: async () => {return await ipcRenderer.invoke('unmaximizeWindow');},
    maxUnmaxWindow: async () => {return await ipcRenderer.invoke('maxUnmaxWindow');},
    closeWindow: async () => {return await ipcRenderer.invoke('closeWindow');},
    getVersion: async () => {return await ipcRenderer.invoke('getVersion');},
    pythonInstall: async(useAMDInstaller: boolean) => {return await ipcRenderer.invoke('pythonInstall', useAMDInstaller);},
    pythonInstallDependencies: async() => {return await ipcRenderer.invoke('pythonInstallDependencies');},
    fixButtonProgress: async (callback: (event: Electron.IpcRendererEvent, ...args: any[]) => void) => {ipcRenderer.on('fixButtonProgress', callback);},
    downloadProgress: async (callback: (event: Electron.IpcRendererEvent, info: ProgressInfo) => void) => {ipcRenderer.on('downloadProgress', callback);}
}

if (process.env.NODE_ENV !== 'production') {
    window.api = api;
} else {
    contextBridge.exposeInMainWorld(
        'api', api
    );
}

declare global {
    interface Window {
        api: typeof api;
    }
}
