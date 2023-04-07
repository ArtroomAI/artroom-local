import { ipcRenderer } from "electron";
import type { ProgressInfo } from "electron-updater";

const api = {
    startArtroom: async (artroomPath: string, isDebug: boolean) => {return await ipcRenderer.invoke('startArtroom',artroomPath, isDebug);},
    runPyTests: async (artroomPath: string) => {return await ipcRenderer.invoke('runPyTests', artroomPath);},
    getCkpts: async (data: any) => {return await ipcRenderer.invoke('getCkpts',data);},
    getLoras: async (data: any) => {return await ipcRenderer.invoke('getLoras',data);},
    getVaes: async (data: any) => {return await ipcRenderer.invoke('getVaes',data);},
    getImages: async (data: any): Promise<string[]> => {return await ipcRenderer.invoke('getImages',data);},
    imageViewer: async (folder_path: string, batch_path: string): Promise<ImageViewerResultType> => { return await ipcRenderer.invoke('imageViewer', folder_path, batch_path); },
    uploadSettings: async () => {return await ipcRenderer.invoke('uploadSettings');},
    chooseImages: async (): Promise<string[]> => {return await ipcRenderer.invoke('chooseImages');},
    openDiscord: async () => {return await ipcRenderer.invoke('openDiscord');},
    openCivitai: async () => {return await ipcRenderer.invoke('openCivitai');},
    openEquilibrium: async () => {return await ipcRenderer.invoke('openEquilibrium');},
    uploadInitImage: async (data: any) => {return await ipcRenderer.invoke('uploadInitImage',data);},
    getImageFromPath: async (data: string): Promise<{ b64: string, metadata: string }> => {return await ipcRenderer.invoke('getImageFromPath',data);},
    copyToClipboard: async (data: string, type?: string) => {return await ipcRenderer.invoke('copyToClipboard', data, type);},
    saveFromDataURL: async (data: string) => {return await ipcRenderer.invoke('saveFromDataURL',data);},
    chooseUploadPath: async (): Promise<string> => {return await ipcRenderer.invoke('chooseUploadPath');},
    restartServer: async (artroomPath: string, isDebug: boolean) => {return await ipcRenderer.invoke('restartServer', artroomPath, isDebug);},
    showInExplorer: async (path: string) => {return await ipcRenderer.invoke('showInExplorer', path);},
    minimizeWindow: async () => {return await ipcRenderer.invoke('minimizeWindow');},
    unmaximizeWindow: async () => {return await ipcRenderer.invoke('unmaximizeWindow');},
    maxUnmaxWindow: async () => {return await ipcRenderer.invoke('maxUnmaxWindow');},
    closeWindow: async () => {return await ipcRenderer.invoke('closeWindow');},
    getVersion: async () => {return await ipcRenderer.invoke('getVersion');},
    pythonInstall: async(artroomPath: string, gpuType: string) => {return await ipcRenderer.invoke('pythonInstall', artroomPath, gpuType);},
    pythonInstallDependencies: async(artroomPath: string) => {return await ipcRenderer.invoke('pythonInstallDependencies', artroomPath);},
    downloadStarterModels: async(dir: string, realisticStarter: boolean, animeStarter: boolean, landscapesStarter: boolean) => {return await ipcRenderer.invoke('downloadStarterModels', dir, realisticStarter, animeStarter, landscapesStarter);},
    fixButtonProgress: async (callback: (event: Electron.IpcRendererEvent, ...args: any[]) => void) => {ipcRenderer.on('fixButtonProgress', callback);},
    downloadProgress: async (callback: (event: Electron.IpcRendererEvent, info: ProgressInfo) => void) => {ipcRenderer.on('downloadProgress', callback);},
    saveQueue: async (queue: string, artroom_path: string) => { ipcRenderer.invoke('saveQueue', queue, artroom_path) },
    readQueue: async (artroom_path: string): Promise<string | undefined> => { return await ipcRenderer.invoke('readQueue', artroom_path) }
}

window.api = api;

declare global {
    interface Window {
        api: typeof api;
    }
}
