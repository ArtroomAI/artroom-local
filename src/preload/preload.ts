import { ipcRenderer as IPC, IpcRendererEvent } from "electron";
import type { ProgressInfo } from "electron-updater";

const api = {
    startArtroom: async (artroomPath: string, isDebug: boolean) => IPC.invoke('startArtroom',artroomPath, isDebug),
    runPyTests: async (artroomPath: string) => IPC.invoke('runPyTests', artroomPath),
    getCkpts: async (data: any) => IPC.invoke('getCkpts',data),
    getLoras: async (data: any) => IPC.invoke('getLoras',data),
    getVaes: async (data: any) => IPC.invoke('getVaes',data),
    imageViewer: async (folder_path: string, batch_path: string): Promise<ImageViewerResultType> => IPC.invoke('imageViewer', folder_path, batch_path),
    getDisks: async(): Promise<string[]> => IPC.invoke('getDisks'),
    uploadSettings: async (): Promise<string> => IPC.invoke('uploadSettings'),
    chooseImages: async (): Promise<string[]> => IPC.invoke('chooseImages'),
    openDiscord: async () => IPC.invoke('openDiscord'),
    openCivitai: async () => IPC.invoke('openCivitai'),
    openEquilibrium: async () => IPC.invoke('openEquilibrium'),
    uploadInitImage: async (data: any) => IPC.invoke('uploadInitImage',data),
    getImageFromPath: async (data: string): Promise<{ b64: string, metadata: string }> => IPC.invoke('getImageFromPath',data),
    copyToClipboard: async (data: string, type?: string) => IPC.invoke('copyToClipboard', data, type),
    saveFromDataURL: async (data: string) => IPC.invoke('saveFromDataURL',data),
    chooseUploadPath: async (): Promise<string> => IPC.invoke('chooseUploadPath'),
    restartServer: async (artroomPath: string, isDebug: boolean) => IPC.invoke('restartServer', artroomPath, isDebug),
    showInExplorer: async (path: string) => IPC.invoke('showInExplorer', path),
    minimizeWindow: async () => IPC.invoke('minimizeWindow'),
    unmaximizeWindow: async () => IPC.invoke('unmaximizeWindow'),
    maxUnmaxWindow: async () => IPC.invoke('maxUnmaxWindow'),
    closeWindow: async () => IPC.invoke('closeWindow'),
    getVersion: async () => IPC.invoke('getVersion'),
    pythonInstall: async(artroomPath: string, gpuType: string) => IPC.invoke('pythonInstall', artroomPath, gpuType),
    pythonInstallDependencies: async(artroomPath: string) => IPC.invoke('pythonInstallDependencies', artroomPath),
    downloadStarterModels: async(dir: string, realisticStarter: boolean, animeStarter: boolean, landscapesStarter: boolean) => IPC.invoke('downloadStarterModels', dir, realisticStarter, animeStarter, landscapesStarter),
    saveQueue: async (queue: string, artroom_path: string) => IPC.invoke('saveQueue', queue, artroom_path),
    readQueue: async (artroom_path: string): Promise<string | undefined> => IPC.invoke('readQueue', artroom_path),
    fixButtonProgress: (callback: (_: IpcRendererEvent, message: string) => void) => {
        IPC.on('fixButtonProgress', callback);
        return () => IPC.off('fixButtonProgress', callback);
    },
    downloadProgress: (callback: (_: IpcRendererEvent, info: ProgressInfo) => void) => {
        IPC.on('downloadProgress', callback);
        return () => IPC.off('downloadProgress', callback);
    },
    imageViewerChange: (callback: (_: IpcRendererEvent, result: ImageViewerResultType) => void) => {
        IPC.on('imageViewerChange', callback);
        return () => IPC.off('imageViewerChange', callback);
    }
}

window.api = api;

declare global {
    interface Window {
        api: typeof api;
    }
}
