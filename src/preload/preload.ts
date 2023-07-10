import { ipcRenderer as IPC, IpcRendererEvent } from "electron";
import type { ProgressInfo } from "electron-updater";
import type { BackEndImage, ExifValidation } from "../interfaces/imageData";

const api = {
    startArtroom: (artroomPath: string, isDebug: boolean) => IPC.invoke('startArtroom',artroomPath, isDebug),
    runPyTests: (artroomPath: string) => IPC.invoke('runPyTests', artroomPath),
    imageViewer: (folder_path: string, batch_path: string): Promise<ImageViewerResultType> => IPC.invoke('imageViewer', folder_path, batch_path),
    getDisks: (): Promise<string[]> => IPC.invoke('getDisks'),
    uploadSettings: (): Promise<ExifValidation | null> => IPC.invoke('uploadSettings'),
    chooseImages: (): Promise<string[]> => IPC.invoke('chooseImages'),
    openWebsite: () => IPC.invoke('openWebsite'),
    openDiscord: () => IPC.invoke('openDiscord'),
    openCivitai: () => IPC.invoke('openCivitai'),
    openTutorial: () => IPC.invoke('openTutorial'),
    uploadInitImage: (data: any) => IPC.invoke('uploadInitImage',data),
    getImageFromPath: (image_path: string): Promise<BackEndImage> => IPC.invoke('getImageFromPath',image_path),
    copyToClipboard: (data: string, type?: string) => IPC.invoke('copyToClipboard', data, type),
    saveFromDataURL: (data: string) => IPC.invoke('saveFromDataURL',data),
    chooseUploadPath: (): Promise<string> => IPC.invoke('chooseUploadPath'),
    restartServer: (artroomPath: string, isDebug: boolean) => IPC.invoke('restartServer', artroomPath, isDebug),
    showInExplorer: (path: string) => IPC.invoke('showInExplorer', path),
    minimizeWindow: () => IPC.invoke('minimizeWindow'),
    unmaximizeWindow: () => IPC.invoke('unmaximizeWindow'),
    maxUnmaxWindow: () => IPC.invoke('maxUnmaxWindow'),
    closeWindow: () => IPC.invoke('closeWindow'),
    getVersion: () => IPC.invoke('getVersion'),
    pythonInstall: (artroomPath: string, gpuType: string) => IPC.invoke('pythonInstall', artroomPath, gpuType),
    pythonInstallDependencies: (artroomPath: string, gpuType: string) => IPC.invoke('pythonInstallDependencies', artroomPath, gpuType),
    downloadStarterModels: (dir: string, realisticStarter: boolean, animeStarter: boolean, landscapesStarter: boolean) => IPC.invoke('downloadStarterModels', dir, realisticStarter, animeStarter, landscapesStarter),
    saveQueue: (queue: string, artroom_path: string) => IPC.invoke('saveQueue', queue, artroom_path),
    readQueue: (artroom_path: string): Promise<string | undefined> => IPC.invoke('readQueue', artroom_path),
    modelsFolder: (folder_name: string) => IPC.invoke('modelsFolder', folder_name),
    modelsChange: (callback: (_: IpcRendererEvent, models: InternalModelsType) => void) => {
        IPC.on('modelsChange', callback);
        return () => IPC.off('modelsChange', callback);
    },
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
