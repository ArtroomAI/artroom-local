import { watch, FSWatcher } from 'fs';

type FileWatcherCallback = (event: string, filename: string) => void;

export class FileWatcher {
    private current_folder_path = '';
    private watcher: FSWatcher | null = null;
    private callback: FileWatcherCallback = () => {};

    public reassignWatcher = (folder_path: string, callback: FileWatcherCallback) => {
        if(this.current_folder_path === folder_path) return;

        this.callback = callback;
        this.current_folder_path = folder_path;
        this.watcher?.close();
        this.watcher = watch(folder_path, this.callback)
    }
}
