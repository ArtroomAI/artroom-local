import http from 'http';
import fs from 'fs';
import path from 'path';
import { ChildProcessWithoutNullStreams, spawn } from 'child_process';
import { ipcMain } from "electron";
import decompress from 'decompress';
import { deleteSync } from 'del';
 
// 1. Download from URL
// 2. Delete the miniconda3 folder from artroom_folder (defaults to C/Users/yourname/artroom)
// 3. Unzip the file downloaded, move to miniconda3, delete .zip file
// 4. Run (with replacing program files with the app directory and %UserProfile% with artroom_folder 
// 5. "%UserProfile%\artroom\miniconda3\condabin\activate" && "%UserProfile%\artroom\miniconda3\Scripts\conda" run -n artroom-ldm pip install -r stable-diffusion/requirements.txt

let installationProcess: ChildProcessWithoutNullStreams;

const backupPythonInstallation = (artroom_path: string) => () => {
    const URL = '';
    const PATH = path.resolve(artroom_path, "\\artroom\\miniconda3");
    const instalationCommand = `"${PATH}\\condabin\\activate" && "${PATH}\\Scripts\\conda" run -n artroom-ldm pip install -r stable-diffusion/requirements.txt`;

    if(fs.existsSync(PATH)) {
        deleteSync(PATH);
    }

    const request = http.get(URL, (response) => {
        const len = parseInt(response.headers['content-length'], 10);
        let cur = 0;
        const toMB = (n: number) => (n / 1048576).toFixed(2);

        const file = fs.createWriteStream("file.zip");
        response.pipe(file);

        const total = toMB(len); //1048576 - bytes in 1 MB

        response.on("data", (chunk) => {
            cur += chunk.length;
            
            ipcMain.emit('fixButtonProgress', `Downloading ${(100 * cur / len).toFixed(2)}% - ${toMB(cur)}mb / ${total}mb`);
        });

        file.on("finish", () => {
            file.close();
            ipcMain.emit('fixButtonProgress', 'Downloading complete. Decompressing...');
            decompress(file.path, PATH).then(() => {
                ipcMain.emit('fixButtonProgress', 'Decompresed!');

                installationProcess = spawn(instalationCommand, { detached: true, shell: true });
            });
        });

        request.on("error", (e) => {
            ipcMain.emit('fixButtonProgress', `Error: ${e.message}`);
            file.close();
        });
    });
};

export const handlers = (artroom_path: string) => {
    ipcMain.handle('saveFromDataURL', backupPythonInstallation(artroom_path));
}