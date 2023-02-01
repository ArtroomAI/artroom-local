import https from 'https';
import fs from 'fs';
import path from 'path';
import { ChildProcessWithoutNullStreams, spawn } from 'child_process';
import { ipcMain } from "electron";
import StreamZip from 'node-stream-zip';
import { deleteSync } from 'del';
 
// 1. Download from URL
// 2. Delete the miniconda3 folder from artroom_folder (defaults to C/Users/yourname/artroom)
// 3. Unzip the file downloaded, move to miniconda3, delete .zip file
// 4. Run (with replacing program files with the app directory and %UserProfile% with artroom_folder 
// 5. "%UserProfile%\artroom\miniconda3\condabin\activate" && "%UserProfile%\artroom\miniconda3\Scripts\conda" run -n artroom-ldm pip install -r stable-diffusion/requirements.txt

let installationProcess: ChildProcessWithoutNullStreams;

const backupPythonInstallation = (artroom_path: string) => () => {
    const URL = 'https://pub-060d7c8cf5e64af8b884ebb86d34de1a.r2.dev/miniconda3.zip';
    const PATH = path.resolve(artroom_path, "\\artroom\\miniconda3");

    const PATH_requirements = path.resolve('stable-diffusion/requirements.txt');

    const instalationCommand = `"${PATH}\\condabin\\activate.bat" && "${PATH}\\Scripts\\conda.exe" run -n artroom-ldm pip install -r "${PATH_requirements}"`;

    if(fs.existsSync(PATH)) {
        deleteSync(PATH);
    }

    const request = https.get(URL, (response) => {
        const len = parseInt(response.headers['content-length'], 10);
        let cur = 0;
        const toMB = (n: number) => (n / 1048576).toFixed(2);

        const file = fs.createWriteStream("file.zip");
        response.pipe(file);

        const total = toMB(len); //1048576 - bytes in 1 MB

        let chunk_counter = 0;

        response.on("data", (chunk) => {
            cur += chunk.length;
            ++chunk_counter;
            if(chunk_counter === 5000) {
                console.log(`Downloading ${(100 * cur / len).toFixed(2)}% - ${toMB(cur)}mb / ${total}mb`);
                ipcMain.emit('fixButtonProgress', `Downloading ${(100 * cur / len).toFixed(2)}% - ${toMB(cur)}mb / ${total}mb`);
                chunk_counter = 0;
            }
       });

        file.on("finish", () => {
            file.close();
            console.log('Downloading complete. Decompressing...');
            
            ipcMain.emit('fixButtonProgress', 'Downloading complete. Decompressing...');
            const zip = new StreamZip({ file: 'file.zip' });

            zip.on('ready', () => {
                fs.mkdirSync(PATH, { recursive: true });
                zip.extract(null, path.resolve(PATH), (err, count) => {
                    ipcMain.emit('fixButtonProgress', err ? 'Extract error' : `Extracted ${count} entries`);
                    console.log(err ? 'Extract error' : `Extracted ${count} entries`);
                    installationProcess = spawn(instalationCommand, { shell: true, detached: true });
                    installationProcess.stdout.on("data", (data) => {
                        console.log(`stdout: ${data}`);
                    });
                    installationProcess.stderr.on("data", (data) => {
                        console.error(`stderr: ${data}`);
                    });
                    installationProcess.on("close", (code) => {
                        console.log(`child process exited with code ${code}`);
                        ipcMain.emit('fixButtonProgress', `child process exited with code ${code}`);
                    });
                    zip.close();
                });
            });
        });

        request.on("error", (e) => {
            ipcMain.emit('fixButtonProgress', `Error: ${e.message}`);
            file.close();
        });
    });
};

export const handlers = (artroom_path: string) => {
    ipcMain.handle('pythonInstall', backupPythonInstallation(artroom_path));
}
