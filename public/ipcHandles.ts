import https from 'https';
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
    const URL = 'https://download.wetransfer.com/usgv/ff33323817a15accb4979b7dde9435c020230114065819/46127b455b9d6cea3c6c651facafdc54d46f3800/miniconda3.zip?token=eyJhbGciOiJIUzI1NiJ9.eyJpYXQiOjE2NzUyNjQ1OTMsImV4cCI6MTY3NTI2NTE5MywidW5pcXVlIjoiZmYzMzMyMzgxN2ExNWFjY2I0OTc5YjdkZGU5NDM1YzAyMDIzMDExNDA2NTgxOSIsImZpbGVuYW1lIjoibWluaWNvbmRhMy56aXAiLCJ3YXliaWxsX3VybCI6Imh0dHA6Ly9zdG9ybS1pbnRlcm5hbC5zZXJ2aWNlLnVzLWVhc3QtMS53ZXRyYW5zZmVyLm5ldC9hcGkvd2F5YmlsbHM_c2lnbmVkX3dheWJpbGxfaWQ9ZXlKZmNtRnBiSE1pT25zaWJXVnpjMkZuWlNJNklrSkJhSE5MZDJodFlYZGpSa0ZSUVQwaUxDSmxlSEFpT2lJeU1ESXpMVEF5TFRBeFZERTFPakkyT2pNekxqQXdNRm9pTENKd2RYSWlPaUozWVhsaWFXeHNYMmxrSW4xOS0tYWYwYzRmMTcxMjAyYjA3ZTA1Njk2N2Y1YzA5MDE5NjQwMjVlYTE0OGQzZTUyZThiZDY4MDdiZjk4YzE4ZTI1YyIsImZpbmdlcnByaW50IjoiNDYxMjdiNDU1YjlkNmNlYTNjNmM2NTFmYWNhZmRjNTRkNDZmMzgwMCIsImNhbGxiYWNrIjoie1wiZm9ybWRhdGFcIjp7XCJhY3Rpb25cIjpcImh0dHA6Ly9mcm9udGVuZC5zZXJ2aWNlLmV1LXdlc3QtMS53ZXRyYW5zZmVyLm5ldC93ZWJob29rcy9iYWNrZW5kXCJ9LFwiZm9ybVwiOntcInRyYW5zZmVyX2lkXCI6XCJmZjMzMzIzODE3YTE1YWNjYjQ5NzliN2RkZTk0MzVjMDIwMjMwMTE0MDY1ODE5XCIsXCJkb3dubG9hZF9pZFwiOjE3NzYxNzgyMDAwfX0ifQ.gK6QqHKGgRH4uWsWsD0N2cdGP_dlTGWl0TFtmMDxw0c&cf=y';
    const PATH = path.resolve(artroom_path, "\\artroom\\miniconda3");
    const instalationCommand = `"${PATH}\\condabin\\activate" && "${PATH}\\Scripts\\conda" run -n artroom-ldm pip install -r stable-diffusion/requirements.txt`;

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
    ipcMain.handle('pythonInstall', backupPythonInstallation(artroom_path));
}