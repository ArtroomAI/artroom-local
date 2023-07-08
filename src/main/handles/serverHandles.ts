import axios from 'axios';
import { ChildProcessWithoutNullStreams, spawn } from 'child_process';
import kill from 'tree-kill';

const LOCAL_URL = process.env.LOCAL_URL;

export class Server {
  private server: ChildProcessWithoutNullStreams;

  public start(artroomPath: string, debug_mode: boolean) {
    const command = `"${artroomPath}/artroom/artroom_backend/python.exe" server.cpython-310.opt-2.pyc`;
    this.kill();
    const options = {
      detached: debug_mode,
      shell: true,
      windowsHide: true, // Add this line to hide the CMD window
    };
    this.server = spawn(command, options);

    this.server.on('error', (error) => {
      console.log(`Error: ${error.message}`);
    });

    this.server.stderr.on('data', (data) => {
      console.log(`stderr: ${data}`);
    });
  }


  
  public kill() {
    if (this.server && this.server.pid) {
      kill(this.server.pid);
      spawn("taskkill", ["/pid", `${this.server.pid}`, '/f', '/t']);
      axios.get(`${LOCAL_URL}/shutdown`)
    }
  }

  public serverHandles(ipcMain: Electron.IpcMain) {
    ipcMain.handle('restartServer', async (_, artroomPath, debug_mode) => {
      this.start(artroomPath, debug_mode);
    });
    
    ipcMain.handle('startArtroom', async (_, artroomPath, debug_mode) => {
      this.start(artroomPath, debug_mode);
    });
  }
}
