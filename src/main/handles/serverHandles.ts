import axios from 'axios';
import { ChildProcessWithoutNullStreams, spawn } from 'child_process';
import kill from 'tree-kill';

const LOCAL_URL = process.env.LOCAL_URL;

export class Server {
  private server: ChildProcessWithoutNullStreams;

  public start(artroomPath: string, debug_mode: boolean) {
    const command = `"${artroomPath}\\artroom\\miniconda3\\Scripts\\conda" run --no-capture-output -p "${artroomPath}/artroom/miniconda3/envs/artroom-ldm" python server.py`;
    console.log(`debug mode: ${debug_mode}`)
    this.kill();
    this.server = spawn(command, { detached: debug_mode, shell: true });
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
