import fs from 'fs';
import { spawn } from 'child_process';
import { ipcMain } from 'electron';

function runPyTests(artroomPath: string) {
  return new Promise((resolve, reject) => {
    console.log("TEST ARTROOM PATH", artroomPath)
    const pyTestCmd = `${artroomPath}\\artroom\\artroom_backend\\python.exe`;
    let childPython = spawn(pyTestCmd, ['pytest.py']);
    let result = '';
    childPython.stdout.on(`data`, (data) => {
      result += data.toString();
    });

    childPython.on('close', () => resolve(result));
    childPython.on('error', reject);
  })
};

async function runTest(artroomPath: string) {
  const python_path = `${artroomPath}\\artroom\\artroom_backend\\python.exe`;
  if (!(fs.existsSync(python_path))) {
    return "cannot find python in " + python_path;
  }
  try {
    const res = await runPyTests(artroomPath);
    return res
  } catch (err) {
    return err
  }
}

export const pytestHandles = () => {
  ipcMain.handle('runPyTests', async (_, artroomPath) => runTest(artroomPath));
}
