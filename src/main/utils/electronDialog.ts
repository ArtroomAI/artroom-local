import { dialog, OpenDialogOptions } from 'electron'
import os from 'os'

type Mode = 'Directory' | 'File' | 'Files'

export const getOpenDialogProps = (mode: Mode): OpenDialogOptions['properties'] => {
  if (mode === 'Directory') {
    return ['openDirectory']
  }

  if (mode === 'Files') {
    if (!(os.platform() === 'linux' || os.platform() === 'win32')) {
      return ['openDirectory', 'openFile', 'multiSelections']
    }
    return ['openFile', 'multiSelections']
  }

  if (mode === 'File') {
    return ['openFile']
  }

  return []
}

export const electronDialog = async (mode: Mode, extensions?: string[]) => {
  const filters = extensions ? [{ name: 'Files', extensions }] : undefined

  const results = await dialog.showOpenDialog({
    properties: getOpenDialogProps(mode),
    filters,
  })

  return results.filePaths
}
