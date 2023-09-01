import { atom } from 'recoil'

// Can be used to show that some image processing is happening, and it will block some actions, like redo
export const isProcessingAtom = atom({
  key: 'system.isProcessing',
  default: false,
})
