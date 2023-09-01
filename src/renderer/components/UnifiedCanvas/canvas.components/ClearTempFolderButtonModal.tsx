import React from 'react'
import { FC } from 'react'
import { FaTrash } from 'react-icons/fa'
import { useRecoilValue, useSetRecoilState } from 'recoil'
import { Button, AlertDialog } from '../components'
import {
  clearCanvasHistoryAction,
  resetCanvasAction,
  isStagingSelector,
} from '../atoms/canvas.atoms'

// import { emptyTempFolder } from 'app/socketio/actions';
// import { isStagingSelector } from 'canvas/store/canvasSelectors';
// import {
// 	clearCanvasHistory,
// 	resetCanvas,
// } from 'canvas/store/canvasSlice';

export const EmptyTempFolderButtonModal: FC = () => {
  // const isStaging = useAppSelector(isStagingSelector);

  const clearCanvasHistory = useSetRecoilState(clearCanvasHistoryAction)
  const resetCanvas = useSetRecoilState(resetCanvasAction)
  const isStaging = useRecoilValue(isStagingSelector)

  const acceptCallback = () => {
    // dispatch(emptyTempFolder());
    alert('emptyTempFolder action placeholder')
    resetCanvas()
    clearCanvasHistory()
  }

  return (
    <AlertDialog
      title="Empty Temp Image Folder"
      acceptCallback={acceptCallback}
      acceptButtonText="Empty Folder"
      triggerComponent={
        <Button leftIcon={<FaTrash />} size="sm" isDisabled={isStaging}>
          Empty Temp Image Folder
        </Button>
      }
    >
      <p>
        Emptying the temp image folder also fully resets the Unified Canvas. This includes all
        undo/redo history, images in the staging area, and the canvas base layer.
      </p>
      <br />
      <p>Are you sure you want to empty the temp folder?</p>
    </AlertDialog>
  )
}
