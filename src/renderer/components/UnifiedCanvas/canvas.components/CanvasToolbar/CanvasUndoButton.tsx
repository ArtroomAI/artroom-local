import React, { useCallback } from 'react'
import { FC } from 'react'
import { useHotkeys } from 'react-hotkeys-hook'
import { FaUndo } from 'react-icons/fa'
import { useSetRecoilState, useRecoilValue } from 'recoil'
import { IconButton } from '../../components'
import { undoAction, canUndoSelector } from '../../atoms/canvas.atoms'

// import { canvasSelector } from 'canvas/store/canvasSelectors';
// import _ from 'lodash';
// import { activeTabNameSelector } from 'options/store/optionsSelectors';
// import { undo } from 'canvas/store/canvasSlice';
// import { systemSelector } from 'system/store/systemSelectors';

// const canvasUndoSelector = createSelector(
// 	[canvasSelector, activeTabNameSelector, systemSelector],
// 	(canvas, activeTabName, system) => {
// 		const { pastLayerStates } = canvas;

// 		return {
// 			canUndo: pastLayerStates.length > 0 && !system.isProcessing,
// 			activeTabName,
// 		};
// 	},
// 	{
// 		memoizeOptions: {
// 			resultEqualityCheck: _.isEqual,
// 		},
// 	},
// );

export const CanvasUndoButton: FC = () => {
  // const { canUndo, activeTabName } = useAppSelector(canvasUndoSelector);
  const undo = useSetRecoilState(undoAction)
  const canUndo = useRecoilValue(canUndoSelector)

  const handleUndo = useCallback(() => {
    undo()
  }, [])

  useHotkeys(
    ['meta+z', 'ctrl+z'],
    () => {
      handleUndo()
    },
    {
      enabled: () => canUndo,
      preventDefault: true,
    },
    [canUndo]
  )

  return (
    <IconButton
      aria-label="Undo (Ctrl+Z)"
      tooltip="Undo (Ctrl+Z)"
      icon={<FaUndo />}
      onClick={handleUndo}
      isDisabled={!canUndo}
    />
  )
}
