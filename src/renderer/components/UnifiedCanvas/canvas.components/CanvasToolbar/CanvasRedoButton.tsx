import React, { useCallback } from 'react'
import { FC } from 'react'
import { useHotkeys } from 'react-hotkeys-hook'
import { FaRedo } from 'react-icons/fa'
import { useRecoilValue, useSetRecoilState } from 'recoil'
import { IconButton } from '../../components'
import { redoAction, canRedoSelector } from '../../atoms/canvas.atoms'

// import _ from 'lodash';
// import { activeTabNameSelector } from 'options/store/optionsSelectors';
// import { canvasSelector } from 'canvas/store/canvasSelectors';
// import { redo } from 'canvas/store/canvasSlice';
// import { systemSelector } from 'system/store/systemSelectors';

// const canvasRedoSelector = createSelector(
// 	[canvasSelector, activeTabNameSelector, systemSelector],
// 	(canvas, activeTabName) => {
// 		const { futureLayerStates } = canvas;

// 		return {
// 			canRedo: futureLayerStates.length > 0 && !system.isProcessing,
// 			activeTabName,
// 		};
// 	},
// 	{
// 		memoizeOptions: {
// 			resultEqualityCheck: _.isEqual,
// 		},
// 	},
// );

export const CanvasRedoButton: FC = () => {
  // const { canRedo, activeTabName } = useAppSelector(canvasRedoSelector);

  const redo = useSetRecoilState(redoAction)
  const canRedo = useRecoilValue(canRedoSelector)

  const handleRedo = useCallback(() => {
    redo()
  }, [])

  useHotkeys(
    ['meta+shift+z', 'ctrl+shift+z', 'control+y', 'meta+y'],
    () => {
      handleRedo()
    },
    {
      enabled: () => canRedo,
      preventDefault: true,
    },
    [canRedo]
  )

  return (
    <IconButton
      aria-label="Redo (Ctrl+Shift+Z)"
      tooltip="Redo (Ctrl+Shift+Z)"
      icon={<FaRedo />}
      onClick={handleRedo}
      isDisabled={!canRedo}
    />
  )
}
