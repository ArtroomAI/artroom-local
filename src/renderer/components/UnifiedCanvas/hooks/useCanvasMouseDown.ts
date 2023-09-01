import Konva from 'konva'
import { KonvaEventObject } from 'konva/lib/Node'
import _ from 'lodash'
import { MutableRefObject, useCallback } from 'react'
import { getScaledCursorPosition } from '../util'
import { useColorPicker } from './useColorUnderCursor'
import {
  addLineAction,
  isDrawingAtom,
  isMovingStageAtom,
  toolAtom,
  isStagingSelector,
} from '../atoms/canvas.atoms'
import { useRecoilValue, useSetRecoilState } from 'recoil'

// import { activeTabNameSelector } from 'options/store/optionsSelectors';
// import {
// 	canvasSelector,
// 	isStagingSelector,
// } from 'canvas/store/canvasSelectors';
// import {
// 	addLine,
// 	setIsDrawing,
// 	setIsMovingStage,
// } from 'canvas/store/canvasSlice';

// const selector = createSelector(
// 	[activeTabNameSelector, canvasSelector, isStagingSelector],
// 	(activeTabName, canvas, isStaging) => {
// 		const { tool } = canvas;
// 		return {
// 			tool,
// 			activeTabName,
// 			isStaging,
// 		};
// 	},
// 	{ memoizeOptions: { resultEqualityCheck: _.isEqual } },
// );

export const useCanvasMouseDown = (stageRef: MutableRefObject<Konva.Stage | null>) => {
  // const { tool, isStaging } = useAppSelector(selector);
  const { commitColorUnderCursor } = useColorPicker()

  const addLine = useSetRecoilState(addLineAction)
  const setIsDrawing = useSetRecoilState(isDrawingAtom)
  const setIsMovingStage = useSetRecoilState(isMovingStageAtom)
  const tool = useRecoilValue(toolAtom)
  const isStaging = useRecoilValue(isStagingSelector)

  return useCallback(
    (e: KonvaEventObject<MouseEvent | TouchEvent>) => {
      if (!stageRef.current) return

      stageRef.current.container().focus()

      e.evt.preventDefault()

      if (e.evt.button === 1) {
        stageRef.current.startDrag()
        return
      }

      if (tool === 'move' || isStaging) {
        setIsMovingStage(true)
        return
      }

      if (tool === 'colorPicker') {
        commitColorUnderCursor()
        return
      }

      const scaledCursorPosition = getScaledCursorPosition(stageRef.current)

      if (!scaledCursorPosition) return

      setIsDrawing(true)

      // Add a new line starting from the current cursor position.
      addLine([scaledCursorPosition.x, scaledCursorPosition.y])
    },
    [stageRef, tool, isStaging, commitColorUnderCursor]
  )
}
