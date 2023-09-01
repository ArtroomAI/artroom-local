import Konva from 'konva'
import { KonvaEventObject } from 'konva/lib/Node'
import _ from 'lodash'
import { MutableRefObject, useCallback } from 'react'
import { CANVAS_SCALE_BY, MAX_CANVAS_SCALE, MIN_CANVAS_SCALE } from '../util'
import { useSetRecoilState, useRecoilState, useRecoilValue } from 'recoil'
import { stageCoordinatesAtom, stageScaleAtom, isMoveStageKeyHeldAtom } from '../atoms/canvas.atoms'

export const useCanvasWheel = (stageRef: MutableRefObject<Konva.Stage | null>) => {
  const setStageCoordinates = useSetRecoilState(stageCoordinatesAtom)
  const [stageScale, setStageScale] = useRecoilState(stageScaleAtom)
  const isMoveStageKeyHeld = useRecoilValue(isMoveStageKeyHeldAtom)

  return useCallback(
    (e: KonvaEventObject<WheelEvent>) => {
      // stop default scrolling
      if (!stageRef.current || isMoveStageKeyHeld) return

      e.evt.preventDefault()

      const cursorPos = stageRef.current.getPointerPosition()

      if (!cursorPos) return

      const mousePointTo = {
        x: (cursorPos.x - stageRef.current.x()) / stageScale,
        y: (cursorPos.y - stageRef.current.y()) / stageScale,
      }

      let delta = e.evt.deltaY

      // when we zoom on trackpad, e.evt.ctrlKey is true
      // in that case lets revert direction
      if (e.evt.ctrlKey) {
        delta = -delta
      }

      const newScale = _.clamp(
        stageScale * CANVAS_SCALE_BY ** delta,
        MIN_CANVAS_SCALE,
        MAX_CANVAS_SCALE
      )

      const newCoordinates = {
        x: cursorPos.x - mousePointTo.x * newScale,
        y: cursorPos.y - mousePointTo.y * newScale,
      }

      setStageScale(newScale)
      setStageCoordinates(newCoordinates)
    },
    [stageRef, isMoveStageKeyHeld, stageScale]
  )
}
