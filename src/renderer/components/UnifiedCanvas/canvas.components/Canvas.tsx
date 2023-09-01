import React, { useCallback, useRef, FC } from 'react'
import Konva from 'konva'
import { Layer, Stage } from 'react-konva'
import { CanvasMaskLines } from './CanvasMaskLines'
import { CanvasToolPreview } from './CanvasToolPreview'
import { Vector2d } from 'konva/lib/types'
import { CanvasBoundingBox } from './CanvasToolbar'
import { CanvasMaskCompositer } from './CanvasMaskCompositer'
import {
  useCanvasDragMove,
  useCanvasMouseDown,
  useCanvasMouseMove,
  useCanvasMouseOut,
  useCanvasMouseUp,
  useCanvasWheel,
  useInpaintingCanvasHotkeys,
} from '../hooks'
import { CanvasObjectRenderer } from './CanvasObjectRenderer'
import { CanvasGrid } from './CanvasGrid'
import { CanvasIntermediateImage } from './CanvasIntermediateImage'
import { CanvasStatusText } from './CanvasStatusText'
import { CanvasStagingArea } from './CanvasStagingArea'
import { CanvasStagingAreaToolbar } from './CanvasStagingAreaToolbar'
import { setCanvasBaseLayer, setCanvasStage } from '../util'
import { KonvaEventObject } from 'konva/lib/Node'
import { CanvasBoundingBoxOverlay } from './CanvasBoundingBoxOverlay'
import {
  toolSelector,
  isMaskEnabledAtom,
  shouldShowBoundingBoxAtom,
  shouldShowGridAtom,
  stageCoordinatesAtom,
  stageDimensionsAtom,
  stageScaleAtom,
  shouldShowIntermediatesAtom,
  stageCursorSelector,
  isModifyingBoundingBoxSelector,
  isStagingSelector,
} from '../atoms/canvas.atoms'
import { useRecoilValue } from 'recoil'

export const Canvas: FC = () => {
  useInpaintingCanvasHotkeys()

  const stageRef = useRef<Konva.Stage | null>(null)
  const canvasBaseLayerRef = useRef<Konva.Layer | null>(null)

  const canvasStageRefCallback = useCallback((el: Konva.Stage) => {
    setCanvasStage(el)
    stageRef.current = el
  }, [])

  const canvasBaseLayerRefCallback = useCallback((el: Konva.Layer) => {
    setCanvasBaseLayer(el)
    canvasBaseLayerRef.current = el
  }, [])

  const lastCursorPositionRef = useRef<Vector2d>({ x: 0, y: 0 })

  // Use refs for values that do not affect rendering, other values in redux
  const didMouseMoveRef = useRef<boolean>(false)

  const handleWheel = useCanvasWheel(stageRef)
  const handleMouseDown = useCanvasMouseDown(stageRef)
  const handleMouseUp = useCanvasMouseUp(stageRef, didMouseMoveRef)
  const handleMouseMove = useCanvasMouseMove(stageRef, didMouseMoveRef, lastCursorPositionRef)
  const handleMouseOut = useCanvasMouseOut()
  const { handleDragStart, handleDragMove, handleDragEnd } = useCanvasDragMove()

  const tool = useRecoilValue(toolSelector)
  const isMaskEnabled = useRecoilValue(isMaskEnabledAtom)
  const shouldShowBoundingBox = useRecoilValue(shouldShowBoundingBoxAtom)
  const shouldShowGrid = useRecoilValue(shouldShowGridAtom)
  const stageCoordinates = useRecoilValue(stageCoordinatesAtom)
  const stageDimensions = useRecoilValue(stageDimensionsAtom)
  const stageScale = useRecoilValue(stageScaleAtom)
  const shouldShowIntermediates = useRecoilValue(shouldShowIntermediatesAtom)
  const stageCursor = useRecoilValue(stageCursorSelector)
  const isModifyingBoundingBox = useRecoilValue(isModifyingBoundingBoxSelector)
  const isStaging = useRecoilValue(isStagingSelector)

  return (
    <div className="inpainting-canvas-container">
      <Stage
        tabIndex={-1}
        ref={canvasStageRefCallback}
        className={'inpainting-canvas-stage'}
        style={{
          ...(stageCursor ? { cursor: stageCursor } : {}),
        }}
        x={stageCoordinates.x}
        y={stageCoordinates.y}
        width={stageDimensions.width}
        height={stageDimensions.height}
        scale={{ x: stageScale, y: stageScale }}
        onTouchStart={handleMouseDown}
        onTouchMove={handleMouseMove}
        onTouchEnd={handleMouseUp}
        onMouseDown={handleMouseDown}
        onMouseLeave={handleMouseOut}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onDragStart={handleDragStart}
        onDragMove={handleDragMove}
        onDragEnd={handleDragEnd}
        onContextMenu={(e: KonvaEventObject<MouseEvent>) => e.evt.preventDefault()}
        onWheel={handleWheel}
        draggable={(tool === 'move' || isStaging) && !isModifyingBoundingBox}
      >
        <Layer id={'grid'} visible={shouldShowGrid}>
          <CanvasGrid />
        </Layer>

        <Layer
          id={'base'}
          ref={canvasBaseLayerRefCallback}
          listening={false}
          imageSmoothingEnabled={false}
        >
          <CanvasObjectRenderer />
        </Layer>
        <Layer id={'mask'} visible={isMaskEnabled} listening={false}>
          <CanvasMaskLines visible={true} listening={false} />
          <CanvasMaskCompositer listening={false} />
        </Layer>
        <Layer>
          <CanvasBoundingBoxOverlay />
        </Layer>
        <Layer id="preview" imageSmoothingEnabled={false}>
          {!isStaging && tool !== 'move' && <CanvasToolPreview visible={true} listening={false} />}
          <CanvasStagingArea visible={isStaging} />
          {shouldShowIntermediates && <CanvasIntermediateImage />}
          <CanvasBoundingBox visible={shouldShowBoundingBox && !isStaging} />
        </Layer>
      </Stage>
      <CanvasStatusText />
      <CanvasStagingAreaToolbar />
    </div>
  )
}
