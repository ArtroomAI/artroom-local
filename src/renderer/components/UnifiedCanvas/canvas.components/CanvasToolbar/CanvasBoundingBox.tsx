import React from 'react';
import { FC, useCallback, useEffect, useRef, useState } from 'react';
import Konva from 'konva';
import { KonvaEventObject } from 'konva/lib/Node';
import { Vector2d } from 'konva/lib/types';
import { Group, Rect, Transformer } from 'react-konva';
import { GroupConfig } from 'konva/lib/Group';
import { useRecoilState, useRecoilValue, useSetRecoilState } from 'recoil';
import { roundDownToMultiple, roundToMultiple } from '../../util';
import {
  setBoundingBoxDimensionsAction,
  setBoundingBoxCoordinatesAction,
  isTransformingBoundingBoxAtom,
  isMovingBoundingBoxAtom,
  isMouseOverBoundingBoxAtom,
  stageScaleAtom,
  isDrawingAtom,
  toolAtom,
  shouldSnapToGridAtom,
} from '../../atoms/canvas.atoms';

// import _ from 'lodash';
// import { canvasSelector } from 'canvas/store/canvasSelectors';
// import {
// 	setBoundingBoxCoordinates,
// 	setBoundingBoxDimensions,
// 	setIsMouseOverBoundingBox,
// 	setIsMovingBoundingBox,
// 	setIsTransformingBoundingBox,
// } from 'canvas/store/canvasSlice';

// const boundingBoxPreviewSelector = createSelector(
// 	canvasSelector,
// 	canvas => {
// 		const {
// 			boundingBoxCoordinates,
// 			boundingBoxDimensions,
// 			stageScale,
// 			isDrawing,
// 			isTransformingBoundingBox,
// 			isMovingBoundingBox,
// 			tool,
// 			shouldSnapToGrid,
// 		} = canvas;
// 		return {
// 			boundingBoxCoordinates,
// 			boundingBoxDimensions,
// 			isDrawing,
// 			isMovingBoundingBox,
// 			isTransformingBoundingBox,
// 			stageScale,
// 			shouldSnapToGrid,
// 			tool,
// 			hitStrokeWidth: 20 / stageScale,
// 		};
// 	},
// 	{
// 		memoizeOptions: {
// 			resultEqualityCheck: _.isEqual,
// 		},
// 	},
// );

type ICanvasBoundingBoxPreviewProps = GroupConfig;

export const CanvasBoundingBox: FC<ICanvasBoundingBoxPreviewProps> = (
  props
) => {
  const { ...rest } = props;

  // const {
  // 	boundingBoxCoordinates,
  // 	boundingBoxDimensions,
  // 	isDrawing,
  // 	isMovingBoundingBox,
  // 	isTransformingBoundingBox,
  // 	stageScale,
  // 	shouldSnapToGrid,
  // 	tool,
  // 	hitStrokeWidth,
  // } = useAppSelector(boundingBoxPreviewSelector);

  const [boundingBoxDimensions, setBoundingBoxDimensions] = useRecoilState(
    setBoundingBoxDimensionsAction
  );
  const [boundingBoxCoordinates, setBoundingBoxCoordinates] = useRecoilState(
    setBoundingBoxCoordinatesAction
  );
  const [isTransformingBoundingBox, setIsTransformingBoundingBox] =
    useRecoilState(isTransformingBoundingBoxAtom);
  const [isMovingBoundingBox, setIsMovingBoundingBox] = useRecoilState(
    isMovingBoundingBoxAtom
  );
  const setIsMouseOverBoundingBox = useSetRecoilState(
    isMouseOverBoundingBoxAtom
  );
  const stageScale = useRecoilValue(stageScaleAtom);
  const isDrawing = useRecoilValue(isDrawingAtom);
  const tool = useRecoilValue(toolAtom);
  const shouldSnapToGrid = useRecoilValue(shouldSnapToGridAtom);

  const transformerRef = useRef<Konva.Transformer>(null);
  const shapeRef = useRef<Konva.Rect>(null);

  const [isMouseOverBoundingBoxOutline, setIsMouseOverBoundingBoxOutline] =
    useState(false);

  useEffect(() => {
    if (!transformerRef.current || !shapeRef.current) return;
    transformerRef.current.nodes([shapeRef.current]);
    transformerRef.current.getLayer()?.batchDraw();
  }, []);

  const scaledStep = 64 * stageScale;

  const handleOnDragMove = useCallback(
    (e: KonvaEventObject<DragEvent>) => {
      if (!shouldSnapToGrid) {
        setBoundingBoxCoordinates({
          x: Math.floor(e.target.x()),
          y: Math.floor(e.target.y()),
        });
        return;
      }

      const dragX = e.target.x();
      const dragY = e.target.y();

      const newX = roundToMultiple(dragX, 64);
      const newY = roundToMultiple(dragY, 64);

      e.target.x(newX);
      e.target.y(newY);

      setBoundingBoxCoordinates({
        x: newX,
        y: newY,
      });
    },
    [shouldSnapToGrid]
  );

  const handleOnTransform = useCallback(() => {
    /**
     * The Konva Transformer changes the object's anchor point and scale factor,
     * not its width and height. We need to un-scale the width and height before
     * setting the values.
     */
    if (!shapeRef.current) return;

    const rect = shapeRef.current;

    const scaleX = rect.scaleX();
    const scaleY = rect.scaleY();

    // undo the scaling
    const width = Math.round(rect.width() * scaleX);
    const height = Math.round(rect.height() * scaleY);

    const x = Math.round(rect.x());
    const y = Math.round(rect.y());

    setBoundingBoxDimensions({
      width,
      height,
    });

    setBoundingBoxCoordinates({
      x: shouldSnapToGrid ? roundDownToMultiple(x, 64) : x,
      y: shouldSnapToGrid ? roundDownToMultiple(y, 64) : y,
    });

    // Reset the scale now that the coords/dimensions have been un-scaled
    rect.scaleX(1);
    rect.scaleY(1);
  }, [shouldSnapToGrid]);

  const anchorDragBoundFunc = useCallback(
    (
      oldPos: Vector2d, // old absolute position of anchor point
      newPos: Vector2d, // new absolute position (potentially) of anchor point
      // eslint-disable-next-line @typescript-eslint/no-unused-vars
      _e: MouseEvent
    ) => {
      /**
       * Konva does not transform with width or height. It transforms the anchor point
       * and scale factor. This is then sent to the shape's onTransform listeners.
       *
       * We need to snap the new dimensions to steps of 64. But because the whole
       * stage is scaled, our actual desired step is actually 64 * the stage scale.
       *
       * Additionally, we need to ensure we offset the position so that we snap to a
       * multiple of 64 that is aligned with the grid, and not from the absolute zero
       * coordinate.
       */

      // Calculate the offset of the grid.
      const offsetX = oldPos.x % scaledStep;
      const offsetY = oldPos.y % scaledStep;

      const newCoordinates = {
        x: roundDownToMultiple(newPos.x, scaledStep) + offsetX,
        y: roundDownToMultiple(newPos.y, scaledStep) + offsetY,
      };

      return newCoordinates;
    },
    [scaledStep]
  );

  const handleStartedTransforming = () => {
    setIsTransformingBoundingBox(true);
  };

  const handleEndedTransforming = () => {
    setIsTransformingBoundingBox(false);
    setIsMovingBoundingBox(false);
    setIsMouseOverBoundingBox(false);
    setIsMouseOverBoundingBoxOutline(false);
  };

  const handleStartedMoving = () => {
    setIsMovingBoundingBox(true);
  };

  const handleEndedModifying = () => {
    setIsTransformingBoundingBox(false);
    setIsMovingBoundingBox(false);
    setIsMouseOverBoundingBox(false);
    setIsMouseOverBoundingBoxOutline(false);
  };

  const handleMouseOver = () => {
    setIsMouseOverBoundingBoxOutline(true);
  };

  const handleMouseOut = () => {
    !isTransformingBoundingBox &&
      !isMovingBoundingBox &&
      setIsMouseOverBoundingBoxOutline(false);
  };

  const handleMouseEnterBoundingBox = () => {
    setIsMouseOverBoundingBox(true);
  };

  const handleMouseLeaveBoundingBox = () => {
    setIsMouseOverBoundingBox(false);
  };

  return (
    <Group {...rest}>
      <Rect
        height={boundingBoxDimensions.height}
        width={boundingBoxDimensions.width}
        x={boundingBoxCoordinates.x}
        y={boundingBoxCoordinates.y}
        onMouseEnter={handleMouseEnterBoundingBox}
        onMouseOver={handleMouseEnterBoundingBox}
        onMouseLeave={handleMouseLeaveBoundingBox}
        onMouseOut={handleMouseLeaveBoundingBox}
      />
      <Rect
        draggable
        fillEnabled={false}
        height={boundingBoxDimensions.height}
        hitStrokeWidth={20 / stageScale}
        listening={!isDrawing && tool === 'move'}
        onDragStart={handleStartedMoving}
        onDragEnd={handleEndedModifying}
        onDragMove={handleOnDragMove}
        onMouseDown={handleStartedMoving}
        onMouseOut={handleMouseOut}
        onMouseOver={handleMouseOver}
        onMouseEnter={handleMouseOver}
        onMouseUp={handleEndedModifying}
        onTransform={handleOnTransform}
        onTransformEnd={handleEndedTransforming}
        ref={shapeRef}
        stroke={
          isMouseOverBoundingBoxOutline ? 'rgba(255,255,255,0.7)' : 'white'
        }
        strokeWidth={(isMouseOverBoundingBoxOutline ? 8 : 1) / stageScale}
        width={boundingBoxDimensions.width}
        x={boundingBoxCoordinates.x}
        y={boundingBoxCoordinates.y}
      />
      <Transformer
        anchorCornerRadius={3}
        anchorDragBoundFunc={anchorDragBoundFunc}
        anchorFill="rgba(212,216,234,1)"
        anchorSize={15}
        anchorStroke="rgb(42,42,42)"
        borderDash={[4, 4]}
        borderEnabled
        borderStroke="black"
        draggable={false}
        enabledAnchors={tool === 'move' ? undefined : []}
        flipEnabled={false}
        ignoreStroke
        keepRatio={false}
        listening={!isDrawing && tool === 'move'}
        onDragStart={handleStartedMoving}
        onDragEnd={handleEndedModifying}
        onMouseDown={handleStartedTransforming}
        onMouseUp={handleEndedTransforming}
        onTransformEnd={handleEndedTransforming}
        ref={transformerRef}
        rotateEnabled={false}
      />
    </Group>
  );
};
