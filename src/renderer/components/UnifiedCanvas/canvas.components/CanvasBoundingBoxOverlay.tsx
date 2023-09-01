import React from 'react'
import { FC } from 'react'
import { Group, Rect } from 'react-konva'
import { useRecoilValue } from 'recoil'
import {
  stageScaleAtom,
  stageDimensionsAtom,
  stageCoordinatesAtom,
  shouldDarkenOutsideBoundingBoxAtom,
  boundingBoxDimensionsAtom,
  boundingBoxCoordinatesAtom,
} from '../atoms/canvas.atoms'

// import _ from 'lodash';
// import { canvasSelector } from '../store/canvasSelectors';

// const selector = createSelector(
// 	canvasSelector,
// 	canvas => {
// 		const {
// 			boundingBoxCoordinates,
// 			boundingBoxDimensions,
// 			stageDimensions,
// 			stageScale,
// 			shouldDarkenOutsideBoundingBox,
// 			stageCoordinates,
// 		} = canvas;

// 		return {
// 			boundingBoxCoordinates,
// 			boundingBoxDimensions,
// 			shouldDarkenOutsideBoundingBox,
// 			stageCoordinates,
// 			stageDimensions,
// 			stageScale,
// 		};
// 	},
// 	{
// 		memoizeOptions: {
// 			resultEqualityCheck: _.isEqual,
// 		},
// 	},
// );

export const CanvasBoundingBoxOverlay: FC = () => {
  // const {
  // 	boundingBoxCoordinates,
  // 	boundingBoxDimensions,
  // 	shouldDarkenOutsideBoundingBox,
  // 	stageCoordinates,
  // 	stageDimensions,
  // 	stageScale,
  // } = useAppSelector(selector);

  const stageScale = useRecoilValue(stageScaleAtom)
  const stageDimensions = useRecoilValue(stageDimensionsAtom)
  const stageCoordinates = useRecoilValue(stageCoordinatesAtom)
  const shouldDarkenOutsideBoundingBox = useRecoilValue(shouldDarkenOutsideBoundingBoxAtom)
  const boundingBoxDimensions = useRecoilValue(boundingBoxDimensionsAtom)
  const boundingBoxCoordinates = useRecoilValue(boundingBoxCoordinatesAtom)

  return (
    <Group>
      <Rect
        offsetX={stageCoordinates.x / stageScale}
        offsetY={stageCoordinates.y / stageScale}
        height={stageDimensions.height / stageScale}
        width={stageDimensions.width / stageScale}
        fill={'rgba(0,0,0,0.4)'}
        listening={false}
        visible={shouldDarkenOutsideBoundingBox}
      />
      <Rect
        x={boundingBoxCoordinates.x}
        y={boundingBoxCoordinates.y}
        width={boundingBoxDimensions.width}
        height={boundingBoxDimensions.height}
        fill={'rgb(255,255,255)'}
        listening={false}
        visible={shouldDarkenOutsideBoundingBox}
        globalCompositeOperation={'destination-out'}
      />
    </Group>
  )
}
