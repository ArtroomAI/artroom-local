import React from 'react'
import { FC } from 'react'
import { GroupConfig } from 'konva/lib/Group'
import { Group, Rect } from 'react-konva'
import { CanvasImage } from './CanvasImage'
import { useRecoilValue } from 'recoil'
import {
  shouldShowStagingImageAtom,
  shouldShowStagingOutlineAtom,
  layerStateAtom,
  boundingBoxDimensionsAtom,
  boundingBoxCoordinatesAtom,
} from '../atoms/canvas.atoms'

// import _ from 'lodash';
// import { canvasSelector } from 'canvas/store/canvasSelectors';

// const selector = createSelector(
// 	[canvasSelector],
// 	canvas => {
// 		const {
// 			layerState: {
// 				stagingArea: { images, selectedImageIndex },
// 			},
// 			shouldShowStagingImage,
// 			shouldShowStagingOutline,
// 			boundingBoxCoordinates: { x, y },
// 			boundingBoxDimensions: { width, height },
// 		} = canvas;

// 		return {
// 			currentStagingAreaImage:
// 				images.length > 0 ? images[selectedImageIndex] : undefined,
// 			isOnFirstImage: selectedImageIndex === 0,
// 			isOnLastImage: selectedImageIndex === images.length - 1,
// 			shouldShowStagingImage,
// 			shouldShowStagingOutline,
// 			x,
// 			y,
// 			width,
// 			height,
// 		};
// 	},
// 	{
// 		memoizeOptions: {
// 			resultEqualityCheck: _.isEqual,
// 		},
// 	},
// );

type Props = GroupConfig

export const CanvasStagingArea: FC<Props> = (props) => {
  const { ...rest } = props
  // const {
  // 	currentStagingAreaImage,
  // 	shouldShowStagingImage,
  // 	shouldShowStagingOutline,
  // 	x,
  // 	y,
  // 	width,
  // 	height,
  // } = useAppSelector(selector);

  const shouldShowStagingImage = useRecoilValue(shouldShowStagingImageAtom)
  const shouldShowStagingOutline = useRecoilValue(shouldShowStagingOutlineAtom)
  const boundingBoxDimensions = useRecoilValue(boundingBoxDimensionsAtom)
  const boundingBoxCoordinates = useRecoilValue(boundingBoxCoordinatesAtom)
  const { width, height } = boundingBoxDimensions
  const { x, y } = boundingBoxCoordinates
  const layerState = useRecoilValue(layerStateAtom)
  const {
    stagingArea: { images, selectedImageIndex },
  } = layerState
  const currentStagingAreaImage = images.length > 0 ? images[selectedImageIndex] : undefined

  return (
    <Group {...rest}>
      {shouldShowStagingImage && currentStagingAreaImage && (
        <CanvasImage url={currentStagingAreaImage.image.url} x={x} y={y} />
      )}
      {shouldShowStagingOutline && (
        <Group>
          <Rect
            x={x}
            y={y}
            width={width}
            height={height}
            strokeWidth={1}
            stroke={'white'}
            strokeScaleEnabled={false}
          />
          <Rect
            x={x}
            y={y}
            width={width}
            height={height}
            dash={[4, 4]}
            strokeWidth={1}
            stroke={'black'}
            strokeScaleEnabled={false}
          />
        </Group>
      )}
    </Group>
  )
}
