import { FC } from 'react';
import { GroupConfig } from 'konva/lib/Group';
import { Group, Rect } from 'react-konva';
import { CanvasImage } from './CanvasImage';
import { useRecoilValue } from 'recoil';
import {
	shouldShowStagingImageAtom,
	shouldShowStagingOutlineAtom,
	layerStateAtom,
	boundingBoxDimensionsAtom,
	boundingBoxCoordinatesAtom,
} from '../atoms/canvas.atoms';

type Props = GroupConfig;

export const CanvasStagingArea: FC<Props> = props => {
	const { ...rest } = props;

	const shouldShowStagingImage = useRecoilValue(shouldShowStagingImageAtom);
	const shouldShowStagingOutline = useRecoilValue(
		shouldShowStagingOutlineAtom,
	);
	const boundingBoxDimensions = useRecoilValue(boundingBoxDimensionsAtom);
	const boundingBoxCoordinates = useRecoilValue(boundingBoxCoordinatesAtom);
	const { width, height } = boundingBoxDimensions;
	const { x, y } = boundingBoxCoordinates;
	const layerState = useRecoilValue(layerStateAtom);
	const {
		stagingArea: { images, selectedImageIndex },
	} = layerState;
	const currentStagingAreaImage =
		images.length > 0 ? images[selectedImageIndex] : undefined;

	return (
		<Group {...rest}>
			{shouldShowStagingImage && currentStagingAreaImage && (
				<CanvasImage
					// @ts-ignore
					url={currentStagingAreaImage.image.url}
					x={x}
					y={y}
				/>
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
	);
};
