import { FC } from 'react';
import { Group, Rect } from 'react-konva';
import { useRecoilValue } from 'recoil';
import {
	stageScaleAtom,
	stageDimensionsAtom,
	stageCoordinatesAtom,
	shouldDarkenOutsideBoundingBoxAtom,
	boundingBoxDimensionsAtom,
	boundingBoxCoordinatesAtom,
} from '../atoms/canvas.atoms';

export const CanvasBoundingBoxOverlay: FC = () => {
	const stageScale = useRecoilValue(stageScaleAtom);
	const stageDimensions = useRecoilValue(stageDimensionsAtom);
	const stageCoordinates = useRecoilValue(stageCoordinatesAtom);
	const shouldDarkenOutsideBoundingBox = useRecoilValue(
		shouldDarkenOutsideBoundingBoxAtom,
	);
	const boundingBoxDimensions = useRecoilValue(boundingBoxDimensionsAtom);
	const boundingBoxCoordinates = useRecoilValue(boundingBoxCoordinatesAtom);

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
	);
};
