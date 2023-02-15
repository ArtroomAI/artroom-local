import { KonvaEventObject } from 'konva/lib/Node';
import { useCallback } from 'react';
import { useRecoilValue, useSetRecoilState } from 'recoil';
import {
	stageCoordinatesAtom,
	isMovingStageAtom,
	toolAtom,
	isMovingBoundingBoxAtom,
	isStagingSelector,
} from '../atoms/canvas.atoms';

export const useCanvasDragMove = () => {
	const setStageCoordinates = useSetRecoilState(stageCoordinatesAtom);
	const setIsMovingStage = useSetRecoilState(isMovingStageAtom);
	const tool = useRecoilValue(toolAtom);
	const isMovingBoundingBox = useRecoilValue(isMovingBoundingBoxAtom);
	const isStaging = useRecoilValue(isStagingSelector);

	return {
		handleDragStart: useCallback(() => {
			if (
				!(
					(tool === 'moveBoundingBox' || isStaging) &&
					!isMovingBoundingBox
				)
			)
				return;
			setIsMovingStage(true);
		}, [isMovingBoundingBox, isStaging, tool]),

		handleDragMove: useCallback(
			(e: KonvaEventObject<MouseEvent>) => {
				if (
					!(
						(tool === 'moveBoundingBox' || isStaging) &&
						!isMovingBoundingBox
					)
				)
					return;

				const newCoordinates = { x: e.target.x(), y: e.target.y() };

				setStageCoordinates(newCoordinates);
			},
			[isMovingBoundingBox, isStaging, tool],
		),

		handleDragEnd: useCallback(() => {
			if (
				!(
					(tool === 'moveBoundingBox' || isStaging) &&
					!isMovingBoundingBox
				)
			)
				return;
			setIsMovingStage(false);
		}, [isMovingBoundingBox, isStaging, tool]),
	};
};
