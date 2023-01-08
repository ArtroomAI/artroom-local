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

// import _ from 'lodash';
// import {
// 	canvasSelector,
// 	isStagingSelector,
// } from 'canvas/store/canvasSelectors';
// import {
// 	setIsMovingStage,
// 	setStageCoordinates,
// } from 'canvas/store/canvasSlice';

// const selector = createSelector(
// 	[canvasSelector, isStagingSelector],
// 	(canvas, isStaging) => {
// 		const { tool, isMovingBoundingBox } = canvas;
// 		return {
// 			tool,
// 			isMovingBoundingBox,
// 			isStaging,
// 		};
// 	},
// 	{ memoizeOptions: { resultEqualityCheck: _.isEqual } },
// );

export const useCanvasDragMove = () => {
	// const { tool, isMovingBoundingBox, isStaging } = useAppSelector(selector);

	const setStageCoordinates = useSetRecoilState(stageCoordinatesAtom);
	const setIsMovingStage = useSetRecoilState(isMovingStageAtom);
	const tool = useRecoilValue(toolAtom);
	const isMovingBoundingBox = useRecoilValue(isMovingBoundingBoxAtom);
	const isStaging = useRecoilValue(isStagingSelector);

	return {
		handleDragStart: useCallback(() => {
			if (!((tool === 'move' || isStaging) && !isMovingBoundingBox))
				return;
			setIsMovingStage(true);
		}, [isMovingBoundingBox, isStaging, tool]),

		handleDragMove: useCallback(
			(e: KonvaEventObject<MouseEvent>) => {
				if (!((tool === 'move' || isStaging) && !isMovingBoundingBox))
					return;

				const newCoordinates = { x: e.target.x(), y: e.target.y() };

				setStageCoordinates(newCoordinates);
			},
			[isMovingBoundingBox, isStaging, tool],
		),

		handleDragEnd: useCallback(() => {
			if (!((tool === 'move' || isStaging) && !isMovingBoundingBox))
				return;
			setIsMovingStage(false);
		}, [isMovingBoundingBox, isStaging, tool]),
	};
};
