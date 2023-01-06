import Konva from 'konva';
import { Vector2d } from 'konva/lib/types';
import { MutableRefObject, useCallback } from 'react';
import { getScaledCursorPosition } from '../util';
import { useColorPicker } from './useColorUnderCursor';
import { useRecoilValue, useSetRecoilState } from 'recoil';
import {
	cursorPositionAtom,
	addPointToCurrentLineAction,
	toolAtom,
	isDrawingAtom,
	isStagingSelector,
} from '../atoms/canvas.atoms';

// import { activeTabNameSelector } from 'options/store/optionsSelectors';
// import _ from 'lodash';
// import {
// 	canvasSelector,
// 	isStagingSelector,
// } from 'canvas/store/canvasSelectors';
// import {
// 	addPointToCurrentLine,
// 	setCursorPosition,
// } from 'canvas/store/canvasSlice';

// const selector = createSelector(
// 	[activeTabNameSelector, canvasSelector, isStagingSelector],
// 	(activeTabName, canvas, isStaging) => {
// 		const { tool, isDrawing } = canvas;
// 		return {
// 			tool,
// 			isDrawing,
// 			activeTabName,
// 			isStaging,
// 		};
// 	},
// 	{ memoizeOptions: { resultEqualityCheck: _.isEqual } },
// );

export const useCanvasMouseMove = (
	stageRef: MutableRefObject<Konva.Stage | null>,
	didMouseMoveRef: MutableRefObject<boolean>,
	lastCursorPositionRef: MutableRefObject<Vector2d>,
) => {
	// const { isDrawing, tool, isStaging } = useAppSelector(selector);
	const { updateColorUnderCursor } = useColorPicker();
	const tool = useRecoilValue(toolAtom);
	const isDrawing = useRecoilValue(isDrawingAtom);
	const isStaging = useRecoilValue(isStagingSelector);

	const setCursorPosition = useSetRecoilState(cursorPositionAtom);
	const addPointToCurrentLine = useSetRecoilState(
		addPointToCurrentLineAction,
	);

	return useCallback(() => {
		if (!stageRef.current) return;

		const scaledCursorPosition = getScaledCursorPosition(stageRef.current);

		if (!scaledCursorPosition) return;

		setCursorPosition(scaledCursorPosition);

		lastCursorPositionRef.current = scaledCursorPosition;

		if (tool === 'colorPicker') {
			updateColorUnderCursor();
			return;
		}

		if (!isDrawing || tool === 'move' || isStaging) return;

		didMouseMoveRef.current = true;
		addPointToCurrentLine([scaledCursorPosition.x, scaledCursorPosition.y]);
	}, [
		didMouseMoveRef,
		isDrawing,
		isStaging,
		lastCursorPositionRef,
		stageRef,
		tool,
		updateColorUnderCursor,
	]);
};
