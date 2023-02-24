import React from 'react';
import { FC } from 'react';
import { useRecoilValue } from 'recoil';
import { cursorCoordinatesStringSelector } from '../../atoms/canvas.atoms';

// import { canvasSelector } from 'canvas/store/canvasSelectors';
// import { roundToHundred } from 'canvas/util';

// const cursorPositionSelector = createSelector(
// 	[canvasSelector],
// 	canvas => {
// 		const { cursorPosition } = canvas;

// 		const { cursorX, cursorY } = cursorPosition
// 			? { cursorX: cursorPosition.x, cursorY: cursorPosition.y }
// 			: { cursorX: -1, cursorY: -1 };

// 		return {
// 			cursorCoordinatesString: `(${roundToHundred(
// 				cursorX,
// 			)}, ${roundToHundred(cursorY)})`,
// 		};
// 	},
// 	{
// 		memoizeOptions: {
// 			resultEqualityCheck: _.isEqual,
// 		},
// 	},
// );

export const CanvasStatusTextCursorPos: FC = () => {
	// const { cursorCoordinatesString } = useAppSelector(cursorPositionSelector);

	const cursorCoordinatesString = useRecoilValue(
		cursorCoordinatesStringSelector,
	);

	return <div>{`Cursor Position: ${cursorCoordinatesString}`}</div>;
};
