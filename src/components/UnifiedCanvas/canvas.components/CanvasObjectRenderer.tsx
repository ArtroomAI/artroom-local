import React from 'react';
import { FC } from 'react';
import { Group, Line, Rect } from 'react-konva';
import {
	isCanvasBaseImage,
	isCanvasBaseLine,
	isCanvasEraseRect,
	isCanvasFillRect,
} from '../atoms/canvasTypes';
import {CanvasImage} from './CanvasImage';
import { rgbaColorToString } from '../util';
import { useRecoilValue } from 'recoil';
import { layerStateAtom } from '../atoms/canvas.atoms';

// import _ from 'lodash';
// import {canvasSelector} from 'canvas/store/canvasSelectors';

// const selector = createSelector(
// 	[canvasSelector],
// 	(canvas) => {
// 		const {
// 			layerState: {objects},
// 		} = canvas;
// 		return {
// 			objects,
// 		};
// 	},
// 	{
// 		memoizeOptions: {
// 			resultEqualityCheck: _.isEqual,
// 		},
// 	}
// );

export const CanvasObjectRenderer: FC = () => {
	const layerState = useRecoilValue(layerStateAtom);
	const { objects } = layerState;
	// const { objects } = useAppSelector(selector);

	if (!objects) return null;

	return (
		<Group name="outpainting-objects" listening={false}>
			{objects.map((obj, i) => {
				if (isCanvasBaseImage(obj)) {
					console.log(obj)
					return (
						<CanvasImage
							key={i}
							x={obj.x}
							y={obj.y}
							url={obj.image.url}
							width={obj.width}
							height={obj.height}
						/>
					);
				} else if (isCanvasBaseLine(obj)) {
					const line = (
						<Line
							key={i}
							points={obj.points}
							// stroke="green"
							stroke={
								obj.color
									? rgbaColorToString(obj.color)
									: 'rgb(0,0,0)'
							} // The lines can be any color, just need alpha > 0
							strokeWidth={obj.strokeWidth * 2}
							tension={0}
							lineCap="round"
							lineJoin="round"
							shadowForStrokeEnabled={false}
							listening={false}
							globalCompositeOperation={
								obj.tool === 'brush'
									? 'source-over'
									: 'destination-out'
							}
						/>
					);
					if (obj.clip) {
						return (
							<Group
								key={i}
								clipX={obj.clip.x}
								clipY={obj.clip.y}
								clipWidth={obj.clip.width}
								clipHeight={obj.clip.height}>
								{line}
							</Group>
						);
					} else {
						return line;
					}
				} else if (isCanvasFillRect(obj)) {
					return (
						<Rect
							key={i}
							x={obj.x}
							y={obj.y}
							width={obj.width}
							height={obj.height}
							fill={rgbaColorToString(obj.color)}
						/>
					);
				} else if (isCanvasEraseRect(obj)) {
					return (
						<Rect
							key={i}
							x={obj.x}
							y={obj.y}
							width={obj.width}
							height={obj.height}
							fill={'rgb(255, 255, 255)'}
							globalCompositeOperation={'destination-out'}
						/>
					);
				}
			})}
		</Group>
	);
};
