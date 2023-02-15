import { FC } from 'react';
import { Group, Line, Rect } from 'react-konva';
import {
	isCanvasBaseLine,
	isCanvasEraseRect,
	isCanvasFillRect,
} from '../atoms/canvasTypes';
import { rgbaColorToString } from '../util';
import { useRecoilValue } from 'recoil';
import { layerStateAtom } from '../atoms/canvas.atoms';
import { CanvasLayerRenderer } from './CanvasLayerRenderer';

export const CanvasObjectRenderer: FC = () => {
	const layerState = useRecoilValue(layerStateAtom);

	const { objects, images } = layerState;

	if (!objects) return null;

	return (
		<>
			<Group id="outpainting-objects">
				{objects.map((obj, i) => {
					if (isCanvasBaseLine(obj)) {
						const line = (
							<Line
								key={i}
								points={obj.points}
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
			{images.map(imageData => (
				<CanvasLayerRenderer
					key={imageData.id}
					imageData={imageData}
					objects={objects}
				/>
			))}
		</>
	);
};
