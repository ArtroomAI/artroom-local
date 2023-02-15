import { FC } from 'react';
import { GroupConfig } from 'konva/lib/Group';
import { Circle, Group } from 'react-konva';
import {
	// COLOR_PICKER_SIZE,
	COLOR_PICKER_STROKE_RADIUS,
} from '../util';
import { useRecoilValue } from 'recoil';
import {
	brushColorStringSelector,
	toolAtom,
	layerAtom,
	brushXSelector,
	brushYSelector,
	radiusSelector,
	shouldDrawBrushPreviewSelector,
	clipSelector,
	dotRadiusSelector,
	strokeWidthSelector,
	colorPickerColorStringSelector,
	colorPickerOuterRadiusSelector,
	colorPickerInnerRadiusSelector,
	maskColorStringHalfAlphaSelector,
} from '../atoms/canvas.atoms';

/**
 * Draws a black circle around the canvas brush preview.
 */
export const CanvasToolPreview: FC<GroupConfig> = props => {
	const { ...rest } = props;

	const brushColorString = useRecoilValue(brushColorStringSelector);
	const tool = useRecoilValue(toolAtom);
	const layer = useRecoilValue(layerAtom);
	const brushX = useRecoilValue(brushXSelector);
	const brushY = useRecoilValue(brushYSelector);
	const radius = useRecoilValue(radiusSelector);
	const shouldDrawBrushPreview = useRecoilValue(
		shouldDrawBrushPreviewSelector,
	);
	const clip = useRecoilValue(clipSelector);
	const dotRadius = useRecoilValue(dotRadiusSelector);
	const strokeWidth = useRecoilValue(strokeWidthSelector);
	const colorPickerColorString = useRecoilValue(
		colorPickerColorStringSelector,
	);
	const colorPickerOuterRadius = useRecoilValue(
		colorPickerOuterRadiusSelector,
	);
	const colorPickerInnerRadius = useRecoilValue(
		colorPickerInnerRadiusSelector,
	);
	const maskColorString = useRecoilValue(maskColorStringHalfAlphaSelector);

	if (!shouldDrawBrushPreview) return null;

	return (
		<Group listening={false} {...clip} {...rest}>
			{tool === 'colorPicker' ? (
				<>
					<Circle
						x={brushX}
						y={brushY}
						radius={colorPickerOuterRadius}
						stroke={brushColorString}
						strokeWidth={COLOR_PICKER_STROKE_RADIUS}
						strokeScaleEnabled={false}
					/>
					<Circle
						x={brushX}
						y={brushY}
						radius={colorPickerInnerRadius}
						stroke={colorPickerColorString}
						strokeWidth={COLOR_PICKER_STROKE_RADIUS}
						strokeScaleEnabled={false}
					/>
				</>
			) : (
				<>
					<Circle
						x={brushX}
						y={brushY}
						radius={radius}
						fill={
							layer === 'mask'
								? maskColorString
								: brushColorString
						}
						globalCompositeOperation={
							tool === 'eraser' ? 'destination-out' : 'source-out'
						}
					/>
					<Circle
						x={brushX}
						y={brushY}
						radius={radius}
						stroke={'rgba(255,255,255,0.4)'}
						strokeWidth={strokeWidth * 2}
						strokeEnabled={true}
						listening={false}
					/>
					<Circle
						x={brushX}
						y={brushY}
						radius={radius}
						stroke={'rgba(0,0,0,1)'}
						strokeWidth={strokeWidth}
						strokeEnabled={true}
						listening={false}
					/>
				</>
			)}
			<Circle
				x={brushX}
				y={brushY}
				radius={dotRadius * 2}
				fill={'rgba(255,255,255,0.4)'}
				listening={false}
			/>
			<Circle
				x={brushX}
				y={brushY}
				radius={dotRadius}
				fill={'rgba(0,0,0,1)'}
				listening={false}
			/>
		</Group>
	);
};
