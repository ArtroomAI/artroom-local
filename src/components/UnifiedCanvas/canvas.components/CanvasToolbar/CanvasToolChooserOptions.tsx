import { FC } from 'react';
import { ButtonGroup, Flex } from '@chakra-ui/react';
import {
	FaEraser,
	FaEyeDropper,
	FaFillDrip,
	FaPaintBrush,
	FaPlus,
	FaSlidersH,
} from 'react-icons/fa';
import { useHotkeys } from 'react-hotkeys-hook';
import { useRecoilState, useRecoilValue, useSetRecoilState } from 'recoil';
import { Popover, Slider, IconButton } from '../../components';
import {
	toolSelector,
	brushColorAtom,
	brushSizeAtom,
	addFillRectAction,
	addEraseRectAction,
	isStagingSelector,
} from '../../atoms/canvas.atoms';
import { RgbaColorPicker } from 'react-colorful';

export const CanvasToolChooserOptions: FC = () => {
	const [tool, setTool] = useRecoilState(toolSelector);
	const [brushColor, setBrushColor] = useRecoilState(brushColorAtom);
	const [brushSize, setBrushSize] = useRecoilState(brushSizeAtom);
	const addFillRect = useSetRecoilState(addFillRectAction);
	const addEraseRect = useSetRecoilState(addEraseRectAction);
	const isStaging = useRecoilValue(isStagingSelector);

	useHotkeys(
		['b'],
		() => {
			handleSelectBrushTool();
		},
		{
			enabled: () => !isStaging,
			preventDefault: true,
		},
		[],
	);

	useHotkeys(
		['e'],
		() => {
			handleSelectEraserTool();
		},
		{
			enabled: () => !isStaging,
			preventDefault: true,
		},
		[tool],
	);

	useHotkeys(
		['c'],
		() => {
			handleSelectColorPickerTool();
		},
		{
			enabled: () => !isStaging,
			preventDefault: true,
		},
		[tool],
	);

	useHotkeys(
		['shift+f'],
		() => {
			handleFillRect();
		},
		{
			enabled: () => !isStaging,
			preventDefault: true,
		},
	);

	useHotkeys(
		['delete', 'backspace'],
		() => {
			handleEraseBoundingBox();
		},
		{
			enabled: () => !isStaging,
			preventDefault: true,
		},
	);

	useHotkeys(
		['BracketLeft'],
		() => {
			setBrushSize(Math.max(brushSize - 5, 5));
		},
		{
			enabled: () => !isStaging,
			preventDefault: true,
		},
		[brushSize],
	);

	useHotkeys(
		['BracketRight'],
		() => {
			setBrushSize(Math.min(brushSize + 5, 500));
		},
		{
			enabled: () => !isStaging,
			preventDefault: true,
		},
		[brushSize],
	);

	useHotkeys(
		['shift+BracketLeft'],
		() => {
			setBrushSize(Math.max(brushSize - 20, 5));
		},
		{
			enabled: () => !isStaging,
			preventDefault: true,
		},
		[brushSize],
	);

	useHotkeys(
		['shift+BracketRight'],
		() => {
			setBrushSize(Math.min(brushSize + 20, 500));
		},
		{
			enabled: () => !isStaging,
			preventDefault: true,
		},
		[brushSize],
	);

	const handleSelectBrushTool = () => setTool('brush');
	const handleSelectEraserTool = () => setTool('eraser');
	const handleSelectColorPickerTool = () => setTool('colorPicker');
	const handleFillRect = () => addFillRect();
	const handleEraseBoundingBox = () => addEraseRect();

	return (
		<ButtonGroup isAttached>
			<IconButton
				aria-label="Brush Tool (B)"
				tooltip="Brush Tool (B)"
				icon={<FaPaintBrush />}
				aria-selected={tool === 'brush' && !isStaging}
				onClick={handleSelectBrushTool}
				isDisabled={isStaging}
			/>
			<IconButton
				aria-label="Eraser Tool (E)"
				tooltip="Eraser Tool (E)"
				icon={<FaEraser />}
				aria-selected={tool === 'eraser' && !isStaging}
				isDisabled={isStaging}
				onClick={handleSelectEraserTool}
			/>
			<Popover
				trigger="hover"
				triggerComponent={
					<IconButton
						aria-label="Brush Options"
						tooltip="Brush Options"
						icon={<FaSlidersH />}
					/>
				}>
				<Flex
					minWidth="15rem"
					direction="column"
					gap="1rem"
					width="100%">
					<Flex gap="1rem" justifyContent="space-between">
						<Slider
							label="Size"
							value={brushSize}
							withInput
							onChange={newSize => setBrushSize(newSize)}
							sliderNumberInputProps={{ max: 500 }}
							inputReadOnly={false}
						/>
					</Flex>
					<RgbaColorPicker
						style={{
							width: '100%',
							paddingTop: '0.5rem',
							paddingBottom: '0.5rem',
						}}
						color={brushColor}
						onChange={newColor => setBrushColor(newColor)}
					/>
				</Flex>
			</Popover>
			<IconButton
				aria-label="Fill Bounding Box (Shift+F)"
				tooltip="Fill Bounding Box (Shift+F)"
				icon={<FaFillDrip />}
				isDisabled={isStaging}
				onClick={handleFillRect}
			/>
			<IconButton
				aria-label="Erase Bounding Box Area (Delete/Backspace)"
				tooltip="Erase Bounding Box Area (Delete/Backspace)"
				icon={<FaPlus style={{ transform: 'rotate(45deg)' }} />}
				isDisabled={isStaging}
				onClick={handleEraseBoundingBox}
			/>
			<IconButton
				aria-label="Color Picker (C)"
				tooltip="Color Picker (C)"
				icon={<FaEyeDropper />}
				aria-selected={tool === 'colorPicker' && !isStaging}
				isDisabled={isStaging}
				onClick={handleSelectColorPickerTool}
			/>
		</ButtonGroup>
	);
};
