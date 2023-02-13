import React from 'react';
import { FC } from 'react';
import { ButtonGroup, Flex } from '@chakra-ui/react';
import _ from 'lodash';
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
import { Popover, Slider, ColorPicker, IconButton } from '../../components';
import {
  toolSelector,
  brushColorAtom,
  brushSizeAtom,
  addFillRectAction,
  addEraseRectAction,
  isStagingSelector,
} from '../../atoms/canvas.atoms';

// import {
// 	addEraseRect,
// 	addFillRect,
// 	setBrushColor,
// 	setBrushSize,
// 	setTool,
// } from 'canvas/store/canvasSlice';
// import {
// 	canvasSelector,
// 	isStagingSelector,
// } from 'canvas/store/canvasSelectors';
// import { systemSelector } from 'system/store/systemSelectors';

// export const selector = createSelector(
// 	[canvasSelector, isStagingSelector, systemSelector],
// 	(canvas, isStaging, system) => {
// 		const { isProcessing } = system;
// 		const { tool, brushColor, brushSize } = canvas;

// 		return {
// 			tool,
// 			isStaging,
// 			isProcessing,
// 			brushColor,
// 			brushSize,
// 		};
// 	},
// 	{
// 		memoizeOptions: {
// 			resultEqualityCheck: _.isEqual,
// 		},
// 	},
// );

export const CanvasToolChooserOptions: FC = () => {
  // const { tool, brushColor, brushSize, isStaging } = useAppSelector(selector);

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
    []
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
    [tool]
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
    [tool]
  );

  useHotkeys(
    ['shift+f'],
    () => {
      handleFillRect();
    },
    {
      enabled: () => !isStaging,
      preventDefault: true,
    }
  );

  useHotkeys(
    ['delete', 'backspace'],
    () => {
      handleEraseBoundingBox();
    },
    {
      enabled: () => !isStaging,
      preventDefault: true,
    }
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
    [brushSize]
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
    [brushSize]
  );

  useHotkeys(
    ['shift+BracketLeft'],
    () => {
      setBrushColor({
        ...brushColor,
        a: _.clamp(brushColor.a - 0.05, 0.05, 1),
      });
    },
    {
      enabled: () => !isStaging,
      preventDefault: true,
    },
    [brushColor]
  );

  useHotkeys(
    ['shift+BracketRight'],
    () => {
      setBrushColor({
        ...brushColor,
        a: _.clamp(brushColor.a + 0.05, 0.05, 1),
      });
    },
    {
      enabled: () => !isStaging,
      preventDefault: true,
    },
    [brushColor]
  );

  const handleSelectBrushTool = () => setTool('brush');
  const handleSelectEraserTool = () => setTool('eraser');
  const handleSelectColorPickerTool = () => setTool('colorPicker');
  const handleFillRect = () => addFillRect();
  const handleEraseBoundingBox = () => addEraseRect();

  return (
    <ButtonGroup isAttached>
      <Popover
        trigger="hover"
        triggerComponent={
          <IconButton
          aria-label="Brush Tool (B)"
          tooltip="Brush Tool (B)"
          icon={<FaPaintBrush />}
          data-selected={tool === 'brush' && !isStaging}
          onClick={handleSelectBrushTool}
          isDisabled={isStaging}
        />
        }
      >
        <Flex minWidth="15rem" direction="column" gap="1rem" width="100%">
          <Flex gap="1rem" justifyContent="space-between">
            <Slider
              label="Size"
              value={brushSize}
              withInput
              max={200}
              onChange={(newSize) => setBrushSize(newSize)}
              sliderNumberInputProps={{ max: 500 }}
              inputReadOnly={false}
            />
          </Flex>
          <ColorPicker
            style={{
              width: '100%',
              paddingTop: '0.5rem',
              paddingBottom: '0.5rem',
            }}
            color={brushColor}
            onChange={(newColor) => setBrushColor(newColor)}
          />
        </Flex>
      </Popover>
      <Popover
        trigger="hover"
        triggerComponent={
          <IconButton
          aria-label="Eraser Tool (E)"
          tooltip="Eraser Tool (E)"
          icon={<FaEraser />}
          data-selected={tool === 'eraser' && !isStaging}
          isDisabled={isStaging}
          onClick={handleSelectEraserTool}
        />
        }
      >
        <Flex minWidth="15rem" direction="column" gap="1rem" width="100%">
          <Flex gap="1rem" justifyContent="space-between">
            <Slider
              label="Size"
              value={brushSize}
              withInput
              max={200}
              onChange={(newSize) => setBrushSize(newSize)}
              sliderNumberInputProps={{ max: 500 }}
              inputReadOnly={false}
            />
          </Flex>
        </Flex>
      </Popover>
      <Popover
        trigger="hover"
        triggerComponent={
          <IconButton
          aria-label="Fill Bounding Box (Shift+F)"
          tooltip="Fill Bounding Box (Shift+F)"
          icon={<FaFillDrip />}
          isDisabled={isStaging}
          onClick={handleFillRect}
        />
        }
      >
        <Flex minWidth="15rem" direction="column" gap="1rem" width="100%">
          <ColorPicker
            style={{
              width: '100%',
              paddingTop: '0.5rem',
              paddingBottom: '0.5rem',
            }}
            color={brushColor}
            onChange={(newColor) => setBrushColor(newColor)}
          />
        </Flex>
      </Popover>
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
        data-selected={tool === 'colorPicker' && !isStaging}
        isDisabled={isStaging}
        onClick={handleSelectColorPickerTool}
      />
    </ButtonGroup>
  );
};
