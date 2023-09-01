import React from 'react'
import { FC } from 'react'
import { Button, ButtonGroup, Flex } from '@chakra-ui/react'
import _ from 'lodash'
import { FaEraser, FaEyeDropper, FaFillDrip, FaPaintBrush, FaPlus } from 'react-icons/fa'
import { useHotkeys } from 'react-hotkeys-hook'
import { useRecoilState, useRecoilValue, useSetRecoilState } from 'recoil'
import { Popover, Slider, ColorPicker, IconButton } from '../../components'
import {
  toolSelector,
  brushColorAtom,
  brushSizeAtom,
  addFillRectAction,
  addEraseRectAction,
  isStagingSelector,
  boundingBoxCoordinatesAtom,
  boundingBoxDimensionsAtom,
  futureLayerStatesAtom,
  layerStateAtom,
  maxHistoryAtom,
  pastLayerStatesAtom,
  setInitialCanvasImageAction,
} from '../../atoms/canvas.atoms'
import { uploadImage } from '../../helpers/uploadImage'

const CanvasToolChooserOptionsHotkeys: React.FC = () => {
  const [tool, setTool] = useRecoilState(toolSelector)
  const [brushColor, setBrushColor] = useRecoilState(brushColorAtom)
  const [brushSize, setBrushSize] = useRecoilState(brushSizeAtom)
  const addFillRect = useSetRecoilState(addFillRectAction)
  const addEraseRect = useSetRecoilState(addEraseRectAction)
  const isStaging = useRecoilValue(isStagingSelector)

  const handleSelectBrushTool = () => setTool('brush')
  const handleSelectEraserTool = () => setTool('eraser')
  const handleSelectColorPickerTool = () => setTool('colorPicker')
  const handleFillRect = () => addFillRect()
  const handleEraseBoundingBox = () => addEraseRect()

  const boundingBoxCoordinates = useRecoilValue(boundingBoxCoordinatesAtom)
  const boundingBoxDimensions = useRecoilValue(boundingBoxDimensionsAtom)
  const maxHistory = useRecoilValue(maxHistoryAtom)
  const [layerState, setLayerState] = useRecoilState(layerStateAtom)
  const [pastLayerStates, setPastLayerStates] = useRecoilState(pastLayerStatesAtom)
  const setFutureLayerStates = useSetRecoilState(futureLayerStatesAtom)
  const setInitialCanvasImage = useSetRecoilState(setInitialCanvasImageAction)

  const fileAcceptedCallback = async (file: File) => {
    uploadImage({
      imageFile: file,
      setInitialCanvasImage,
      boundingBoxCoordinates,
      boundingBoxDimensions,
      setPastLayerStates,
      pastLayerStates,
      layerState,
      maxHistory,
      setLayerState,
      setFutureLayerStates,
    })
  }

  const handlePaste = async () => {
    try {
      const clipboardData = await navigator.clipboard.read()
      for (let i = 0; i < clipboardData.length; i++) {
        const clipboardItem = clipboardData[i]
        if (
          clipboardItem.types.includes('image/png') ||
          clipboardItem.types.includes('image/jpeg')
        ) {
          const blob = await clipboardItem.getType('image/png')
          const reader = new FileReader()
          reader.readAsDataURL(blob)
          reader.onloadend = async () => {
            const base64data = String(reader.result)
            blob['b64'] = base64data
            fileAcceptedCallback(blob)
          }
          break
        }
      }
    } catch (err) {
      console.error('Failed to read clipboard contents: ', err)
    }
  }

  useHotkeys(
    ['b'],
    () => {
      handleSelectBrushTool()
    },
    {
      enabled: () => !isStaging,
      preventDefault: true,
    },
    []
  )

  useHotkeys(
    ['e'],
    () => {
      handleSelectEraserTool()
    },
    {
      enabled: () => !isStaging,
      preventDefault: true,
    },
    [tool]
  )

  useHotkeys(
    ['c'],
    () => {
      handleSelectColorPickerTool()
    },
    {
      enabled: () => !isStaging,
      preventDefault: true,
    },
    [tool]
  )

  useHotkeys(
    ['shift+f'],
    () => {
      handleFillRect()
    },
    {
      enabled: () => !isStaging,
      preventDefault: true,
    }
  )

  useHotkeys(
    ['delete', 'backspace'],
    () => {
      handleEraseBoundingBox()
    },
    {
      enabled: () => !isStaging,
      preventDefault: true,
    }
  )

  useHotkeys(
    ['ctrl+v'],
    handlePaste,
    {
      enabled: () => !isStaging,
      preventDefault: true,
    },
    [fileAcceptedCallback]
  )

  useHotkeys(
    ['BracketLeft'],
    () => {
      setBrushSize(Math.max(brushSize - 5, 5))
    },
    {
      enabled: () => !isStaging,
      preventDefault: true,
    },
    [brushSize]
  )

  useHotkeys(
    ['BracketRight'],
    () => {
      setBrushSize(Math.min(brushSize + 5, 500))
    },
    {
      enabled: () => !isStaging,
      preventDefault: true,
    },
    [brushSize]
  )

  useHotkeys(
    ['shift+BracketLeft'],
    () => {
      setBrushColor({
        ...brushColor,
        a: _.clamp(brushColor.a - 0.05, 0.05, 1),
      })
    },
    {
      enabled: () => !isStaging,
      preventDefault: true,
    },
    [brushColor]
  )

  useHotkeys(
    ['shift+BracketRight'],
    () => {
      setBrushColor({
        ...brushColor,
        a: _.clamp(brushColor.a + 0.05, 0.05, 1),
      })
    },
    {
      enabled: () => !isStaging,
      preventDefault: true,
    },
    [brushColor]
  )

  useHotkeys(
    ['shift+BracketRight'],
    () => {
      setBrushColor({
        ...brushColor,
        a: _.clamp(brushColor.a + 0.05, 0.05, 1),
      })
    },
    {
      enabled: () => !isStaging,
      preventDefault: true,
    },
    [brushColor]
  )

  return null
}

export const CanvasToolChooserOptions: FC = () => {
  // const { tool, brushColor, brushSize, isStaging } = useAppSelector(selector);

  const [tool, setTool] = useRecoilState(toolSelector)
  const [brushColor, setBrushColor] = useRecoilState(brushColorAtom)
  const [brushSize, setBrushSize] = useRecoilState(brushSizeAtom)
  const addFillRect = useSetRecoilState(addFillRectAction)
  const addEraseRect = useSetRecoilState(addEraseRectAction)
  const isStaging = useRecoilValue(isStagingSelector)

  const handleSelectBrushTool = () => setTool('brush')
  const handleSelectEraserTool = () => setTool('eraser')
  const handleSelectColorPickerTool = () => setTool('colorPicker')
  const handleFillRect = () => addFillRect()
  const handleEraseBoundingBox = () => addEraseRect()

  return (
    <ButtonGroup isAttached>
      <CanvasToolChooserOptionsHotkeys />
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
          <Button
            size="sm"
            leftIcon={<FaPlus style={{ transform: 'rotate(45deg)' }} />}
            onClick={handleEraseBoundingBox}
          >
            Erase Bounding Box (Delete|Backspace)
          </Button>
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
        aria-label="Color Picker (C)"
        tooltip="Color Picker (C)"
        icon={<FaEyeDropper />}
        data-selected={tool === 'colorPicker' && !isStaging}
        isDisabled={isStaging}
        onClick={handleSelectColorPickerTool}
      />
    </ButtonGroup>
  )
}
