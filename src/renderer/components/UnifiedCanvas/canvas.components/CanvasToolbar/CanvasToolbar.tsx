import React from 'react'
import { ButtonGroup, useToast } from '@chakra-ui/react'
import { FaArrowsAlt, FaCopy, FaCrosshairs, FaSave, FaTrash, FaUpload } from 'react-icons/fa'
import { useHotkeys } from 'react-hotkeys-hook'
import { FC, ChangeEvent } from 'react'
import { useRecoilState, useSetRecoilState, useRecoilValue } from 'recoil'
import { useImageUploader } from '../../hooks'
import {
  toolSelector,
  layerAtom,
  setIsMaskEnabledAction,
  resetCanvasAction,
  resetCanvasViewAction,
  resizeAndScaleCanvasAction,
  shouldCropToBoundingBoxOnSaveAtom,
  isStagingSelector,
  stageScaleAtom,
  boundingBoxCoordinatesAtom,
  boundingBoxDimensionsAtom,
  stageCoordinatesAtom,
} from '../../atoms/canvas.atoms'
import { isProcessingAtom } from '../../atoms/system.atoms'
import { IconButton } from '../../components'
import { CanvasLayer } from '../../atoms/canvasTypes'
import { CanvasToolChooserOptions } from './CanvasToolChooserOptions'
import { getCanvasBaseLayer, layerToDataURL } from '../../util'
import { CanvasMaskOptions } from './CanvasMaskOptions'
import { CanvasSettingsButtonPopover } from './CanvasSettingsButtonPopover'
import { CanvasRedoButton } from './CanvasRedoButton'
import { CanvasUndoButton } from './CanvasUndoButton'
import path from 'path'
import { CanvasUpscaleButtonPopover } from './CanvasUpscaleButtonPopover'
import { RemoveBackgroundButtonPopover } from './RemoveBackgroundButtonPopover'
import { batchNameState, imageSavePathState } from '../../../../SettingsManager'

export const CanvasOutpaintingControls: FC = () => {
  const canvasBaseLayer = getCanvasBaseLayer()

  const [tool, setTool] = useRecoilState(toolSelector)
  const [layer, setLayer] = useRecoilState(layerAtom)
  const [isMaskEnabled, setIsMaskEnabled] = useRecoilState(setIsMaskEnabledAction)
  const resetCanvas = useSetRecoilState(resetCanvasAction)
  const resetCanvasView = useSetRecoilState(resetCanvasViewAction)
  const resizeAndScaleCanvas = useSetRecoilState(resizeAndScaleCanvasAction)
  const shouldCropToBoundingBoxOnSave = useRecoilValue(shouldCropToBoundingBoxOnSaveAtom)
  const boundingBoxCoordinates = useRecoilValue(boundingBoxCoordinatesAtom)
  const boundingBoxDimensions = useRecoilValue(boundingBoxDimensionsAtom)
  const isProcessing = useRecoilValue(isProcessingAtom)
  const isStaging = useRecoilValue(isStagingSelector)

  const { openUploader } = useImageUploader()

  //For Generation
  const stageScale = useRecoilValue(stageScaleAtom)
  const stageCoordinates = useRecoilValue(stageCoordinatesAtom)
  const batchName = useRecoilValue(batchNameState)
  const imageSavePath = useRecoilValue(imageSavePathState)
  const toast = useToast({})

  useHotkeys(
    ['v'],
    () => {
      handleSelectMoveTool()
    },
    {
      enabled: () => !isStaging,
      preventDefault: true,
    },
    []
  )

  useHotkeys(
    ['r'],
    () => {
      handleResetCanvasView()
    },
    {
      enabled: () => true,
      preventDefault: true,
    },
    [canvasBaseLayer]
  )

  useHotkeys(
    ['shift+s'],
    () => {
      handleSaveToGallery()
    },
    {
      enabled: () => !isStaging,
      preventDefault: true,
    },
    [canvasBaseLayer, isProcessing]
  )

  useHotkeys(
    ['meta+c', 'ctrl+c'],
    () => {
      handleCopyImageToClipboard()
    },
    {
      enabled: () => !isStaging,
      preventDefault: true,
    },
    [canvasBaseLayer, isProcessing]
  )

  const handleSelectMoveTool = () => setTool('move')

  const handleResetCanvasView = () => {
    const canvasBaseLayer = getCanvasBaseLayer()
    if (!canvasBaseLayer) return
    const clientRect = canvasBaseLayer.getClientRect({
      skipTransform: true,
    })
    console.log(clientRect, 'clientRect')
    resetCanvasView({
      contentRect: clientRect,
    })
  }

  const handleResetCanvas = () => {
    resetCanvas()
    resizeAndScaleCanvas()
  }

  const handleSaveToGallery = () => {
    const canvasBaseLayer = getCanvasBaseLayer()

    const { dataURL, boundingBox: originalBoundingBox } = layerToDataURL(
      canvasBaseLayer,
      stageScale,
      stageCoordinates,
      shouldCropToBoundingBoxOnSave
        ? { ...boundingBoxCoordinates, ...boundingBoxDimensions }
        : undefined
    )
    const timestamp = new Date().getTime()
    const imagePath = path.join(imageSavePath, batchName, timestamp + '.png')
    console.log(imagePath)
    window.api.saveFromDataURL(JSON.stringify({ dataURL, imagePath }))
    toast({
      title: 'Image Saved',
      status: 'success',
      duration: 1500,
      isClosable: true,
    })
  }

  const handleCopyImageToClipboard = () => {
    const canvasBaseLayer = getCanvasBaseLayer()
    const { dataURL, boundingBox: originalBoundingBox } = layerToDataURL(
      canvasBaseLayer,
      stageScale,
      stageCoordinates,
      shouldCropToBoundingBoxOnSave
        ? { ...boundingBoxCoordinates, ...boundingBoxDimensions }
        : undefined
    )
    window.api.copyToClipboard(dataURL)
    toast({
      title: 'Image Copied',
      status: 'success',
      duration: 1500,
      isClosable: true,
    })
  }

  const handleChangeLayer = (e: ChangeEvent<HTMLSelectElement>) => {
    const newLayer = e.target.value as CanvasLayer
    setLayer(newLayer)
    if (newLayer === 'mask' && !isMaskEnabled) {
      setIsMaskEnabled(true)
    }
  }

  return (
    <div className="inpainting-settings">
      <CanvasMaskOptions />
      <CanvasToolChooserOptions />
      <ButtonGroup isAttached>
        <IconButton
          aria-label="Move Tool (V)"
          tooltip="Move Tool (V)"
          icon={<FaArrowsAlt />}
          data-selected={tool === 'move' || isStaging}
          onClick={handleSelectMoveTool}
        />
        <IconButton
          aria-label="Reset View (R)"
          tooltip="Reset View (R)"
          icon={<FaCrosshairs />}
          onClick={handleResetCanvasView}
        />
      </ButtonGroup>
      <ButtonGroup isAttached>
        <CanvasUpscaleButtonPopover />
        <RemoveBackgroundButtonPopover />
        <IconButton
          aria-label="Save (Shift+S)"
          tooltip="Save (Shift+S)"
          icon={<FaSave />}
          onClick={handleSaveToGallery}
          isDisabled={isStaging}
        />
        <IconButton
          aria-label="Copy to Clipboard (Cmd/Ctrl+C)"
          tooltip="Copy to Clipboard (Cmd/Ctrl+C)"
          icon={<FaCopy />}
          onClick={handleCopyImageToClipboard}
          isDisabled={isStaging}
        />
        <IconButton
          aria-label="Import Image"
          tooltip="Import Image"
          icon={<FaUpload />}
          onClick={openUploader}
          isDisabled={isStaging}
        />
        <IconButton
          aria-label="Clear Canvas"
          tooltip="Clear Canvas"
          icon={<FaTrash />}
          onClick={handleResetCanvas}
          style={{ backgroundColor: 'var(--btn-delete-image)' }}
          isDisabled={isStaging}
        />
      </ButtonGroup>
      <ButtonGroup isAttached>
        <CanvasUndoButton />
        <CanvasRedoButton />
      </ButtonGroup>
      <CanvasSettingsButtonPopover />
    </div>
  )
}
