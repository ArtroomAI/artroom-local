/* eslint-disable no-mixed-spaces-and-tabs */
import { selector } from 'recoil'
import { COLOR_PICKER_SIZE, COLOR_PICKER_STROKE_RADIUS, rgbaColorToString } from '../util'
import {
  boundingBoxCoordinatesAtom,
  boundingBoxDimensionsAtom,
  brushSizeAtom,
  cursorPositionAtom,
  isMovingBoundingBoxAtom,
  isTransformingBoundingBoxAtom,
  maskColorAtom,
  shouldRestrictStrokesToBoxAtom,
  shouldShowBrushAtom,
  stageDimensionsAtom,
  stageScaleAtom,
} from './canvas.atoms'

export const brushXSelector = selector({
  key: 'brushX.selector',
  get: ({ get }) => {
    const cursorPosition = get(cursorPositionAtom)
    const stageDimensions = get(stageDimensionsAtom)
    return cursorPosition ? cursorPosition.x : stageDimensions.width / 2
  },
})

export const brushYSelector = selector({
  key: 'brushY.selector',
  get: ({ get }) => {
    const cursorPosition = get(cursorPositionAtom)
    const stageDimensions = get(stageDimensionsAtom)
    return cursorPosition ? cursorPosition.y : stageDimensions.height / 2
  },
})

export const radiusSelector = selector({
  key: 'radius.selector',
  get: ({ get }) => {
    return get(brushSizeAtom) / 2
  },
})

export const shouldDrawBrushPreviewSelector = selector({
  key: 'shouldDrawBrushPreview.selector',
  get: ({ get }) => {
    const isMovingBoundingBox = get(isMovingBoundingBoxAtom)
    const isTransformingBoundingBox = get(isTransformingBoundingBoxAtom)
    const cursorPosition = get(cursorPositionAtom)
    const shouldShowBrush = get(shouldShowBrushAtom)
    return !(isMovingBoundingBox || isTransformingBoundingBox || !cursorPosition) && shouldShowBrush
  },
})

export const clipSelector = selector({
  key: 'clip.selector',
  get: ({ get }) => {
    const shouldRestrictStrokesToBox = get(shouldRestrictStrokesToBoxAtom)
    const boundingBoxCoordinates = get(boundingBoxCoordinatesAtom)
    const boundingBoxDimensions = get(boundingBoxDimensionsAtom)

    return shouldRestrictStrokesToBox
      ? {
          clipX: boundingBoxCoordinates.x,
          clipY: boundingBoxCoordinates.y,
          clipWidth: boundingBoxDimensions.width,
          clipHeight: boundingBoxDimensions.height,
        }
      : {}
  },
})

export const dotRadiusSelector = selector({
  key: 'dotRadius.selector',
  get: ({ get }) => {
    return 1.5 / get(stageScaleAtom)
  },
})

export const strokeWidthSelector = selector({
  key: 'strokeWidth.selector',
  get: ({ get }) => {
    return 1.5 / get(stageScaleAtom)
  },
})

export const colorPickerOuterRadiusSelector = selector({
  key: 'colorPickerOuterRadius.selector',
  get: ({ get }) => {
    return COLOR_PICKER_SIZE / get(stageScaleAtom)
  },
})

export const colorPickerInnerRadiusSelector = selector({
  key: 'colorPickerInnerRadius.selector',
  get: ({ get }) => {
    return (COLOR_PICKER_SIZE - COLOR_PICKER_STROKE_RADIUS + 1) / get(stageScaleAtom)
  },
})

export const maskColorStringHalfAlphaSelector = selector({
  key: 'maskColorStringHalfAlpha.selector',
  get: ({ get }) => {
    return rgbaColorToString({ ...get(maskColorAtom), a: 0.5 })
  },
})
