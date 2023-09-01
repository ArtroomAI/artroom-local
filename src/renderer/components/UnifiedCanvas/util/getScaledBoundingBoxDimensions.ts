import { roundToMultiple } from './roundDownToMultiple'
import { Dimensions } from '../atoms/canvasTypes'

export const getScaledBoundingBoxDimensions = (dimensions: Dimensions) => {
  const { width, height } = dimensions

  const scaledDimensions = { width, height }
  const targetArea = 512 * 512
  const aspectRatio = width / height
  let currentArea = width * height
  let maxDimension = 448
  while (currentArea < targetArea) {
    maxDimension += 64
    if (width === height) {
      scaledDimensions.width = 512
      scaledDimensions.height = 512
      break
    } else {
      if (aspectRatio > 1) {
        scaledDimensions.width = maxDimension
        scaledDimensions.height = roundToMultiple(maxDimension / aspectRatio, 64)
      } else if (aspectRatio < 1) {
        scaledDimensions.height = maxDimension
        scaledDimensions.width = roundToMultiple(maxDimension * aspectRatio, 64)
      }
      currentArea = scaledDimensions.width * scaledDimensions.height
    }
  }

  return scaledDimensions
}
