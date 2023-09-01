import { roundToMultiple } from './roundDownToMultiple'
import { Dimensions } from '../atoms/canvasTypes'

export const roundDimensionsTo64 = (dimensions: Dimensions): Dimensions => {
  return {
    width: roundToMultiple(dimensions.width, 64),
    height: roundToMultiple(dimensions.height, 64),
  }
}
