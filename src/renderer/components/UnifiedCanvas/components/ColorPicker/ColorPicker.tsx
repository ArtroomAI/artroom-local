import React from 'react'
import { FC } from 'react'
import { RgbaColorPicker } from 'react-colorful'
import { ColorPickerBaseProps, RgbaColor } from 'react-colorful/dist/types'

type IColorPickerProps = ColorPickerBaseProps<RgbaColor> & {
  styleClass?: string
}

export const ColorPicker: FC<IColorPickerProps> = (props) => {
  const { styleClass, ...rest } = props

  return <RgbaColorPicker className={`painter__color-picker ${styleClass}`} {...rest} />
}
