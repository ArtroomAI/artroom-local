import React from 'react'
import { FC } from 'react'
import { Image } from 'react-konva'
import useImage from 'use-image'

type ICanvasImageProps = {
  url: string
  x: number
  y: number
  width: number | null
  height: number | null
}
export const CanvasImage: FC<ICanvasImageProps> = (props) => {
  const { url, x, y } = props
  const [image] = useImage(url)
  return <Image x={x} y={y} image={image} listening={false} />
}
