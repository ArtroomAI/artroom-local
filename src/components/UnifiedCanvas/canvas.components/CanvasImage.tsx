import { FC, useRef } from 'react';
import { Image } from 'react-konva';
import useImage from 'use-image';
import Konva from 'konva';

type ICanvasImageProps = {
	url: string;
	x: number;
	y: number;
};
export const CanvasImage: FC<ICanvasImageProps> = ({ url, x, y }) => {
	const imageRef = useRef<Konva.Image | null>(null);

	const [image] = useImage(url);

	return <Image x={x} y={y} image={image} ref={imageRef} />;
};
