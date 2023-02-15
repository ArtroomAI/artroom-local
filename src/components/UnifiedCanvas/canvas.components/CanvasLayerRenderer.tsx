import React, { useRef, useEffect } from 'react';
import { Group, Line, Rect, Transformer } from 'react-konva';
import Konva from 'konva';
import { useRecoilValue, useRecoilState, useSetRecoilState } from 'recoil';
import _ from 'lodash';
import { useDelta } from 'react-delta';
import { KonvaEventObject } from 'konva/lib/Node';

import * as atom from '../atoms/canvas.atoms';
import { CanvasImage } from './CanvasImage';
import {
	CanvasImage as CanvasImageType,
	ImageLayer,
	CanvasObject,
	isCanvasImageLayerLine,
	isCanvasImageLayerFillRect,
	isCanvasImageLayerEraseRect,
} from '../atoms/canvasTypes';
import { rgbaColorToString } from '../util';
import { useEventEmitter } from '../../../helpers';

interface ICanvasLayerRendererProps {
	imageData: ImageLayer;
	objects: CanvasObject[];
}

export const CanvasLayerRenderer: React.FC<ICanvasLayerRendererProps> = ({
	imageData,
	objects,
}) => {
	const [tool, setTool] = useRecoilState(atom.toolAtom);
	const [layerState, setLayerState] = useRecoilState(atom.layerStateAtom);
	const [pastLayerStates, setPastLayerStates] = useRecoilState(
		atom.pastLayerStatesAtom,
	);

	const layer = useRecoilValue(atom.layerAtom);
	const isDrawing = useRecoilValue(atom.isDrawingAtom);
	const stageScale = useRecoilValue(atom.stageScaleAtom);
	const maxHistory = useRecoilValue(atom.maxHistoryAtom);

	const setPastLayer = useSetRecoilState(atom.pastLayerAtom);

	const transformerRef = useRef<Konva.Transformer>(null);
	const groupRef = useRef<Konva.Group>(null);

	const { useListener } = useEventEmitter();

	const prevTool = useDelta(tool);
	const prevLayer = useDelta(layer);

	useEffect(() => {
		if (!transformerRef.current || !groupRef.current) return;
		transformerRef.current.nodes([groupRef.current]);
	}, []);

	useEffect(() => {
		if ((tool === 'transform' || tool === 'move') && groupRef.current) {
			transformerRef.current?.nodes([groupRef.current]);
		} else {
			transformerRef.current?.nodes([]);
		}
	}, [tool]);

	useListener(
		'acceptTransform',
		async () => {
			if (imageData.id === layer) {
				const groupBlob = await groupRef.current?.toBlob({
					pixelRatio: 1 / stageScale,
				});

				const groupBlobURL = window.URL.createObjectURL(
					groupBlob as Blob,
				);

				setPastLayerStates([
					...pastLayerStates,
					_.cloneDeep(layerState),
				]);
				setPastLayer('');
				if (pastLayerStates.length > maxHistory) {
					setPastLayerStates(pastLayerStates.slice(1));
				}

				setLayerState({
					...layerState,
					objects: layerState.objects.filter(
						elem => elem.layer !== imageData.id,
					),
					images: layerState.images.map(elem => {
						if (elem.id === imageData.id) {
							return {
								...elem,
								picture: {
									...elem.picture,
									x: groupRef.current?.getPosition().x || 0,
									y: groupRef.current?.getPosition().y || 0,
									url: groupBlobURL,
								},
							};
						} else {
							return elem;
						}
					}),
				});
				groupRef.current?.scaleX(1);
				groupRef.current?.scaleY(1);
				setTool('brush');
			}
		},
		[],
	);

	useListener(
		'rejectTransform',
		() => {
			setTool('brush');
			if (groupRef.current) {
				transformerRef.current?.nodes([groupRef.current]);
			}
			groupRef.current?.scaleX(1);
			groupRef.current?.scaleY(1);
		},
		[],
	);

	useEffect(() => {
		if (
			(prevLayer?.prev === imageData.id && tool === 'transform') ||
			(layer === imageData.id && prevTool?.prev === 'transform')
		) {
			groupRef.current?.scaleX(1);
			groupRef.current?.scaleY(1);
		}
	}, [tool, layer]);

	const handleDragEnd = (e: KonvaEventObject<DragEvent>) => {
		const position = e.target.getPosition();
		const groupId = e.target.id();
		setPastLayerStates([...pastLayerStates, _.cloneDeep(layerState)]);
		setPastLayer('');
		if (pastLayerStates.length > maxHistory) {
			setPastLayerStates(pastLayerStates.slice(1));
		}

		setLayerState({
			...layerState,
			images: layerState.images.map(elem => {
				if (elem.id === groupId) {
					return {
						...elem,
						picture: {
							...elem.picture,
							x: position.x,
							y: position.y,
						},
					};
				} else {
					return elem;
				}
			}),
		});
	};

	return (
		<>
			<Group
				name="imageLayer"
				id={imageData.id}
				draggable={tool === 'move' && layer === imageData.id}
				visible={imageData.isVisible}
				opacity={imageData.opacity}
				ref={groupRef}
				onDragEnd={handleDragEnd}
				x={(imageData.picture as CanvasImageType).x}
				y={(imageData.picture as CanvasImageType).y}>
				<CanvasImage
					x={0}
					y={0}
					url={(imageData.picture as CanvasImageType).url}
				/>
				{objects.map((item, i) => {
					if (isCanvasImageLayerLine(item, imageData.id)) {
						const line = (
							<Line
								key={i}
								points={item.points}
								stroke={
									item.color
										? rgbaColorToString(item.color)
										: 'rgb(0,0,0)'
								}
								strokeWidth={item.strokeWidth * 2}
								tension={0}
								lineCap="round"
								lineJoin="round"
								shadowForStrokeEnabled={false}
								listening={false}
								globalCompositeOperation={
									item.tool === 'brush'
										? 'source-over'
										: 'destination-out'
								}
							/>
						);
						if (item.clip) {
							return (
								<Group
									key={i}
									clipX={item.clip.x}
									clipY={item.clip.y}
									clipWidth={item.clip.width}
									clipHeight={item.clip.height}>
									{line}
								</Group>
							);
						} else {
							return line;
						}
					} else if (isCanvasImageLayerFillRect(item, imageData.id)) {
						return (
							<Rect
								key={i}
								x={item.x}
								y={item.y}
								width={item.width}
								height={item.height}
								fill={rgbaColorToString(item.color)}
							/>
						);
					} else if (
						isCanvasImageLayerEraseRect(item, imageData.id)
					) {
						return (
							<Rect
								key={i}
								x={item.x}
								y={item.y}
								width={item.width}
								height={item.height}
								fill={'rgb(255, 255, 255)'}
								globalCompositeOperation={'destination-out'}
							/>
						);
					}
				})}
			</Group>
			<Transformer
				anchorCornerRadius={3}
				anchorFill="rgba(212,216,234,1)"
				anchorSize={15}
				anchorStroke="rgb(42,42,42)"
				borderDash={[4, 4]}
				borderEnabled={
					(tool === 'move' || tool === 'transform') &&
					layer === imageData.id
				}
				borderStrokeWidth={3}
				borderStroke="white"
				draggable={false}
				enabledAnchors={
					tool === 'transform' && layer === imageData.id
						? undefined
						: []
				}
				flipEnabled={false}
				ignoreStroke
				keepRatio={false}
				listening={
					!isDrawing &&
					(tool === 'move' || tool === 'transform') &&
					layer === imageData.id
				}
				ref={transformerRef}
				rotateEnabled={false}
			/>
		</>
	);
};
