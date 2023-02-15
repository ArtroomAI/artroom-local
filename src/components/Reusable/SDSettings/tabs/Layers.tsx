import React, { useState } from 'react';
import {
	Box,
	useDisclosure,
	Center,
	Text,
	useColorModeValue,
	Flex,
} from '@chakra-ui/react';
import { useRecoilState, useRecoilValue } from 'recoil';
import { Menu, Item, useContextMenu } from 'react-contexify';
import Konva from 'konva';
import { DragDropContext, Droppable, DropResult } from 'react-beautiful-dnd';
import { v4 as uuidv4 } from 'uuid';
import _ from 'lodash';
import * as atom from '../../../UnifiedCanvas/atoms/canvas.atoms';
import { ImageLayer } from '../../../UnifiedCanvas/atoms/canvasTypes';
import { RenameLayerModal } from '../../RenameLayerModal/RenameLayerModal';
import { Slider } from '../../../UnifiedCanvas/components';
import { LayerObject } from '../../LayerObject/LayerObject';
import { getCanvasBaseLayer } from '../../../UnifiedCanvas/util';
import { getNameForLayer, MERGED_LAYER_REG_EXP } from '../../../../utils';
import { MENU_LAYER, MENU_MERGE } from '../../../../constants';
import 'react-contexify/dist/ReactContexify.css';

export const Layers: React.FC = () => {
	const [layerState, setLayerState] = useRecoilState(atom.layerStateAtom);
	const [layer, setLayer] = useRecoilState(atom.layerAtom);
	const [selectedMultipleImageLayers, setSelectedMultipleImageLayers] =
		useRecoilState(atom.selectedMultipleImageLayersAtom);
	const [pastLayerStates, setPastLayerStates] = useRecoilState(
		atom.pastLayerStatesAtom,
	);
	const selectedImageLayer = useRecoilValue(atom.selectedImageLayerSelector);
	const maxHistory = useRecoilValue(atom.maxHistoryAtom);
	const stageScale = useRecoilValue(atom.stageScaleAtom);

	const { isOpen, onOpen, onClose } = useDisclosure();
	const droppableBG = useColorModeValue('gray.100', 'gray.700');

	const { show } = useContextMenu({
		id: MENU_LAYER,
	});

	const calculateCoordinates = (): { x: number; y: number } => {
		let minX: number | null = null;
		let minY: number | null = null;
		layerState.images.forEach(elem => {
			if (minX === null) {
				minX = elem.picture.x;
			}
			if (minY === null) {
				minY = elem.picture.y;
			}
			minX = Math.min(minX, elem.picture.x);
			minY = Math.min(minY, elem.picture.y);
		});
		return { x: minX || 0, y: minY || 0 };
	};

	const displayMenu = (
		e: React.MouseEvent<HTMLDivElement, MouseEvent>,
		{ id, name, menuId }: Record<string, string>,
	) => {
		show({
			event: e,
			props: { id, name },
			id: menuId,
		});
	};

	const canvasBaseLayer = getCanvasBaseLayer();

	const [layerToRename, setLayerToRename] = useState<{
		id: string;
		name: string;
	} | null>(null);

	const onDragLayerObjectEnd = (result: DropResult) => {
		const { destination, source, draggableId } = result;

		if (!destination) {
			return;
		}

		if (
			destination.droppableId === source.droppableId &&
			destination.index === source.index
		) {
			return;
		}

		const newImages = Array.from(layerState.images);
		const deletedItemsArray = newImages.splice(source.index, 1);
		newImages.splice(destination.index, 0, deletedItemsArray[0]);

		setPastLayerStates([...pastLayerStates, _.cloneDeep(layerState)]);

		if (pastLayerStates.length > maxHistory) {
			setPastLayerStates(pastLayerStates.slice(1));
		}

		setLayerState({
			...layerState,
			images: newImages,
		});
	};

	const onChangeOpacity = (value: number) => {
		setLayerState({
			...layerState,
			images: layerState.images.map(elem => {
				if (elem.id === layer) {
					return { ...elem, opacity: value };
				} else {
					return elem;
				}
			}),
		});
	};

	const onSelectMultipleLayers = (
		selectedLayer: ImageLayer,
		added: boolean,
	) => {
		if (added) {
			setSelectedMultipleImageLayers(
				selectedMultipleImageLayers.filter(
					elem => elem.id === selectedLayer.id,
				),
			);
		} else {
			setSelectedMultipleImageLayers([
				...selectedMultipleImageLayers,
				selectedLayer,
			]);
		}
	};

	const onMergeLayers = async () => {
		const group = new Konva.Group({
			draggable: true,
		});

		const layersWithImages = canvasBaseLayer?.find('.imageLayer');

		const filter = selectedMultipleImageLayers.map(elem => elem.id);

		if (layersWithImages) {
			layersWithImages.forEach(elem => {
				if (filter.includes(elem.id())) {
					group.add(elem as Konva.Shape);
				}
			});
		}

		const groupBlob = await group.toBlob({});

		setPastLayerStates([...pastLayerStates, _.cloneDeep(layerState)]);

		if (pastLayerStates.length > maxHistory) {
			setPastLayerStates(pastLayerStates.slice(1));
		}

		const groupBlobURL = window.URL.createObjectURL(groupBlob as Blob);
		const newLayerId = uuidv4();
		const newLayer = {
			name: getNameForLayer(
				MERGED_LAYER_REG_EXP,
				layerState.images,
				'Merged layer',
			),
			id: newLayerId,
			isVisible: true,
			opacity: 1,
			picture: {
				kind: 'image' as const,
				x: calculateCoordinates().x,
				y: calculateCoordinates().y,
				width: 0,
				height: 0,
				uuid: newLayerId,
				url: groupBlobURL,
			},
		};

		setLayerState({
			...layerState,
			objects: layerState.objects.filter(
				elem => !filter.includes(elem.layer || ''),
			),
			images: layerState.images
				.filter(elem => !filter.includes(elem.id))
				.concat(newLayer),
		});
		setSelectedMultipleImageLayers([]);
	};

	const onDuplicate = async (id: string, name: string) => {
		const groupToDuplicate = canvasBaseLayer?.findOne(`#${id}`);
		if (groupToDuplicate) {
			const groupBlob = await groupToDuplicate.toBlob({
				pixelRatio: 1 / stageScale,
			});
			const groupBlobURL = window.URL.createObjectURL(groupBlob as Blob);

			const oldLayer = layerState.images.find(elem => elem.id === id);

			const newLayerId = uuidv4();
			const newLayer = {
				name: `${name} (copied)`,
				id: newLayerId,
				isVisible: oldLayer?.isVisible || true,
				opacity: oldLayer?.opacity || 1,
				picture: {
					kind: 'image' as const,
					x: oldLayer?.picture.x || 0,
					y: oldLayer?.picture.y || 0,
					width: groupToDuplicate.getSize().width,
					height: groupToDuplicate.getSize().height,
					uuid: newLayerId,
					url: groupBlobURL,
				},
			};

			setPastLayerStates([...pastLayerStates, _.cloneDeep(layerState)]);

			if (pastLayerStates.length > maxHistory) {
				setPastLayerStates(pastLayerStates.slice(1));
			}

			setLayerState({
				...layerState,
				images: [...layerState.images, newLayer],
			});
		}
	};

	async function handleContextMenuItemClick(obj: {
		id?: string;
		// triggerEvent;
		// event;
		props?: { id: string; name: string; menuId: string };
		// data;
	}) {
		switch (obj.id) {
			case 'duplicate': {
				if (obj.props) {
					onDuplicate(obj.props.id, obj.props.name);
				}
				break;
			}
			case 'rename':
				if (obj.props) {
					setLayerToRename({
						id: obj.props.id,
						name: obj.props.name,
					});
					onOpen();
				}
				break;
			case 'delete':
				setLayerState({
					...layerState,
					images: layerState.images.filter(
						elem => elem.id !== obj.props?.id,
					),
					objects: layerState.objects.filter(
						elem => elem.layer !== obj.id,
					),
				});
				break;
			default:
				return;
		}
	}

	return (
		<Flex direction="column" h="100%">
			<Center minH="30px" px="10px">
				{selectedImageLayer ? (
					<Slider
						label="Opacity"
						value={selectedImageLayer.opacity}
						onChange={newSize => onChangeOpacity(newSize)}
						step={0.01}
						min={0}
						max={1}
					/>
				) : null}
			</Center>
			<DragDropContext onDragEnd={onDragLayerObjectEnd}>
				<Box borderRadius="5px" minH="100px">
					<Droppable droppableId="main-zone">
						{(provided, snapshot) => (
							<Box
								padding="8px"
								bg={
									snapshot.isDraggingOver
										? droppableBG
										: undefined
								}
								borderRadius="5px"
								{...provided.droppableProps}
								ref={provided.innerRef}>
								{[...layerState.images]
									.reverse()
									.map((elem, index) => (
										<LayerObject
											key={elem.id}
											index={index}
											imageUrl={elem.picture.url}
											id={elem.id}
											name={elem.name}
											isVisible={elem.isVisible}
											onSelect={() => {
												setLayer(elem.id);
											}}
											displayMenu={displayMenu}
											onSelectMultiple={isAlreadySelected =>
												onSelectMultipleLayers(
													elem,
													isAlreadySelected,
												)
											}
										/>
									))}
								{provided.placeholder}
							</Box>
						)}
					</Droppable>
				</Box>
			</DragDropContext>
			{!layerState.images.length ? (
				<Center flex={0.9}>
					<Text textAlign="center" fontSize="sm">
						Images that you will add to canvas will show up here as
						layers
					</Text>
				</Center>
			) : null}

			<Menu id={MENU_LAYER} animation={false}>
				<Item onClick={handleContextMenuItemClick} id="duplicate">
					<Text fontSize="sm">Duplicate</Text>
				</Item>
				<Item onClick={handleContextMenuItemClick} id="rename">
					<Text fontSize="sm">Rename</Text>
				</Item>
				<Item onClick={handleContextMenuItemClick} id="delete">
					<Text fontSize="sm">Delete</Text>
				</Item>
			</Menu>
			<Menu id={MENU_MERGE} animation={false}>
				{selectedMultipleImageLayers.length ? (
					<Item onClick={onMergeLayers}>
						<Text fontSize="sm">Merge layers</Text>
					</Item>
				) : null}
			</Menu>
			<RenameLayerModal
				isOpen={isOpen}
				layerId={layerToRename?.id}
				name={layerToRename?.name}
				onClose={() => {
					onClose();
					setLayerToRename(null);
				}}
			/>
		</Flex>
	);
};
