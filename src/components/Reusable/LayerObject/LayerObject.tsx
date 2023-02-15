import React from 'react';
import { useRecoilState, useRecoilValue } from 'recoil';
import { Draggable } from 'react-beautiful-dnd';
import {
	Flex,
	Text,
	Image,
	useColorModeValue,
	HStack,
	IconButton,
} from '@chakra-ui/react';
import * as atom from '../../UnifiedCanvas/atoms/canvas.atoms';
import { VisibleLayerIcon } from './VisibleLayerIcon';
import { MENU_LAYER, MENU_MERGE } from '../../../constants';

interface ILayerObjectProps {
	imageUrl: string;
	index: number;
	id: string;
	name: string;
	isVisible: boolean;
	onSelect: () => void;
	onSelectMultiple: (value: boolean) => void;
	displayMenu: (
		ev: React.MouseEvent<HTMLDivElement, MouseEvent>,
		object: Record<string, string>,
	) => void;
}

export const LayerObject: React.FC<ILayerObjectProps> = ({
	imageUrl,
	index,
	id,
	name,
	isVisible,
	onSelect,
	onSelectMultiple,
	displayMenu,
}) => {
	const layer = useRecoilValue(atom.layerAtom);
	const [layerState, setLayerState] = useRecoilState(atom.layerStateAtom);
	const selectedMultipleImageLayers = useRecoilValue(
		atom.selectedMultipleImageLayersAtom,
	);

	const toggleVisibility = (value: boolean) => {
		setLayerState({
			...layerState,
			images: layerState.images.map(elem => {
				if (elem.id === id) {
					return { ...elem, isVisible: value };
				} else {
					return elem;
				}
			}),
		});
	};

	const onContextMenu = (
		ev: React.MouseEvent<HTMLDivElement, MouseEvent>,
	) => {
		ev.preventDefault();
		displayMenu(ev, {
			id,
			name,
			menuId: isAlreadySelected ? MENU_MERGE : MENU_LAYER,
		});
	};

	const onClick = (ev: React.MouseEvent<HTMLDivElement, MouseEvent>) => {
		if (ev.shiftKey) {
			onSelectMultiple(isAlreadySelected);
		} else {
			onSelect();
		}
	};

	const getBackground = () => {
		if (isAlreadySelected) {
			return 'darkPrimary.500';
		} else if (layer === id) {
			return 'buttonPrimary';
		} else {
			return useColorModeValue('white', 'background');
		}
	};

	const getColor = () => {
		if (isAlreadySelected || layer === id) {
			return 'white';
		}
		return useColorModeValue('black', 'white');
	};

	const isAlreadySelected = !!selectedMultipleImageLayers.find(
		elem => elem.id === id,
	);

	return (
		<Draggable draggableId={id} index={index}>
			{(provided, snapshot) => (
				<Flex
					onContextMenu={onContextMenu}
					minH="70px"
					align="center"
					borderRadius="5px"
					border={snapshot.isDragging ? '3px solid' : '1px solid'}
					borderColor={useColorModeValue('gray.300', 'gray.700')}
					padding="8px"
					mb="8px"
					bg={getBackground()}
					{...provided.draggableProps}
					ref={provided.innerRef}>
					<IconButton
						aria-label="Layer visibility"
						variant="unstyled"
						display="flex"
						alignItems="center"
						justifyContent="center"
						size="sm"
						onClick={() => toggleVisibility(!isVisible)}>
						<VisibleLayerIcon isChecked={isVisible} />
					</IconButton>
					<HStack
						ml="10px"
						w="100%"
						onClick={onClick}
						{...provided.dragHandleProps}>
						<Image w="50px" mr="10px" src={imageUrl} />
						<Text fontSize="sm" color={getColor()}>
							{name}
						</Text>
					</HStack>
				</Flex>
			)}
		</Draggable>
	);
};
