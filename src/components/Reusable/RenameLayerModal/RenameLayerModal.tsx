import React, { useEffect } from 'react';
import {
	Modal,
	ModalOverlay,
	ModalContent,
	ModalHeader,
	ModalCloseButton,
	ModalBody,
	ModalFooter,
	Button,
} from '@chakra-ui/react';
import { InputText } from '../../InputText/InputText';
import { useRecoilState } from 'recoil';
import { layerStateAtom } from '../../UnifiedCanvas/atoms/canvas.atoms';
import { useForm } from 'react-hook-form';
import { renameLayer } from '../../../validation';

interface IRenameLayerModalProps {
	isOpen: boolean;
	onClose: () => void;
	layerId?: string;
	name?: string;
}

interface RenameModalPayload {
	name: string;
}

export const RenameLayerModal: React.FC<IRenameLayerModalProps> = ({
	isOpen,
	onClose,
	layerId,
	name,
}) => {
	const {
		formState: { errors },
		handleSubmit,
		register,
		reset,
	} = useForm<RenameModalPayload>({ resolver: renameLayer });

	const [layerState, setLayerState] = useRecoilState(layerStateAtom);

	useEffect(() => {
		if (name) {
			reset({ name });
		}
	}, [name]);

	const onRename = (data: RenameModalPayload) => {
		setLayerState({
			...layerState,
			images: layerState.images.map(elem => {
				if (elem.id === layerId) {
					return {
						...elem,
						name: data.name,
					};
				} else {
					return elem;
				}
			}),
		});
		onClose();
	};

	return (
		<Modal isOpen={isOpen} onClose={onClose}>
			<ModalOverlay />
			<form onSubmit={handleSubmit(onRename)}>
				<ModalContent>
					<ModalHeader>Rename Layer</ModalHeader>
					<ModalCloseButton />
					<ModalBody>
						<InputText
							{...register('name')}
							label="New Name"
							autoFocus
							errorMsg={errors.name?.message}
						/>
					</ModalBody>

					<ModalFooter>
						<Button
							colorScheme="blue"
							mr={3}
							type="button"
							onClick={onClose}>
							Close
						</Button>
						<Button variant="outline" type="submit">
							Rename
						</Button>
					</ModalFooter>
				</ModalContent>
			</form>
		</Modal>
	);
};
