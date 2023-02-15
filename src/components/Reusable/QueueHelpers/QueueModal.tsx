import React from 'react';
import {
	useDisclosure,
	Modal,
	ModalOverlay,
	ModalContent,
	ModalHeader,
	ModalCloseButton,
	ModalBody,
	Table,
	IconButton,
	Image,
} from '@chakra-ui/react';
import { BsInfoCircle } from 'react-icons/bs';
import { QueueModalRow } from './QueueModalRow';
import { QueueTypeWithIndex } from '../../../atoms/atoms.types';

export const QueueModal: React.FC<QueueTypeWithIndex> = ({
	// device,
	// id,
	// key,
	// lastItem,
	// skip_grid,
	cfg_scale,
	ckpt,
	height,
	index,
	init_image,
	mask,
	n_iter,
	negative_prompts,
	sampler,
	seed,
	steps,
	strength,
	text_prompts,
	width,
}) => {
	const { isOpen, onOpen, onClose } = useDisclosure();

	return (
		<>
			<IconButton
				aria-label="View"
				icon={<BsInfoCircle />}
				onClick={onOpen}
				variant="ghost"
			/>

			<Modal
				isOpen={isOpen}
				motionPreset="slideInBottom"
				onClose={onClose}
				scrollBehavior="outside"
				size="4xl">
				<ModalOverlay />

				<ModalContent>
					<ModalHeader>Queue #{index}</ModalHeader>

					<ModalCloseButton />

					<ModalBody>
						<Table size="sm" variant="simple">
							<QueueModalRow
								name="Prompt:"
								value={text_prompts}
							/>

							<QueueModalRow
								name="Negative Prompt:"
								value={negative_prompts}
							/>

							<QueueModalRow
								name="Number of images:"
								value={n_iter}
							/>

							<QueueModalRow name="Model:" value={ckpt} />

							<QueueModalRow
								name="Dimensions:"
								value={`${width}x${height}`}
							/>

							<QueueModalRow name="Seed:" value={seed} />

							<QueueModalRow name="Steps:" value={steps} />

							<QueueModalRow name="Sampler:" value={sampler} />

							<QueueModalRow
								name="CFG Scale:"
								value={cfg_scale}
							/>

							{init_image.length > 0 ? (
								<QueueModalRow
									name="Strength:"
									value={strength}
								/>
							) : (
								<></>
							)}

							{mask?.length > 0 ? (
								<QueueModalRow name="Mask:" value={mask} />
							) : (
								<></>
							)}

							<QueueModalRow
								name="Image:"
								value={
									init_image.length > 0 &&
									init_image.length < 250
										? init_image
										: ''
								}
							/>

							<Image
								maxHeight={256}
								maxWidth={256}
								src={init_image}
							/>
						</Table>
					</ModalBody>
				</ModalContent>
			</Modal>
		</>
	);
};
