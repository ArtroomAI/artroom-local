import React from 'react';
import {
	Modal as ChakraModal,
	ModalOverlay,
	ModalContent,
	ModalHeader,
	ModalCloseButton,
} from '@chakra-ui/react';

interface IModalProps {
	isOpen: boolean;
	onClose: () => void;
	title: string;
	size?: string;
	topOffset?: number;
	children: React.ReactNode;
}

export const Modal: React.FC<IModalProps> = ({
	isOpen,
	onClose,
	title,
	size = 'xl',
	topOffset,
	children,
}) => {
	return (
		<ChakraModal size={size} isOpen={isOpen} onClose={onClose}>
			<ModalOverlay />
			<ModalContent mt={topOffset}>
				<ModalHeader>{title}</ModalHeader>
				<ModalCloseButton />
				{children}
			</ModalContent>
		</ChakraModal>
	);
};
