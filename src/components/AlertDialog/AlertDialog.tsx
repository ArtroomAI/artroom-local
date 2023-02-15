import {
	AlertDialog as ChakraAlertDialog,
	AlertDialogBody,
	AlertDialogContent,
	AlertDialogFooter,
	AlertDialogHeader,
	AlertDialogOverlay,
	Button,
	forwardRef,
	useDisclosure,
	ButtonProps,
} from '@chakra-ui/react';
import { cloneElement, ReactElement, ReactNode, useRef } from 'react';

type Props = {
	acceptButtonText?: string;
	acceptCallback: () => void;
	cancelButtonText?: string;
	cancelCallback?: () => void;
	children: ReactNode;
	title: string;
	triggerComponent: ReactElement;
	acceptButtonProps?: ButtonProps;
};

export const AlertDialog = forwardRef(
	(
		{
			acceptButtonText = 'Accept',
			acceptCallback,
			cancelButtonText = 'Cancel',
			cancelCallback,
			children,
			title,
			triggerComponent,
			acceptButtonProps,
		}: Props,
		ref,
	) => {
		const { isOpen, onOpen, onClose } = useDisclosure();
		const cancelRef = useRef<HTMLButtonElement | null>(null);

		const handleAccept = () => {
			acceptCallback();
			onClose();
		};

		const handleCancel = () => {
			cancelCallback && cancelCallback();
			onClose();
		};

		return (
			<>
				{cloneElement(triggerComponent, {
					onClick: onOpen,
					ref: ref,
				})}

				<ChakraAlertDialog
					isOpen={isOpen}
					leastDestructiveRef={cancelRef}
					onClose={onClose}>
					<AlertDialogOverlay zIndex={1999}>
						<AlertDialogContent>
							<AlertDialogHeader fontSize="lg" fontWeight="bold">
								{title}
							</AlertDialogHeader>

							<AlertDialogBody>{children}</AlertDialogBody>

							<AlertDialogFooter>
								<Button ref={cancelRef} onClick={handleCancel}>
									{cancelButtonText}
								</Button>
								<Button
									colorScheme="red"
									onClick={handleAccept}
									variant="outline"
									_hover={{
										color: 'red.300',
										borderColor: 'red.300',
									}}
									ml={3}
									{...acceptButtonProps}>
									{acceptButtonText}
								</Button>
							</AlertDialogFooter>
						</AlertDialogContent>
					</AlertDialogOverlay>
				</ChakraAlertDialog>
			</>
		);
	},
);
