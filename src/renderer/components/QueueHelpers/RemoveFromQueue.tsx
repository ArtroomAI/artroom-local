import React, { useCallback } from 'react';
import { useSetRecoilState } from 'recoil';
import * as atom from '../../atoms/atoms';
import {
    useDisclosure,
    AlertDialog,
    AlertDialogOverlay,
    AlertDialogContent,
    AlertDialogHeader,
    AlertDialogBody,
    AlertDialogFooter,
    Button,
    IconButton
} from '@chakra-ui/react';
import {
    FaTrashAlt
} from 'react-icons/fa';

function RemoveFromQueue ({ index } : { index: number }) {
    const { isOpen, onOpen, onClose } = useDisclosure();
    const cancelRef = React.useRef();

    const setQueue = useSetRecoilState(atom.queueState);
    const removeFromQueue = useCallback(() => {
        setQueue((queue) => {
            return queue.filter((_, i) => i !== index - 1);
        });
        onClose();
    }, [onClose, index]);

    return (
        <>
            <IconButton
                aria-label="Remove from Queue"
                background="transparent"
                icon={<FaTrashAlt />}
                onClick={onOpen} />

            <AlertDialog
                isOpen={isOpen}
                leastDestructiveRef={cancelRef}
                onClose={onClose}
                returnFocusOnClose
            >
                <AlertDialogOverlay>
                    <AlertDialogContent bg="gray.800">
                        <AlertDialogHeader
                            fontSize="lg"
                            fontWeight="bold">
                            Remove From Queue
                        </AlertDialogHeader>

                        <AlertDialogBody>
                            Are you sure you want to remove this item from the queue?
                        </AlertDialogBody>

                        <AlertDialogFooter>
                            <Button
                                onClick={onClose}
                                ref={cancelRef}>
                                Cancel
                            </Button>

                            <Button
                                colorScheme="red"
                                ml={3}
                                onClick={removeFromQueue}>
                                Remove
                            </Button>
                        </AlertDialogFooter>
                    </AlertDialogContent>
                </AlertDialogOverlay>
            </AlertDialog>
        </>
    );
}

export default RemoveFromQueue;
