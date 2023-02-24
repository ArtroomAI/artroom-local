import React, { useCallback } from 'react';
import { useRecoilState } from 'recoil';
import * as atom from '../../atoms/atoms';
import {
    useDisclosure,
    AlertDialog,
    AlertDialogOverlay,
    AlertDialogContent,
    AlertDialogHeader,
    AlertDialogBody,
    AlertDialogFooter,
    Button
} from '@chakra-ui/react';

function ClearQueue () {
    const { isOpen, onOpen, onClose } = useDisclosure();
    const cancelRef = React.useRef();

    const [queue, setQueue] = useRecoilState(atom.queueState);

    const clearQueue = useCallback(() => {
        if(queue[0]) {
            setQueue([queue[0]]);
        }
        onClose();
    }, [onClose, queue]);

    return (
        <>
            <Button
                colorScheme="yellow"
                onClick={onOpen}>
                Delete Queue
            </Button>

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
                            Delete Queue
                        </AlertDialogHeader>

                        <AlertDialogBody>
                            Are you sure you want to clear the queue?
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
                                onClick={clearQueue}>
                                Clear
                            </Button>
                        </AlertDialogFooter>
                    </AlertDialogContent>
                </AlertDialogOverlay>
            </AlertDialog>
        </>
    );
}

export default ClearQueue;
