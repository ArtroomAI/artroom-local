import React from 'react';
import { useRecoilState } from 'recoil';
import * as atom from '../../atoms/atoms';
import axios from 'axios';
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

    const clearQueue = (event) => {
        axios.post(
            'http://127.0.0.1:5300/clear_queue',
            {}
        );
        setQueue([]);
        onClose();
    };

    return (
        <>
            <Button
                colorScheme="yellow"
                onClick={onOpen}>
                Clear Queue
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
                            Clear Queue
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
