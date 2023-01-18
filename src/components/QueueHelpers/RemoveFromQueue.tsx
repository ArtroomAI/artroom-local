import React, { useCallback, useContext, useEffect } from 'react';
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
    Button,
    IconButton
} from '@chakra-ui/react';
import {
    FaTrashAlt
} from 'react-icons/fa';
import { SocketContext } from '../..';

function RemoveFromQueue ({ index } : { index: number }) {
    const { isOpen, onOpen, onClose } = useDisclosure();
    const cancelRef = React.useRef();

    const [queue, setQueue] = useRecoilState(atom.queueState);

    const socket = useContext(SocketContext);

    const removeFromQueue = useCallback(() => {
        onClose();
        socket.emit('remove_from_queue', { id: queue[index - 1].id });
    }, [onClose, socket, queue, index]);

    const handleRemoveFromQueue = useCallback((data: { status: 'Success' | 'Failure'; queue: QueueType[] }) => {
        if (data.status === 'Success') {
            setQueue(data.queue);
        }
    }, [setQueue]);

    // on socket message
    useEffect(() => {
        socket.on('remove_from_queue', handleRemoveFromQueue);
    
        return () => {
            socket.off('remove_from_queue', handleRemoveFromQueue);
        };
    }, [socket, handleRemoveFromQueue]);

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
