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
    Button,
    IconButton
} from '@chakra-ui/react';
import {
    FaTrashAlt
} from 'react-icons/fa';
function RemoveFromQueue (props) {
    const index = props.index - 1;

    const { isOpen, onOpen, onClose } = useDisclosure();
    const cancelRef = React.useRef();

    const [queue, setQueue] = useRecoilState(atom.queueState);

    const RemoveFromQueue = (event) => {
        onClose();
        axios.post(
            'http://127.0.0.1:5300/remove_from_queue',
            { id: queue[index].id },
            { headers: { 'Content-Type': 'application/json' } }
        ).then((result) => {
            if (result.data.status === 'Success') {
                setQueue(result.data.content.queue);
            }
        });
    };

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
                                onClick={RemoveFromQueue}>
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
