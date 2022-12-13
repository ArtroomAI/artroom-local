import React from 'react';
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
    Button,
} from "@chakra-ui/react";

function StopQueue() {
    const { isOpen, onOpen, onClose } = useDisclosure()
    const cancelRef = React.useRef()

    const [queue, setQueue] = useRecoilState(atom.queueState);
    const [queueRunning, setQueueRunning] = useRecoilState(atom.queueRunningState);
    const [keep_warm, setKeepWarm] = useRecoilState(atom.keepWarmState);
    const [delay, setDelay] = useRecoilState(atom.delayState);

    const stopQueue = event => {
        onClose();
        let newQueue = {"Queue": queue, "Running": false, "Keep_Warm": false, "Delay": delay}
        setQueueRunning(false);
        setKeepWarm(false);
        window['writeQueue'](newQueue)
    }
    return (
      <>
        <Button bg='red' colorScheme='red' onClick={onOpen}>
          Stop Queue
        </Button>
            <AlertDialog
                isOpen={isOpen}
                leastDestructiveRef={cancelRef}
                onClose={onClose}
                returnFocusOnClose = {true}
                >
                <AlertDialogOverlay>
                    <AlertDialogContent>
                    <AlertDialogHeader fontSize='lg' fontWeight='bold'>
                        Turn Off Queue
                    </AlertDialogHeader>
        
                    <AlertDialogBody>
                        Switch to loading in and generating one image at a time.
                    </AlertDialogBody>
        
                    <AlertDialogFooter>
                        <Button ref={cancelRef} onClick={onClose}>
                        Cancel
                        </Button>
                        <Button colorScheme='red' onClick={stopQueue} ml={3}>
                        Confirm
                        </Button>
                    </AlertDialogFooter>
                    </AlertDialogContent>
                </AlertDialogOverlay>
            </AlertDialog>      
      </>
    )
  }

export default StopQueue;