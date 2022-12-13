import React from 'react';
import { useRecoilState } from 'recoil';
import * as atom from '../src/atoms/atoms';
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

function StartQueue() {
    const { isOpen, onOpen, onClose } = useDisclosure()
    const cancelRef = React.useRef()

    const [queue, setQueue] = useRecoilState(atom.queueState);
    const [delay, setDelay] = useRecoilState(atom.delayState);

    const startQueue = event => {
        onClose();
        let newQueue = {"Queue": queue, "Delay": delay}
        setQueueRunning(false);
        window['writeQueue'](newQueue)
    }

    const runQueue = event => {
        onClose();
        let newQueue = {"Queue": queue, "Running": false, "Keep_Warm": true}
        setQueueRunning(false);
        setKeepWarm(true);
        window['writeQueue'](newQueue).then(()=>{
            
        })
    }

    return (
      <>
        <Button bg='green' colorScheme='green' onClick={onOpen}>
          Start Queue
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
                    Turn On Queue
                    </AlertDialogHeader>
        
                    <AlertDialogBody>
                        Next time you generate, it will preload the model and use the queue sytem. Items added can be viewed in the queue manager below.
                    </AlertDialogBody>
        
                    <AlertDialogFooter>
                        <Button ref={cancelRef} onClick={onClose}>
                        Cancel
                        </Button>
                        <Button colorScheme='green' onClick={startQueue} ml={3}>
                        Confirm
                        </Button>
                    </AlertDialogFooter>
                    </AlertDialogContent>
                </AlertDialogOverlay>
            </AlertDialog>      
      </>
    )
  }

export default StartQueue;