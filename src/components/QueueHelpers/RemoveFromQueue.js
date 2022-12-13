import React from 'react';
import { useRecoilState } from 'recoil';
import * as atom from '../../atoms/atoms';
import axios  from 'axios';
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
    FaTrashAlt,
  } from 'react-icons/fa'
function RemoveFromQueue(props) {
    const index = props.index-1

    const { isOpen, onOpen, onClose } = useDisclosure()
    const cancelRef = React.useRef()

    const [queue, setQueue] = useRecoilState(atom.queueState);

    const RemoveFromQueue = event => {
        onClose();
        axios.post('http://127.0.0.1:5300/remove_from_queue',{id: queue[index].id}, 
            {headers: {'Content-Type': 'application/json'}}).then((result)=>{
                if (result.data.status === 'Success'){
                    setQueue(result.data.content.queue);
                }
            })
    }

    return (
      <>
        <IconButton background='transparent' onClick={onOpen} aria-label='Remove from Queue' icon={<FaTrashAlt/>}></IconButton>
        <AlertDialog
            isOpen={isOpen}
            leastDestructiveRef={cancelRef}
            onClose={onClose}
            returnFocusOnClose = {true}
            >
            <AlertDialogOverlay>
                <AlertDialogContent bg='gray.800'>
                <AlertDialogHeader fontSize='lg' fontWeight='bold'>
                    Remove From Queue
                </AlertDialogHeader>
    
                <AlertDialogBody>
                    Are you sure you want to remove this item from the queue?
                </AlertDialogBody>
    
                <AlertDialogFooter>
                    <Button ref={cancelRef} onClick={onClose}>
                    Cancel
                    </Button>
                    <Button colorScheme='red' onClick={RemoveFromQueue} ml={3}>
                    Remove
                    </Button>
                </AlertDialogFooter>
                </AlertDialogContent>
            </AlertDialogOverlay>
        </AlertDialog>      
      </>
    )
  }

export default RemoveFromQueue;