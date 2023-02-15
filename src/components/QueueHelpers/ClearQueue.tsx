import React, { useCallback, useContext, useEffect } from 'react'
import { useRecoilState } from 'recoil'
import * as atom from '../../atoms/atoms'
import {
  useDisclosure,
  AlertDialog,
  AlertDialogOverlay,
  AlertDialogContent,
  AlertDialogHeader,
  AlertDialogBody,
  AlertDialogFooter,
  Button
} from '@chakra-ui/react'
import { SocketContext, SocketOnEvents } from '../../socket'

function ClearQueue() {
  const { isOpen, onOpen, onClose } = useDisclosure()
  const cancelRef = React.useRef()

  const socket = useContext(SocketContext)

  const [queue, setQueue] = useRecoilState(atom.queueState)

  const handleClearQueue: SocketOnEvents['clear_queue'] = useCallback(
    data => {
      if (data.status === 'Success') {
        setQueue([])
      }
      onClose()
    },
    [onClose, setQueue]
  )

  const clearQueue = useCallback(() => {
    socket.emit('clear_queue')
  }, [socket])

  useEffect(() => {
    socket.on('clear_queue', handleClearQueue)

    return () => {
      socket.off('clear_queue', handleClearQueue)
    }
  }, [socket, handleClearQueue])

  return (
    <>
      <Button colorScheme="yellow" onClick={onOpen}>
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
            <AlertDialogHeader fontSize="lg" fontWeight="bold">
              Clear Queue
            </AlertDialogHeader>

            <AlertDialogBody>
              Are you sure you want to clear the queue?
            </AlertDialogBody>

            <AlertDialogFooter>
              <Button onClick={onClose} ref={cancelRef}>
                Cancel
              </Button>

              <Button colorScheme="red" ml={3} onClick={clearQueue}>
                Clear
              </Button>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialogOverlay>
      </AlertDialog>
    </>
  )
}

export default ClearQueue
