import React, { useCallback } from 'react'
import { useRecoilState, useRecoilValue } from 'recoil'
import * as atom from '../../atoms/atoms'
import {
  useDisclosure,
  AlertDialog,
  AlertDialogOverlay,
  AlertDialogContent,
  AlertDialogHeader,
  AlertDialogBody,
  AlertDialogFooter,
  Button,
} from '@chakra-ui/react'
import { artroomPathState } from '../../SettingsManager'

function ClearQueue() {
  const { isOpen, onOpen, onClose } = useDisclosure()
  const cancelRef = React.useRef(null)

  const [queue, setQueue] = useRecoilState(atom.queueState)
  const artroomPath = useRecoilValue(artroomPathState)

  const clearQueue = useCallback(() => {
    setQueue([])
    window.api.saveQueue(JSON.stringify([]), artroomPath)
    onClose()
  }, [onClose, queue, artroomPath])

  return (
    <>
      <Button colorScheme="yellow" onClick={onOpen}>
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
            <AlertDialogHeader fontSize="lg" fontWeight="bold">
              Delete Queue
            </AlertDialogHeader>

            <AlertDialogBody>Are you sure you want to clear the queue?</AlertDialogBody>

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
