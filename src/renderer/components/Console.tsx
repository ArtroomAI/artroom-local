import React, { useCallback, useContext, useEffect, useState } from 'react';
import { Box, Flex, Icon, Modal, ModalBody, ModalContent, ModalHeader, ModalOverlay } from '@chakra-ui/react';
import { SocketContext, SocketOnEvents } from '../socket';
import { FaTerminal } from 'react-icons/fa';

type MessageType<K extends keyof SocketOnEvents> = Parameters<SocketOnEvents[K]>[0];

function parseSocketMessage<T extends keyof SocketOnEvents>(type: T, message: MessageType<T>) {
    return JSON.stringify(message);
    // TODO: Custom messages for each event
    if(type === 'get_images') {
        const typed = message as MessageType<'get_images'>;
        return;
    }
}

export const Console = () => {
    const socket = useContext(SocketContext);
    const [log, setLog] = useState<string[]>([]);
    const [logPos, setLogPos] = useState<number>(0);
    const [showConsole, setShowConsole] = useState(false);
    
    function handleClose() {
        setShowConsole(false)
    }
    function handleOpen() {
        setShowConsole(true)
    }

    const handleMessage = useCallback((type: keyof SocketOnEvents, message: MessageType<keyof SocketOnEvents>) => {
        console.log(type, message);
        log[logPos] = `${type} : ${parseSocketMessage(type, message)}`;
        setLog(log);
        if(logPos === 100) {
            setLogPos(0);
        } else {
            setLogPos(logPos + 1);
        }
    }, [log, logPos]);

    useEffect(() => {
        // subscribe to socket events
        socket.onAny(handleMessage); 
    
        return () => {
          // before the component is destroyed
          // unbind all event handlers used in this component
          socket.offAny(handleMessage);
        };
    }, [handleMessage, socket]);

    const displayMessages = useCallback(() => {
        const messages = [];
        for(let i = logPos; i < log.length; ++i) {
            messages.push(<Flex w="100%">{log[i]}</Flex>);
        }
        for(let i = 0; i < logPos; ++i) {
            messages.push(<Flex w="100%">{log[i]}</Flex>);
        }
        return messages;
    }, [log, logPos])

    return (<>
        <Box pos="fixed" bottom="1" right="1" width="50px" height="50px" bgColor='#000' onClick={handleOpen}>
            <Icon as={FaTerminal} fontSize="xl" justifyContent="center" />
        </Box>
        <Modal
            size='6xl'
            isOpen={showConsole}
            onClose={handleClose}
            scrollBehavior='outside'>
                <ModalOverlay bg='blackAlpha.900' />
                <ModalContent>
                    <ModalHeader />
                    <ModalBody>
                        {displayMessages()}
                    </ModalBody>
                </ModalContent>
        </Modal>
    </>);
};
