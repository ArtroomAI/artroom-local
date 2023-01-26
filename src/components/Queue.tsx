import React, { useCallback, useContext } from 'react';
import { useState, useEffect } from 'react';
import { useRecoilState } from 'recoil';
import * as atom from '../atoms/atoms';
import axios from 'axios';
import {
    Box,
    Flex,
    Button,
    Text,
    Grid,
    Table,
    Thead,
    Th,
    Tr,
    Tbody,
    HStack,
    IconButton,
    useToast
} from '@chakra-ui/react';
import {
    FaPlay,
    FaPause,
    FaStop
} from 'react-icons/fa';
import Card from '../helpers/Card';
import CardHeader from '../helpers/CardHeader';
import QueueRow from './QueueHelpers/QueueRow';
import ClearQueue from './QueueHelpers/ClearQueue';
import { SocketContext, SocketOnEvents } from '../socket';

function Queue () {
    const toast = useToast({});
    const [queue, setQueue] = useRecoilState(atom.queueState);
    const [serverRunning, setServerRunning] = useState(false);
    const socket = useContext(SocketContext);

    const getQueue = useCallback(() => {
        socket.emit('get_queue');
    }, [socket]);
    
    const startQueue = useCallback(() => {
        socket.emit('start_queue');
    }, [socket]);
    
    const pauseQueue = useCallback(() => {
        socket.emit('pause_queue');
    }, [socket]);

    const stopQueue = useCallback(() => {
        socket.emit('stop_queue');
    }, [socket]);

    // handles
    const handleGetServerStatus: SocketOnEvents['get_server_status']  = useCallback((data) => {
        setServerRunning(data.server_running);
    }, []);

    const handleGetQueue: SocketOnEvents['get_queue'] = useCallback((data) => {
        setQueue(data.queue);
    }, [setQueue]);
    
    const handleStartQueue: SocketOnEvents['start_queue'] = useCallback((data) => {
        if(data.status === 'Success') {
            setServerRunning(true);
            toast({
                title: 'Started queue',
                description: 'Queue will pick up from where it left off',
                status: 'success',
                position: 'top',
                duration: 4000,
                isClosable: false,
                containerStyle: {
                    pointerEvents: 'none'
                }
            });
        }
    }, [toast]);
    
    const handlePauseQueue: SocketOnEvents['pause_queue'] = useCallback((data) => {
        if(data.status === 'Success') {
            setServerRunning(false);
            toast({
                title: 'Paused queue',
                description: 'Will stop after current batch',
                status: 'success',
                position: 'top',
                duration: 4000,
                isClosable: false,
                containerStyle: {
                    pointerEvents: 'none'
                }
            });
        }
    }, [toast]);

    
    const handleStopQueue: SocketOnEvents['stop_queue'] = useCallback((data) => {
        if(data.status === 'Success') {
            setServerRunning(false);
            toast({
                title: 'Stopped queue',
                description: 'Current batch has been interrupted and queue has been stopped',
                status: 'success',
                position: 'top',
                duration: 4000,
                isClosable: false,
                containerStyle: {
                    pointerEvents: 'none'
                }
            });
        }
    }, [toast]);

    // on socket message
    useEffect(() => {
        socket.on('get_server_status', handleGetServerStatus);
        socket.on('get_queue', handleGetQueue);
        socket.on('start_queue', handleStartQueue);
        socket.on('pause_queue', handlePauseQueue);
        socket.on('stop_queue', handleStopQueue);
        
        socket.emit('get_server_status');
        getQueue();

        return () => {
          socket.off('get_server_status', handleGetServerStatus);
          socket.off('get_queue', handleGetQueue);
          socket.off('start_queue', handleStartQueue);
          socket.off('pause_queue', handlePauseQueue);
          socket.off('stop_queue', handleStopQueue);
        };
    }, [socket, handleGetServerStatus, handleGetQueue, handleStartQueue, handlePauseQueue, handleStopQueue, getQueue]);

    return (
        <Box
            className="queue"
            height="90%"
            ml="30px"
            p={4}
            rounded="md"
            width="100%">
            <Grid
                gap="24px"
                templateColumns="4fr 1fr">
                <Card
                    overflowX="hidden"
                    p="16px">
                    <CardHeader p="12px 0px 28px 0px">
                        <Flex direction="column">
                            <HStack pt={5}>
                                {serverRunning
                                    ? <IconButton
                                        aria-label="Pause Queue"
                                        colorScheme="yellow"
                                        icon={<FaPause />}
                                        onClick={pauseQueue} />
                                    : <IconButton
                                        aria-label="Start Queue"
                                        colorScheme="green"
                                        icon={<FaPlay />}
                                        onClick={startQueue} />}

                                <IconButton
                                    aria-label="Stop Queue"
                                    colorScheme="red"
                                    icon={<FaStop />}
                                    onClick={stopQueue} />

                                <Button
                                    className="refresh-queue-button"
                                    colorScheme="purple"
                                    onClick={getQueue}>
                                    Refresh
                                </Button>

                                <Box className="clear-queue-button">
                                    <ClearQueue />
                                </Box>
                            </HStack>

                            <Flex align="center">
                                <Text
                                    color="gray.400"
                                    fontSize="sm"
                                    fontWeight="normal">
                                    {queue.length}

                                    {' '}
                                    items in queue.
                                </Text>
                            </Flex>
                        </Flex>
                    </CardHeader>

                    <Table
                        className="queue-table"
                        color="#fff"
                        variant="simple">
                        <Thead>
                            <Tr
                                my=".8rem"
                                ps="0px">
                                <Th
                                    borderBottomColor="#56577A"
                                    color="gray.400">
                                    #
                                </Th>

                                <Th
                                    borderBottomColor="#56577A"
                                    color="gray.400"
                                    fontFamily="Plus Jakarta Display"
                                    ps="0px">
                                    Prompt
                                </Th>

                                <Th
                                    borderBottomColor="#56577A"
                                    color="gray.400">
                                    Model
                                </Th>

                                <Th
                                    borderBottomColor="#56577A"
                                    color="gray.400">
                                    Dimensions
                                </Th>

                                <Th
                                    borderBottomColor="#56577A"
                                    color="gray.400">
                                    Num
                                </Th>

                                <Th
                                    borderBottomColor="#56577A"
                                    color="gray.400">
                                    Actions
                                </Th>
                            </Tr>
                        </Thead>

                        <Tbody>
                            {queue.map((row, index, arr) => (
                                <QueueRow
                                    index={index + 1}
                                    key={index}
                                    lastItem={index === arr.length - 1}
                                    { ...row }
                                />
                            ))}
                        </Tbody>
                    </Table>
                </Card>
            </Grid>
        </Box>
    );
}

export default Queue;
