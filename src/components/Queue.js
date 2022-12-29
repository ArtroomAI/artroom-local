import React from 'react';
import { useState, useEffect } from 'react';
import { useRecoilState } from 'recoil';
import * as atom from '../atoms/atoms';
import axios from 'axios';
import {
    Box,
    Flex,
    createStandaloneToast,
    Button,
    Text,
    Grid,
    Table,
    Thead,
    Th,
    Tr,
    Tbody,
    HStack,
    IconButton
} from '@chakra-ui/react';
import {
    FaPlay,
    FaPause,
    FaStop
} from 'react-icons/fa';
import Card from '../helpers/Card.js';
import CardHeader from '../helpers/CardHeader.js';
import QueueRow from './QueueHelpers/QueueRow.js';
import ClearQueue from './QueueHelpers/ClearQueue.js';
function Queue () {
    const { ToastContainer, toast } = createStandaloneToast();
    const [queue, setQueue] = useRecoilState(atom.queueState);
    const [serverRunning, setServerRunning] = useState(false);

    useEffect(
        () => {
            getQueue();
            getServerStatus();
        },
        []
    );

    const getServerStatus = (event) => {
        axios.get(
            'http://127.0.0.1:5300/get_server_status',
            { headers: { 'Content-Type': 'application/json' } }
        ).then((result) => {
            if (result.data.status === 'Success') {
                setServerRunning(result.data.content.server_running);
            }
        });
    };
    const getQueue = (event) => {
        axios.get(
            'http://127.0.0.1:5300/get_queue',
            { headers: { 'Content-Type': 'application/json' } }
        ).then((result) => {
            if (result.data.status === 'Success') {
                setQueue(result.data.content.queue);
            }
        });
    };

    const startQueue = (event) => {
        axios.get(
            'http://127.0.0.1:5300/start_queue',
            { headers: { 'Content-Type': 'application/json' } }
        ).then((result) => {
            if (result.data.status === 'Success') {
                setServerRunning(true);
                toast({
                    title: 'Started queue',
                    desciption: 'Queue will pick up from where it left off',
                    status: 'success',
                    position: 'top',
                    duration: 4000,
                    isClosable: false,
                    containerStyle: {
                        pointerEvents: 'none'
                    }
                });
            }
        });
    };

    const pauseQueue = (event) => {
        axios.get(
            'http://127.0.0.1:5300/pause_queue',
            { headers: { 'Content-Type': 'application/json' } }
        ).then((result) => {
            if (result.data.status === 'Success') {
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
        });
    };

    const stopQueue = (event) => {
        axios.get(
            'http://127.0.0.1:5300/stop_queue',
            { headers: { 'Content-Type': 'application/json' } }
        ).then((result) => {
            if (result.data.status === 'Success') {
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
        });
    };

    return (
        <Box
            className="queue"
            width="100%">
            <Card
                overflowX="hidden"
                p="24px">
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
                                {queue?.length}

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
                        {queue?.map((row, index, arr) => (
                            <QueueRow
                                H={row.height}
                                W={row.width}
                                cfg_scale={row.cfg_scale}
                                ckpt={row.ckpt}
                                device={row.device}
                                id={row.id}
                                index={index + 1}
                                init_image={row.init_image}
                                invert={row.invert}
                                keep_warm={row.keep_warm}
                                key={index}
                                lastItem={index === arr.length - 1}
                                mask={row.mask}
                                n_iter={row.n_iter}
                                negative_prompt={row.negative_prompts}
                                precision={row.precision}
                                prompt={row.text_prompts}
                                sampler={row.sampler}
                                seed={row.seed}
                                skip_grid={row.skip_grid}
                                steps={row.steps}
                            />
                        ))}
                    </Tbody>
                </Table>
            </Card>
        </Box>
    );
}

export default Queue;
