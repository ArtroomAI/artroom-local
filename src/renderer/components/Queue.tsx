import React, { useCallback, useContext } from 'react';
import { useRecoilState } from 'recoil';
import * as atom from '../atoms/atoms';
import {
    Box,
    Flex,
    Text,
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
import Card from '../helpers/Card';
import CardHeader from '../helpers/CardHeader';
import { SortableItem } from './QueueHelpers/QueueRow';
import ClearQueue from './QueueHelpers/ClearQueue';
import { SocketContext } from '../socket';
import {
    DndContext, 
    closestCenter,
    KeyboardSensor,
    PointerSensor,
    useSensor,
    useSensors,
    DragEndEvent,
} from '@dnd-kit/core';
import {
    arrayMove,
    SortableContext,
    sortableKeyboardCoordinates,
    verticalListSortingStrategy
} from '@dnd-kit/sortable';

function Queue () {
    const [queue, setQueue] = useRecoilState(atom.queueState);
    const [isQueuePaused, setIsQueuePaused] = useRecoilState(atom.queuePausedState);
    const socket = useContext(SocketContext);
    
    const startQueue = useCallback(() => {
        setIsQueuePaused(false);
    }, []);
    
    const pauseQueue = useCallback(() => {
        setIsQueuePaused(true);
    }, []);

    const stopQueue = useCallback(() => {
        socket.emit('stop_queue');
    }, [socket]);

    function handleDragEnd(event: DragEndEvent) {
        const {active, over} = event;
        
        if (active.id !== over.id) {
            setQueue((items) => {
                const oldIndex = items.findIndex(el => el.id === active.id);
                const newIndex = items.findIndex(el => el.id === over.id);
                console.log(oldIndex, newIndex)
                console.log(JSON.stringify(items.map(e => e.id)));
                console.log(JSON.stringify(arrayMove(items, oldIndex, newIndex).map(e => e.id)));
                return [...arrayMove(items, oldIndex, newIndex)];
            });
        }
    }

    const sensors = useSensors(
        useSensor(PointerSensor),
        useSensor(KeyboardSensor, {
          coordinateGetter: sortableKeyboardCoordinates,
        })
    );

    return (
        <Box
            className="queue"
            height="90%"
            ml="30px"
            p={4}
            rounded="md"
            width="100%">
                <Card
                    overflowX="hidden"
                    p="16px">
                    <CardHeader p="12px 0px 28px 0px">
                        <Flex direction="column">
                            <HStack pt={5}>
                                {!isQueuePaused
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

                    <DndContext 
                        sensors={sensors}
                        collisionDetection={closestCenter}
                        onDragEnd={handleDragEnd}
                    >
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
                                    </Th>
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
                                <SortableContext 
                                    items={queue}
                                    strategy={verticalListSortingStrategy}
                                >
                                    {queue.map((elem, index, arr) => <SortableItem key={elem.id} id={elem.id} index={index} row={elem} arr={arr} />)}
                                </SortableContext>
                            </Tbody>
                        </Table>
                    </DndContext>
                </Card>
        </Box>
    );
}

export default Queue;
