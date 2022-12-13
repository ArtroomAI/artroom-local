import {useState, useEffect} from 'react';
import { useRecoilState } from 'recoil';
import * as atom from '../atoms/atoms';
import axios  from 'axios';
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
import Card from '../helpers/Card.js'
import CardHeader from '../helpers/CardHeader.js'
import QueueRow from './QueueHelpers/QueueRow.js'
import ClearQueue from './QueueHelpers/ClearQueue.js'
function Queue() {
    const { ToastContainer, toast } = createStandaloneToast()
    const [navSize, changeNavSize] = useRecoilState(atom.navSizeState);
    const [queue, setQueue] = useRecoilState(atom.queueState);
    const [serverRunning, setServerRunning] = useState(false);

    useEffect(()=> {
        getQueue();
        getServerStatus();
    }, [])

    const getServerStatus = event => {
        axios.get('http://127.0.0.1:5300/get_server_status',
            {headers: {'Content-Type': 'application/json'}}).then((result)=>{
                if (result.data.status === 'Success'){
                    setServerRunning(result.data.content.server_running);
                }
            })
    }
    const getQueue = event => {
        axios.get('http://127.0.0.1:5300/get_queue',
            {headers: {'Content-Type': 'application/json'}}).then((result)=>{
                if (result.data.status === 'Success'){
                    setQueue(result.data.content.queue);
                }
            })
    }

    const startQueue = event => {
        axios.get('http://127.0.0.1:5300/start_queue',
            {headers: {'Content-Type': 'application/json'}}).then((result)=>{
                if(result.data.status === 'Success'){
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
                        },
                    })
                }
            })
    }

    const pauseQueue = event => {
        axios.get('http://127.0.0.1:5300/pause_queue',
            {headers: {'Content-Type': 'application/json'}}).then((result)=>{
            if(result.data.status === 'Success'){
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
                    },
                })
            }
            })
    }

    const stopQueue = event => {
        axios.get('http://127.0.0.1:5300/stop_queue',
            {headers: {'Content-Type': 'application/json'}}).then((result)=>{
            if(result.data.status === 'Success'){
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
                    },
                })
            }
            })
    }

    return (
        <Flex transition='all .25s ease' ml={navSize === 'large' ? '240px' : '100px'} align='center' justify='center' width='100%'>
            <Box className='queue' ml='30px' p={4} width='100%' height='90%' rounded='md'>
                <Grid templateColumns='4fr 1fr' gap='24px'>
                    <Card p='16px' overflowX='hidden'>
                        <CardHeader p='12px 0px 28px 0px'>
                            <Flex direction='column'>
                                <HStack pt={5}>
                                    {serverRunning ?
                                    <IconButton colorScheme='yellow' onClick = {pauseQueue} aria-label='Pause Queue' icon={<FaPause/>}></IconButton>
                                    :
                                    <IconButton colorScheme='green' onClick = {startQueue} aria-label='Start Queue' icon={<FaPlay/>}></IconButton>
                                    }
                                    <IconButton colorScheme='red' onClick = {stopQueue} aria-label='Stop Queue' icon={<FaStop/>}></IconButton>
                                    <Button className='refresh-queue-button' onClick={getQueue} colorScheme='purple'>
                                            Refresh
                                    </Button>
                                    <Box className='clear-queue-button'>
                                        <ClearQueue />
                                    </Box>
                                </HStack>
                                <Flex align='center'>
                                    <Text fontSize='sm' color='gray.400' fontWeight='normal'>
                                        {queue?.length}{' '}items in queue.
                                    </Text>
                                </Flex>
                            </Flex>
                        </CardHeader>
                        <Table className='queue-table' variant='simple' color='#fff'>
                            <Thead>
                                <Tr my='.8rem' ps='0px'>
                                    <Th color='gray.400' borderBottomColor='#56577A'>
                                        #
                                    </Th>
                                    <Th
                                        ps='0px'
                                        color='gray.400'
                                        fontFamily='Plus Jakarta Display'
                                        borderBottomColor='#56577A'>
                                        Prompt
                                    </Th>
                                    <Th color='gray.400' borderBottomColor='#56577A'>
                                        Model
                                    </Th>
                                    <Th color='gray.400' borderBottomColor='#56577A'>
                                        Dimensions
                                    </Th>
                                    <Th color='gray.400' borderBottomColor='#56577A'>
                                        Num
                                    </Th>
                                    <Th color='gray.400' borderBottomColor='#56577A'>
                                        Actions
                                    </Th>
                                </Tr>
                            </Thead>
                            <Tbody>
                                {queue?.map((row, index, arr) => {
                                    return (
                                        <QueueRow
                                            index={index+1}
                                            id={row.id}
                                            prompt={row.text_prompts}
                                            negative_prompt={row.negative_prompts}
                                            skip_grid={row.skip_grid}
                                            steps={row.steps}
                                            n_iter={row.n_iter}
                                            H={row.height}
                                            W={row.width}
                                            cfg_scale={row.cfg_scale}
                                            seed={row.seed}
                                            precision={row.precision}
                                            ckpt={row.ckpt}
                                            device={row.device}
                                            sampler={row.sampler}
                                            init_image={row.init_image}
                                            mask={row.mask}
                                            invert={row.invert}
                                            keep_warm={row.keep_warm}
                                            lastItem={index === arr.length - 1 ? true : false}
                                        />
                                    );
                                })}
                            </Tbody>
                        </Table>
                    </Card>
                </Grid>
            </Box>
        </Flex>
    )
}

export default Queue;
