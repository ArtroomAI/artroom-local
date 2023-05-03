import React, { useEffect, useContext, useCallback, useState } from 'react';
import { useRecoilState, useRecoilValue, useSetRecoilState } from 'recoil';
import * as atom from '../atoms/atoms';
import {
    Box,
    SimpleGrid,
    useToast,
    Image as ChakraImage,
    VStack,
    Button,
    HStack,
    IconButton,
    Text
} from '@chakra-ui/react';
import ImageObj from './Reusable/ImageObj';
import ProtectedReqManager from '../helpers/ProtectedReqManager';
import { SocketContext } from '../socket';
import { queueSettingsSelector, randomSeedState, seedState } from '../SettingsManager';
import { addToQueueState, cloudModeState } from '../atoms/atoms';
import { ImageState } from '../atoms/atoms.types';
import { parseSettings } from './Utils/utils';
import { GenerationProgressBars } from './Reusable/GenerationProgressBars';
import { latestImagesDispatcher, useKeyPress } from './latestImagesDispatcher';
import { FaStop } from 'react-icons/fa';
import Prompt from './Prompt';
import { computeShardCost } from './Utils/shardCost';
import Shards from '../images/shards.png';

const Body = () => {
    const ARTROOM_URL = process.env.REACT_APP_ARTROOM_URL;
    const toast = useToast({});

    const [focused, setFocused] = useState(false);

    const [mainImage, setMainImage] = useRecoilState(atom.mainImageState);
    const latestImages = useRecoilValue(atom.latestImageState);
    
    const socket = useContext(SocketContext);

    const useRandomSeed = useRecoilValue(randomSeedState);
    const [queue, setQueue] = useRecoilState(atom.queueState);
    const setAddToQueue = useSetRecoilState(addToQueueState);
    const imageSettings = useRecoilValue(queueSettingsSelector);
    const setCloudRunning = useSetRecoilState(atom.cloudRunningState);
    const setShard = useSetRecoilState(atom.shardState);
    const setSeed = useSetRecoilState(seedState);
    const cloudMode = useRecoilValue(cloudModeState);

    const altRPressed = useKeyPress('r', true);

    const addToQueue = useCallback(() => {
        toast({
            title: 'Added to Queue!',
            description: `Currently ${queue.length + 1} elements in queue`,
            status: 'success',
            position: 'top',
            duration: 2000,
            isClosable: false
        });
        setAddToQueue(true);
        const settings = parseSettings(
            {
                ...imageSettings,
                id: `${Math.floor(Math.random() * Number.MAX_SAFE_INTEGER)}`,
                palette_fix: false
            },
            useRandomSeed
        );

        if(useRandomSeed) setSeed(settings.seed);

        setQueue((queue) => [...queue, settings]);
    }, [imageSettings, queue, toast]);

    const submitCloud = useCallback(() => {
        ProtectedReqManager.make_post_request(`${ARTROOM_URL}/gpu/submit_job_to_queue`, imageSettings).then((response: any) => {
            setShard(response.data.shard_balance);
            toast({
                title: 'Job Submission Success',
                status: 'success',
                position: 'top',
                duration: 5000,
                isClosable: true
            });
            setCloudRunning(true);
        }).catch((err: any) => {
            console.log(err);
            toast({
                title: 'Error',
                status: 'error',
                description: err.response.data.detail,
                position: 'top',
                duration: 5000,
                isClosable: true
            });
        });
    }, [ARTROOM_URL, imageSettings, toast]);

    const stopQueue = useCallback(() => {
        socket.emit('stop_queue');
    }, [socket]);
    
    useEffect(() => {
        if (altRPressed) {
            addToQueue();
        }
    }, [addToQueue, altRPressed]);

    const [state, dispatch] = latestImagesDispatcher();

    const arrowRightPressed = useKeyPress('ArrowRight');
    const arrowLeftPressed = useKeyPress('ArrowLeft');

    useEffect(() => {
        if (arrowRightPressed && !focused) {
            dispatch({ type: 'arrowRight' });
        }
    }, [arrowRightPressed, focused]);

    useEffect(() => {
        if (arrowLeftPressed && !focused) {
            dispatch({ type: 'arrowLeft' });
        }
    }, [arrowLeftPressed, focused]);

    const handleIntermediateImages = useCallback((data: ImageState) => {
        dispatch({ type: 'intermediate' });
        setMainImage(data);
    }, [setMainImage])

    useEffect(() => {
        socket.on('intermediate_image', handleIntermediateImages); 

        return () => {
          socket.off('intermediate_image', handleIntermediateImages);
        };
    }, [socket, handleIntermediateImages]);

    useEffect(() => {
        if(state.selectedIndex !== -1) {
            setMainImage(latestImages[state.selectedIndex]);
        } 
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [state.selectedIndex]);

    const getProfile = () => {
        ProtectedReqManager.make_get_request(`${ARTROOM_URL}/users/me`).then((response: { data: { email: string; }; }) => {
            console.log(response);
            toast({
                title: 'Auth Test Success: ' + response.data.email,
                status: 'success',
                position: 'top',
                duration: 2000,
                isClosable: true,
                containerStyle: {
                    pointerEvents: 'none'
                }
            });
        }).catch((err: any) => {
            console.log(err);
        });
    };

    return (
        <Box width="100%">
            <VStack align="center" spacing={4}>
                <Box
                    className="image-box"
                    width="80%"
                    display="flex"
                    alignItems="center"
                    justifyContent="center"
                    flexDirection="column"
                    >
                    <ImageObj
                        b64={mainImage?.b64}
                        path={mainImage?.path}
                        active />
                    <GenerationProgressBars />
                </Box>
                <Box
                    maxHeight="120px"
                    overflowY="auto"
                    width="60%">
                    <SimpleGrid
                        minChildWidth="100px"
                        spacing="10px">
                        {latestImages.map((imageData, index) => (<ChakraImage
                            h="5vh"
                            key={index}
                            src={imageData.b64}
                            onClick={() => {
                                dispatch({ type: 'select', payload: index })
                            }}
                        />))}
                    </SimpleGrid>
                </Box>
                {cloudMode
                    ? <Button
                        className="run-button"
                        ml={2}
                        onClick={submitCloud}
                        variant="outline"
                        width="200px">
                        <Text pr={2}>
                            Run
                        </Text>

                        <ChakraImage
                            src={Shards}
                            width="12px" />

                        <Text pl={1}>
                            {computeShardCost(imageSettings)}
                        </Text>
                    </Button>
                    : 
                    <HStack>
                        <Button
                            className="run-button"
                            ml={2}
                            onClick={addToQueue}
                            width="200px"> 
                            Run
                        </Button>
                        <IconButton
                            aria-label="Stop Queue"
                            colorScheme="red"
                            icon={<FaStop />}
                            onClick={stopQueue} />
                        </HStack>
                    }
                <Box width="80%">
                    <Prompt setFocused={setFocused} />
                </Box>
            </VStack>
        </Box>
    );
};

export default Body;
