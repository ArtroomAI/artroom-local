import React, { useState, useEffect, useReducer, useContext, useCallback } from 'react';
import { useRecoilState, useRecoilValue, useSetRecoilState } from 'recoil';
import * as atom from '../atoms/atoms';
import {
    Box,
    Button,
    VStack,
    SimpleGrid,
    Image,
    Text,
    useToast,
    HStack,
    IconButton
} from '@chakra-ui/react';
import ImageObj from './Reusable/ImageObj';
import Prompt from './Prompt';
import Shards from '../images/shards.png';
import ProtectedReqManager from '../helpers/ProtectedReqManager';
import { SocketContext } from '../socket';
import { queueSettingsSelector, randomSeedState, seedState } from '../SettingsManager';
import { addToQueueState } from '../atoms/atoms';
import { FaStop } from 'react-icons/fa';
import { ImageState } from '../atoms/atoms.types';
import { parseSettings } from './Utils/utils';
import { GenerationProgressBars } from './Reusable/GenerationProgressBars';

const QueueButtons = () => {
    const ARTROOM_URL = process.env.REACT_APP_ARTROOM_URL;

    const toast = useToast({
        containerStyle: {
            pointerEvents: 'none'
        }
    });

    const socket = useContext(SocketContext);

    const cloudMode = useRecoilValue(atom.cloudModeState);
    const useRandomSeed = useRecoilValue(randomSeedState);
    const [queue, setQueue] = useRecoilState(atom.queueState);
    const setAddToQueue = useSetRecoilState(addToQueueState);
    const imageSettings = useRecoilValue(queueSettingsSelector);
    const setCloudRunning = useSetRecoilState(atom.cloudRunningState);
    const setShard = useSetRecoilState(atom.shardState);
    const setSeed = useSetRecoilState(seedState);

    const useKeyPress = (targetKey: string, useAltKey = false) => {
        const [keyPressed, setKeyPressed] = useState(false);

        useEffect(() => {
            const handler = (isKeyDown: boolean) => ({ key, altKey } : { key: string, altKey: boolean }) => {
                if (key === targetKey && altKey === useAltKey) {
                    console.log(`key: ${key}, altKey: ${altKey}, keydown: ${isKeyDown}`);
                    setKeyPressed(isKeyDown);
                }
            };

            const keydownHandler = handler(true);
            const keyupHandler = handler(false);

            window.addEventListener('keydown', keydownHandler);
            window.addEventListener('keyup', keyupHandler);

            return () => {
                window.removeEventListener('keydown', keydownHandler);
                window.removeEventListener('keyup', keyupHandler);
            };
        }, [targetKey, useAltKey]);

        return keyPressed;
    };

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
            {...imageSettings, id: `${Math.floor(Math.random() * Number.MAX_SAFE_INTEGER)}`},
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

    const computeShardCost = useCallback(() => {
        //estimated_price = (width * height) / (512 * 512) * (steps / 50) * num_images * 10
        let estimated_price = Math.round((imageSettings.width * imageSettings.height) / (512 * 512) * (imageSettings.steps / 50) * imageSettings.n_iter * 10);
        return estimated_price;
    }, [imageSettings]);

    
    useEffect(() => {
        if (altRPressed) {
            addToQueue();
        }
    }, [addToQueue, altRPressed]);

    if(cloudMode) {
        return (
            <Button
                className="run-button"
                ml={2}
                onClick={submitCloud}
                variant="outline"
                width="200px">
                <Text pr={2}>Run</Text>

                <Image src={Shards} width="12px" />

                <Text pl={1}>{computeShardCost()}</Text>
            </Button>
        )
    }
    return (
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
    );
}

interface ReducerState {
    selectedIndex: number;
}
  
interface ReducerAction {
    type: "arrowLeft" | "arrowRight" | "select" | "intermediate";
    payload?: number;
}

const Body = () => {
    const ARTROOM_URL = process.env.REACT_APP_ARTROOM_URL;
    const toast = useToast({});

    const [mainImage, setMainImage] = useRecoilState(atom.mainImageState);
    const latestImages = useRecoilValue(atom.latestImageState);

    const [focused, setFocused] = useState(false);
    
    const socket = useContext(SocketContext);

    const reducer = (state: ReducerState, action: ReducerAction): ReducerState => {
        switch (action.type) {
        case 'arrowLeft':
            console.log('Arrow Left');
            return {
                selectedIndex:
              state.selectedIndex !== 0
                  ? state.selectedIndex - 1
                  : latestImages.length - 1
            };
        case 'arrowRight':
            console.log('Arrow Right');
            return {
                selectedIndex:
              state.selectedIndex !== latestImages.length - 1
                  ? state.selectedIndex + 1
                  : 0
            };
        case 'select':
            console.log('Select');
            return { selectedIndex: action.payload ?? -1 };
        case 'intermediate':
            console.log('intermediate');
            return { selectedIndex: -1 };
        default:
            throw new Error();
        }
    };
    
    const [state, dispatch] = useReducer(reducer, { selectedIndex: 0 });

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

    const useKeyPress = (targetKey: string, useAltKey = false) => {
        const [keyPressed, setKeyPressed] = useState(false);

        useEffect(() => {
            const handler = (isKeyDown: boolean) => ({ key, altKey } : { key: string, altKey: boolean }) => {
                if (key === targetKey && altKey === useAltKey) {
                    console.log(`key: ${key}, altKey: ${altKey}, keydown: ${isKeyDown}`);
                    setKeyPressed(isKeyDown);
                }
            };

            const keydownHandler = handler(true);
            const keyupHandler = handler(false);

            window.addEventListener('keydown', keydownHandler);
            window.addEventListener('keyup', keyupHandler);

            return () => {
                window.removeEventListener('keydown', keydownHandler);
                window.removeEventListener('keyup', keyupHandler);
            };
        }, [targetKey, useAltKey]);

        return keyPressed;
    };

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
        <Box
            width="100%" 
            alignContent="center">
            <VStack spacing={3}>
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
                        {latestImages.map((imageData, index) => (<Image
                            h="5vh"
                            key={index}
                            onClick={() => dispatch({ type: 'select', payload: index })}
                            src={imageData?.b64}
                        />))}
                    </SimpleGrid>
                </Box>

                <QueueButtons />

                <Box width="80%">
                    <Prompt setFocused={setFocused} />
                </Box>
            </VStack>
        </Box>
    );
};

export default Body;
