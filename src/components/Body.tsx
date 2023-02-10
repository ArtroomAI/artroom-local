import React, { useState, useEffect, useReducer, useContext, useCallback } from 'react';
import { useRecoilState, useRecoilValue, useSetRecoilState } from 'recoil';
import * as atom from '../atoms/atoms';
import {
    Box,
    Button,
    VStack,
    Progress,
    SimpleGrid,
    Image,
    Text,
    useToast
} from '@chakra-ui/react';
import ImageObj from './Reusable/ImageObj';
import Prompt from './Prompt';
import Shards from '../images/shards.png';
import ProtectedReqManager from '../helpers/ProtectedReqManager';
import { SocketContext, SocketOnEvents } from '../socket';

const Body = () => {
    const ARTROOM_URL = process.env.REACT_APP_ARTROOM_URL;

    const toast = useToast({});

    const imageSettings = useRecoilValue(atom.imageSettingsState)

    const [mainImage, setMainImage] = useRecoilState(atom.mainImageState);
    const latestImages = useRecoilValue(atom.latestImageState);
    const setCloudRunning = useSetRecoilState(atom.cloudRunningState);
    
    const [progress, setProgress] = useState(-1);
    const [batchProgress, setBatchProgress] = useState(-1);

    const [focused, setFocused] = useState(false);

    const cloudMode = useRecoilValue(atom.cloudModeState);
    const setShard = useSetRecoilState(atom.shardState);
    
    const socket = useContext(SocketContext);

    const addToQueue = useCallback(() => {
        socket.emit('add_to_queue', imageSettings);
    }, [socket, imageSettings]);

    const handleGetProgress: SocketOnEvents['get_progress'] = useCallback((data) => {
        setProgress((100 * data.current_step / data.total_steps));
        setBatchProgress(100 * (data.current_num * data.total_steps + data.current_step) / (data.total_steps * data.total_num));
    }, []);

    const handleGetStatus: SocketOnEvents['get_status'] = useCallback((data) => {
        if (data.status === 'Loading Model') {
            toast({
                id: 'loading-model',
                title: 'Loading model...',
                status: 'info',
                position: 'bottom-right',
                duration: null,
                isClosable: false
            });
        } else if (data.status === 'Finished Loading Model') {
            if (toast.isActive('loading-model')) {
                toast.close('loading-model');
            }
        }
    }, [toast]);

    const handleAddToQueue: SocketOnEvents['add_to_queue'] = useCallback((data) => {
        if (data.status === 'Success') {
            toast({
                title: 'Added to Queue!',
                description: `Currently ${data.queue_size} elements in queue`,
                status: 'success',
                position: 'top',
                duration: 2000,
                isClosable: false,
                containerStyle: {
                    pointerEvents: 'none'
                }
            });
        } else {
            toast({
                title: 'Error',
                status: 'error',
                description: data.status_message,
                position: 'top',
                duration: 5000,
                isClosable: true,
                containerStyle: {
                    pointerEvents: 'none'
                }
            });
        }
    }, [toast]);

    // on socket message
    useEffect(() => {
        socket.on('add_to_queue', handleAddToQueue);
        socket.on('get_progress', handleGetProgress);
        socket.on('get_status', handleGetStatus);
    
        return () => {
            socket.off('get_progress', handleGetProgress);
            socket.off('add_to_queue', handleAddToQueue);
            socket.off('get_status', handleGetStatus);
        };
    }, [socket, handleAddToQueue, handleGetProgress, handleGetStatus]);
    
    const mainImageIndex = { selectedIndex: 0 };
    const reducer = (state: { selectedIndex: number; }, action: { type: any; payload: any; }) => {
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
            return { selectedIndex: action.payload };
        default:
            throw new Error();
        }
    };

    const [state, dispatch] = useReducer(reducer, mainImageIndex);

    const computeShardCost = () => {
        //estimated_price = (width * height) / (512 * 512) * (steps / 50) * num_images * 10
        let estimated_price = Math.round((imageSettings.width * imageSettings.height) / (512 * 512) * (imageSettings.steps / 50) * imageSettings.n_iter * 10);
        return estimated_price;
    }

    const useKeyPress = (targetKey: string, useAltKey = false) => {
        const [keyPressed, setKeyPressed] = useState(false);

        useEffect(() => {
            const leftHandler = ({ key, altKey } : { key: string, altKey: boolean}) => {
                if (key === targetKey && altKey === useAltKey) {
                    console.log(key);
                    console.log(altKey);
                    setKeyPressed(true);
                }
            };

            const rightHandler = ({ key, altKey } : { key: string, altKey: boolean}) => {
                if (key === targetKey && altKey === useAltKey) {
                    setKeyPressed(false);
                }
            };

            window.addEventListener('keydown', leftHandler);
            window.addEventListener('keyup', rightHandler);

            return () => {
                window.removeEventListener('keydown', leftHandler);
                window.removeEventListener('keyup', rightHandler);
            };
        }, [targetKey, useAltKey]);

        return keyPressed;
    };

    const arrowRightPressed = useKeyPress('ArrowRight');
    const arrowLeftPressed = useKeyPress('ArrowLeft');
    const altRPressed = useKeyPress('r', true);

    useEffect(() => {
        if (arrowRightPressed && !focused) {
            dispatch({
                type: 'arrowRight',
                payload: undefined
            });
        }
    }, [arrowRightPressed, focused]);

    useEffect(() => {
        if (arrowLeftPressed && !focused) {
            dispatch({
                type: 'arrowLeft',
                payload: undefined
            });
        }
    }, [arrowLeftPressed, focused]);

    useEffect(() => {
        if (altRPressed) {
            addToQueue();
        }
    }, [addToQueue, altRPressed]);

    useEffect(() => {
        setMainImage(latestImages[state.selectedIndex]);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [state.selectedIndex]);

    const submitCloud = () => {
        ProtectedReqManager.make_post_request(`${ARTROOM_URL}/gpu/submit_job_to_queue`, imageSettings).then((response: any) => {
            setShard(response.data.shard_balance);
            toast({
                title: 'Job Submission Success',
                status: 'success',
                position: 'top',
                duration: 5000,
                isClosable: true,
                containerStyle: {
                    pointerEvents: 'none'
                }
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
                isClosable: true,
                containerStyle: {
                    pointerEvents: 'none'
                }
            });
        });
    };


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
                    {
                        (batchProgress >= 0 && batchProgress !== 100)
                            ? <Progress
                                alignContent="left"
                                hasStripe
                                width="100%"
                                value={batchProgress} />
                            : <></>
                    }
                    {
                        (progress >= 0 && progress !== 100)
                            ? <Progress
                                alignContent="left"
                                hasStripe
                                width="100%"
                                value={progress} />
                            : <></>
                    }
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
                            onClick={() => dispatch({ type: 'select',
                                payload: index })}
                            src={imageData?.b64}
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

                        <Image
                            src={Shards}
                            width="12px" />

                        <Text pl={1}>
                            {computeShardCost()}
                        </Text>
                    </Button>
                    : <Button
                        className="run-button"
                        ml={2}
                        onClick={addToQueue}
                        width="200px"> 
                        Run
                    </Button>}

                <Box width="80%">
                    <Prompt setFocused={setFocused} />
                </Box>
            </VStack>
        </Box>
    );
};

export default Body;
