import React, { useState, useEffect, useReducer } from 'react';
import { useInterval } from './Reusable/useInterval/useInterval';
import { useRecoilState } from 'recoil';
import * as atom from '../atoms/atoms';
import axios from 'axios';
import {
    Box,
    Button,
    Flex,
    VStack,
    Progress,
    SimpleGrid,
    Image,
    Text,
    createStandaloneToast
} from '@chakra-ui/react';
import ImageObj from './Reusable/ImageObj';
import Prompt from './Prompt';
import Shards from '../images/shards.png';
import ProtectedReqManager from '../helpers/ProtectedReqManager';

function Body () {
    const LOCAL_URL = process.env.REACT_APP_LOCAL_URL;
    const ARTROOM_URL = process.env.REACT_APP_SERVER_URL;
    const baseURL = LOCAL_URL;

    const { ToastContainer, toast } = createStandaloneToast();

    const [imageSettings, setImageSettings] = useRecoilState(atom.imageSettingsState)

    const [mainImage, setMainImage] = useRecoilState(atom.mainImageState);
    const [latestImages, setLatestImages] = useRecoilState(atom.latestImageState);
    const [latestImagesID, setLatestImagesID] = useRecoilState(atom.latestImagesIDState);

    const [progress, setProgress] = useState(-1);
    const [running, setRunning] = useState(false);
    const [focused, setFocused] = useState(false);

    const [cloudMode, setCloudMode] = useRecoilState(atom.cloudModeState);

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

    const [state, dispatch] = useReducer(
        reducer,
        mainImageIndex
    );

    const useKeyPress = (targetKey: string, useAltKey = false) => {
        const [keyPressed, setKeyPressed] = useState(false);

        useEffect(
            () => {
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

                window.addEventListener(
                    'keydown',
                    leftHandler
                );
                window.addEventListener(
                    'keyup',
                    rightHandler
                );

                return () => {
                    window.removeEventListener(
                        'keydown',
                        leftHandler
                    );
                    window.removeEventListener(
                        'keyup',
                        rightHandler
                    );
                };
            },
            [targetKey]
        );

        return keyPressed;
    };

    const arrowRightPressed = useKeyPress('ArrowRight');
    const arrowLeftPressed = useKeyPress('ArrowLeft');
    const altRPressed = useKeyPress(
        'r',
        true
    );

    useEffect(
        () => {
            if (arrowRightPressed && !focused) {
                dispatch({
                    type: 'arrowRight',
                    payload: undefined
                });
            }
        },
        [arrowRightPressed]
    );

    useEffect(
        () => {
            if (arrowLeftPressed && !focused) {
                dispatch({
                    type: 'arrowLeft',
                    payload: undefined
                });
            }
        },
        [arrowLeftPressed]
    );

    useEffect(
        () => {
            if (altRPressed) {
                submitMain();
            }
        },
        [altRPressed]
    );

    useEffect(
        () => {
            setMainImage(latestImages[state.selectedIndex]);
        },
        [state]
    );

    useEffect(
        () => {
            const interval = setInterval(
                () => axios.get(
                    `${baseURL}/get_progress`,
                    { headers: { 'Content-Type': 'application/json' } }
                ).then((result) => {
                    if (result.data.status === 'Success') {
                        setProgress(result.data.content.percentage);
                        setRunning(result.data.content.running);
                        if (result.data.content.status === 'Loading Model' && !toast.isActive('loading-model')) {
                            toast({
                                id: 'loading-model',
                                title: 'Loading model...',
                                status: 'info',
                                position: 'bottom-right',
                                duration: 30000,
                                isClosable: false
                            });
                        }
                        if (!(result.data.content.status === 'Loading Model')) {
                            if (toast.isActive('loading-model')) {
                                toast.close('loading-model');
                            }
                        }
                    } else {
                        setProgress(-1);
                        setRunning(false);
                        if (toast.isActive('loading-model')) {
                            toast.close('loading-model');
                        }
                    }
                }),
                500
            );
            return () => {
                clearInterval(interval);
            };
        },
        []
    );


    useInterval(
        () => {
            axios.get(
                `${baseURL}/get_images`,
                { params: { 'path': 'latest',
                    'id': latestImagesID },
                headers: { 'Content-Type': 'application/json' } }
            ).then((result) => {
                const id = result.data.content.latest_images_id;
                if (result.data.status === 'Success') {
                    if (id !== latestImagesID) {
                        setLatestImagesID(id);
                        setLatestImages(result.data.content.latest_images);
                        setMainImage(result.data.content.latest_images[result.data.content.latest_images.length - 1]);
                    }
                } else if (result.data.status === 'Failure') {
                    setMainImage('');
                }
            });
        },
        3000
    );

    const submitMain = () => {
        axios.post(
            `${baseURL}/add_to_queue`, imageSettings,
            {
                headers: { 'Content-Type': 'application/json' }
            }
        ).then((result) => {
            if (result.data.status === 'Success') {
                toast({
                    title: 'Added to Queue!',
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
                    description: result.data.status_message,
                    position: 'top',
                    duration: 5000,
                    isClosable: true,
                    containerStyle: {
                        pointerEvents: 'none'
                    }
                });
            }
        }).
            catch((error) => console.log(error));
    };

    const getProfile = () => {
        ProtectedReqManager.make_request(`${ARTROOM_URL}/users/me`).then((response: { data: { email: string; }; }) => {
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
    }


    return (
        <Box
            alignContent="center"
            width="100%" >
            {/* Center Portion */}

            <VStack spacing={3}>
                <Box
                    className="image-box"
                    width="80%">
                    <ImageObj
                        b64={mainImage}
                        active />

                    {
                        progress >= 0
                            ? <Progress
                                alignContent="left"
                                hasStripe
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
                        {latestImages.map((image, index) => (<Image
                            h="5vh"
                            key={index}
                            onClick={() => dispatch({ type: 'select',
                                payload: index })}
                            src={image}
                        />))}
                    </SimpleGrid>
                </Box>

                {cloudMode
                    ? <Button
                        className="run-button"
                        ml={2}
                        onClick={getProfile}
                        variant="outline"
                        width="200px">
                        <Text pr={2}>
                            {running
                                ? 'Add to Queue'
                                : 'Run'}
                        </Text>

                        <Image
                            src={Shards}
                            width="12px" />

                        <Text pl={1}>
                            6
                        </Text>
                    </Button>
                    : <Button
                        className="run-button"
                        ml={2}
                        onClick={submitMain}
                        width="200px">
                        {running
                            ? 'Add to Queue'
                            : 'Run'}
                    </Button>}

                <Box width="80%">
                    <Prompt setFocused={setFocused} />
                </Box>
            </VStack>
        </Box>
    );
};

export default Body;
