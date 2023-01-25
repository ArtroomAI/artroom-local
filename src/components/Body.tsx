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
    useToast
} from '@chakra-ui/react';
import ImageObj from './Reusable/ImageObj';
import Prompt from './Prompt';
import Shards from '../images/shards.png';
import ProtectedReqManager from '../helpers/ProtectedReqManager';
import path from 'path';

function Body () {
    const LOCAL_URL = process.env.REACT_APP_LOCAL_URL;
    const ARTROOM_URL = process.env.REACT_APP_ARTROOM_URL;
    const baseURL = LOCAL_URL;

    const toast = useToast({});

    const [imageSettings, setImageSettings] = useRecoilState(atom.imageSettingsState)

    const [mainImage, setMainImage] = useRecoilState(atom.mainImageState);
    const [latestImages, setLatestImages] = useRecoilState(atom.latestImageState);
    const [latestImagesID, setLatestImagesID] = useRecoilState(atom.latestImagesIDState);
    const [cloudRunning, setCloudRunning] = useRecoilState(atom.cloudRunningState);
    

    const [progress, setProgress] = useState(-1);

    const [focused, setFocused] = useState(false);

    const [cloudMode, setCloudMode] = useRecoilState(atom.cloudModeState);
    const [shard, setShard] = useRecoilState(atom.shardState);
    
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

    const computeShardCost = () => {
        //estimated_price = (width * height) / (512 * 512) * (steps / 50) * num_images * 10
        let estimated_price = Math.round((imageSettings.width * imageSettings.height) / (512 * 512) * (imageSettings.steps / 50) * imageSettings.n_iter * 10);
        return estimated_price;
    }

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
            setMainImage(latestImages[state.selectedIndex]?.b64);
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
                        setMainImage(result.data.content.latest_images[result.data.content.latest_images.length - 1]?.b64);
                    }
                } else if (result.data.status === 'Failure') {
                    setMainImage('');
                }
            });
        },
        3000
    );


    useInterval(
        () => {
            ProtectedReqManager.make_get_request(`${ARTROOM_URL}/image/get_status`).then((response: any) => {
                if (response.data.jobs.length == 0) {
                    toast({
                        title: 'Cloud jobs Complete!',
                        status: 'success',
                        position: 'top',
                        duration: 5000,
                        isClosable: true,
                        containerStyle: {
                            pointerEvents: 'none'
                        }
                    });
                    setCloudRunning(false);
                } else {
                    let job_list = response.data.jobs;
                    let text = "";
                    let pending_cnt = 0;
                    let newCloudImages = [];
                    for (let i = 0; i < job_list.length; i++) {
                        for (let j = 0; j < job_list[i].images.length; j++) {
                            if (job_list[i].images[j].status == 'PENDING') {
                                pending_cnt = pending_cnt + 1;
                            } else if (job_list[i].images[j].status == 'SUCCESS') {
                                //text = text + "job_" + job_list[i].id.slice(0, 5) + 'img_' + job_list[i].images[j].id + '\n';
                                let img_name = job_list[i].id + '_' + job_list[i].images[j].id;
                                const imagePath = path.join(imageSettings.image_save_path, imageSettings.batch_name, img_name + "_cloud.jpg");
                                toast({
                                    title: "Image completed: " + imagePath,
                                    status: 'info',
                                    position: 'top',
                                    duration: 5000,
                                    isClosable: true,
                                    containerStyle: {
                                        pointerEvents: 'none'
                                    }
                                });
                                //const timestamp = new Date().getTime();
                                console.log(imagePath);
                                let dataURL = job_list[i].images[j].url;
                                newCloudImages.push({"b64": dataURL})
                                window.api.saveFromDataURL(JSON.stringify({dataURL, imagePath}));
                            }
                        }
                    }
                    setLatestImages([...latestImages, ...newCloudImages])
                    toast({
                        title: 'Cloud jobs running!\n',
                        description: text + pending_cnt + " jobs pending",
                        status: 'info',
                        position: 'top',
                        duration: 5000,
                        isClosable: true,
                        containerStyle: {
                            pointerEvents: 'none'
                        }
                    });
                }

            }).catch((err: any) => {
                console.log(err);
            });
        },
        cloudRunning ? 5000 : null
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
                    >
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
                        onClick={submitMain}
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
