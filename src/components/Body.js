import { React, useState, useEffect, useCallback, useReducer } from 'react';
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
import ImageObj from './ImageObj';
import Prompt from './Prompt';
import Shards from '../images/shards.png';

function Body () {
    const LOCAL_URL = process.env.REACT_APP_LOCAL_URL;
    const ARTROOM_URL = process.env.REACT_APP_ARTROOM_URL;

    const baseURL = LOCAL_URL;

    const { ToastContainer, toast } = createStandaloneToast();
    const [navSize, changeNavSize] = useRecoilState(atom.navSizeState);

    const [width, setWidth] = useRecoilState(atom.widthState);
    const [height, setHeight] = useRecoilState(atom.heightState);
    const [text_prompts, setTextPrompts] = useRecoilState(atom.textPromptsState);
    const [negative_prompts, setNegativePrompts] = useRecoilState(atom.negativePromptsState);
    const [batch_name, setBatchName] = useRecoilState(atom.batchNameState);
    const [steps, setSteps] = useRecoilState(atom.stepsState);
    const [aspect_ratio, setAspectRatio] = useRecoilState(atom.aspectRatioState);
    const [seed, setSeed] = useRecoilState(atom.seedState);
    const [use_random_seed, setUseRandomSeed] = useRecoilState(atom.useRandomSeedState);
    const [n_iter, setNIter] = useRecoilState(atom.nIterState);
    const [sampler, setSampler] = useRecoilState(atom.samplerState);
    const [cfg_scale, setCFGScale] = useRecoilState(atom.CFGScaleState);
    const [init_image, setInitImage] = useRecoilState(atom.initImageState);
    const [ckpt, setCkpt] = useRecoilState(atom.ckptState);
    const [image_save_path, setImageSavePath] = useRecoilState(atom.imageSavePathState);
    const [long_save_path, setLongSavePath] = useRecoilState(atom.longSavePathState);
    const [highres_fix, setHighresFix] = useRecoilState(atom.highresFixState);
    const [speed, setSpeed] = useRecoilState(atom.speedState);
    const [use_full_precision, setUseFullPrecision] = useRecoilState(atom.useFullPrecisionState);
    const [use_cpu, setUseCPU] = useRecoilState(atom.useCPUState);
    const [save_grid, setSaveGrid] = useRecoilState(atom.saveGridState);
    const [debug_mode, setDebugMode] = useRecoilState(atom.debugMode);
    const [ckpt_dir, setCkptDir] = useRecoilState(atom.ckptDirState);
    const [strength, setStrength] = useRecoilState(atom.strengthState);
    const [delay, setDelay] = useRecoilState(atom.delayState);

    const [mainImage, setMainImage] = useRecoilState(atom.mainImageState);
    const [latestImages, setLatestImages] = useRecoilState(atom.latestImageState);
    const [latestImagesID, setLatestImagesID] = useRecoilState(atom.latestImagesIDState);

    const [progress, setProgress] = useState(-1);
    const [stage, setStage] = useState('');
    const [running, setRunning] = useState(false);
    const [focused, setFocused] = useState(false);

    const [cloudMode, setCloudMode] = useRecoilState(atom.cloudModeState);

    const mainImageIndex = { selectedIndex: 0 };
    const reducer = (state, action) => {
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

    const useKeyPress = (targetKey, useAltKey = false) => {
        const [keyPressed, setKeyPressed] = useState(false);

        useEffect(
            () => {
                const leftHandler = ({ key, altKey }) => {
                    if (key === targetKey && altKey === useAltKey) {
                        console.log(key);
                        console.log(altKey);
                        setKeyPressed(true);
                    }
                };

                const rightHandler = ({ key, altKey }) => {
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
                dispatch({ type: 'arrowRight' });
            }
        },
        [arrowRightPressed]
    );

    useEffect(
        () => {
            if (arrowLeftPressed && !focused) {
                dispatch({ type: 'arrowLeft' });
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
                        setStage(result.data.content.stage);
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
                        setStage('');
                        setRunning(false);
                        if (toast.isActive('loading-model')) {
                            toast.close('loading-model');
                        }
                    }
                }),
                1500
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

                /*
                 * Console.log(id);
                 * console.log(latestImagesID);
                 */
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

    const submitMain = (event) => {
        axios.post(
            `${baseURL}/add_to_queue`,
            {
                text_prompts,
                negative_prompts,
                batch_name,
                steps,
                aspect_ratio,
                width,
                height,
                seed,
                use_random_seed,
                n_iter,
                cfg_scale,
                sampler,
                init_image,
                strength,
                reverse_mask: false,
                ckpt,
                image_save_path,
                long_save_path,
                highres_fix,
                speed,
                use_full_precision,
                use_cpu,
                save_grid,
                debug_mode,
                ckpt_dir,
                mask: '',
                delay
            },
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

    return (
        <Flex
            ml={navSize === 'large'
                ? '180px'
                : '100px'}
            transition="all .25s ease"
            width="100%">
            <Box
                align="center"
                width="100%" >
                {/* Center Portion */}

                <VStack spacing={3}>
                    <Box
                        className="image-box"
                        ratio={16 / 9}
                        width="80%">
                        <ImageObj
                            B64={mainImage}
                            active />

                        {
                            progress >= 0
                                ? <Progress
                                    align="left"
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
                            {latestImages?.map((image, index) => (<Image
                                fit="scale-left"
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
                            onClick={submitMain}
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
        </Flex>
    );
}
export default Body;
