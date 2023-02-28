import React, {useEffect, useState, useReducer, useRef, useContext, useCallback} from 'react';
import { useRecoilState, useRecoilValue, useSetRecoilState } from 'recoil';
import * as atom from '../atoms/atoms';
import { boundingBoxCoordinatesAtom, boundingBoxDimensionsAtom, layerStateAtom, maxHistoryAtom, pastLayerStatesAtom, futureLayerStatesAtom, stageScaleAtom, shouldPreserveMaskedAreaAtom } from './UnifiedCanvas/atoms/canvas.atoms';
import { UnifiedCanvas } from './UnifiedCanvas/UnifiedCanvas';
import {
    Text,
    Box,
    VStack,
    SimpleGrid,
    Image as ChakraImage,
    Button,
    useToast,
    Progress
} from '@chakra-ui/react';
import Prompt from './Prompt';
import Shards from '../images/shards.png';
import _ from 'lodash';
import { v4 } from 'uuid';
import { generateMask, getCanvasBaseLayer, getScaledBoundingBoxDimensions } from './UnifiedCanvas/util';
import { CanvasImage, isCanvasMaskLine } from './UnifiedCanvas/atoms/canvasTypes';
import { SocketContext, SocketOnEvents } from '../socket';
import { queueSettingsSelector, randomSeedState } from '../SettingsManager';
import { addToQueueState } from '../atoms/atoms';

function randomIntFromInterval(min: number, max: number) { // min and max included 
    return Math.floor(Math.random() * (max - min + 1) + min)
}

function parseSettings(settings: QueueType, useRandom: boolean) {
    settings.seed = useRandom ? randomIntFromInterval(1, 4294967295) : settings.seed;

    const sampler_format_mapping = {
        'k_euler': 'euler',
        'k_euler_ancestral': 'euler_a',
        'k_dpm_2': 'dpm',
        'k_dpm_2_ancestral': 'dpm_a',
        'k_lms': 'lms',
        'k_heun': 'heun'
    }
    if (settings.sampler in sampler_format_mapping) {
        settings.sampler = sampler_format_mapping[settings.sampler]
    }

    return settings;
}

function Paint () {
    const toast = useToast({});

    const latestImages = useRecoilValue(atom.latestImageState);

    const [progress, setProgress] = useState(-1);
    const [batchProgress, setBatchProgress] = useState(-1);
    const [focused, setFocused] = useState(false);
    const cloudMode = useRecoilValue(atom.cloudModeState);
    const setQueue = useSetRecoilState(atom.queueState);
    const setAddToQueue = useSetRecoilState(addToQueueState);

    const boundingBoxCoordinates = useRecoilValue(boundingBoxCoordinatesAtom);  
    const boundingBoxDimensions = useRecoilValue(boundingBoxDimensionsAtom);  
    const [layerState, setLayerState] = useRecoilState(layerStateAtom);  
    const maxHistory = useRecoilValue(maxHistoryAtom);  
    const [pastLayerStates, setPastLayerStates] = useRecoilState(pastLayerStatesAtom);  
    const setFutureLayerStates = useSetRecoilState(futureLayerStatesAtom);
    const imageSettings = useRecoilValue(queueSettingsSelector);
    const shouldPreserveMaskedArea = useRecoilValue(shouldPreserveMaskedAreaAtom)
    const stageScale = useRecoilValue(stageScaleAtom);   
    const useRandomSeed = useRecoilValue(randomSeedState);

    const addOutpaintingLayer = (imageDataURL: string, maskDataURL: string, width?: number, height?: number) => {
        // Create a new canvas element
        var canvas = document.createElement('canvas');
        var ctx = canvas.getContext('2d');
    
        // Load the transparent PNG and the mask PNG
        var regular = new Image();
        regular.src = imageDataURL;
        var mask = new Image();
        mask.src = maskDataURL;
    
        return new Promise<string>((resolve) => {
            regular.onload = function() {
                mask.onload = function() {
                    // Draw the image on the canvas
                    canvas.width = width || regular.width;
                    canvas.height = height || regular.height;
                    ctx.drawImage(regular, 0, 0);
                    // Get the image data from the canvas
                    var imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
                    var data = imageData.data;
                    // Iterate through the image data and set the alpha value to the color value
                    for (var i = 0; i < data.length; i += 4) {
                        var alpha = data[i + 3];
                        data[i] = alpha;
                        data[i + 1] = alpha;
                        data[i + 2] = alpha;
                    }
                    // Put the image data back on the canvas
                    ctx.putImageData(imageData, 0, 0);
                    ctx.globalCompositeOperation = "source-in";
                    ctx.drawImage(mask, 0, 0);
                    var dataUrl = canvas.toDataURL();
                    resolve(dataUrl);
                }
            };
        });
    }

    const socket = useContext(SocketContext);

    const handleRunInpainting = useCallback(() => {
        const canvasBaseLayer = getCanvasBaseLayer();
        const boundingBox = {
          ...boundingBoxCoordinates,
          ...boundingBoxDimensions,
        };
        const maskDataURL = generateMask(
          layerState.objects.filter(isCanvasMaskLine),
          boundingBox
        );
        const tempScale = canvasBaseLayer.scale();
      
        canvasBaseLayer.scale({
          x: 1 / stageScale,
          y: 1 / stageScale,
        });
      
        const absPos = canvasBaseLayer.getAbsolutePosition();
      
        const imageDataURL = canvasBaseLayer.toDataURL({
          x: boundingBox.x + absPos.x,
          y: boundingBox.y + absPos.y,
          width: boundingBox.width,
          height: boundingBox.height,
        });
    
        canvasBaseLayer.scale(tempScale);
        addOutpaintingLayer(imageDataURL, maskDataURL, boundingBox.width, boundingBox.height).then(combinedMask => {
            const body = {
                ...imageSettings,
                init_image: imageDataURL,
                mask_image: combinedMask,
                invert: shouldPreserveMaskedArea
            }

            setAddToQueue(true);
            setQueue((queue) => {
                return [
                    ...queue,
                    parseSettings(
                        {...body, id: `${Math.floor(Math.random() * Number.MAX_SAFE_INTEGER)}`},
                        useRandomSeed
                    )
                ];
            });
        }).catch(err =>{
           console.log(err);
        })        

    }, [boundingBoxCoordinates, boundingBoxDimensions, layerState.objects, stageScale, imageSettings, socket, shouldPreserveMaskedArea]);

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

    // on socket message
    useEffect(() => {
        socket.on('get_progress', handleGetProgress);
        socket.on('get_status', handleGetStatus);
    
        return () => {
            socket.off('get_progress', handleGetProgress);
            socket.off('get_status', handleGetStatus);
        };
    }, [socket, handleGetProgress, handleGetStatus]);

    const computeShardCost = () => {
        //estimated_price = (width * height) / (512 * 512) * (steps / 50) * num_images * 10
        let estimated_price = Math.round((imageSettings.width * imageSettings.height) / (512 * 512) * (imageSettings.steps / 50) * imageSettings.n_iter * 10);
        return estimated_price;
    }

    function addToCanvas(imageData: { b64?: string, path?: string; batch_id?: number }){
        const scaledDimensions = getScaledBoundingBoxDimensions(
            boundingBoxDimensions
        );         
        
        const boundingBox = {
            ...boundingBoxCoordinates,
            ...scaledDimensions,
        };

        const image = {
            ...scaledDimensions,
            category: "user",
            mtime: 1673399421.3987432,
            url: imageData.path,
            uuid: v4(),
            kind: "image",
            layer: "base",
            x: 0,
            y: 0
        }

        if (!boundingBox || !image) return;

        setPastLayerStates([...pastLayerStates, _.cloneDeep(layerState)]);

        if (pastLayerStates.length > maxHistory) {
            setPastLayerStates(pastLayerStates.slice(1));
        }


        //Filters so that the same image isn't used in the same spot, saving memory
        setLayerState({
            ...layerState,
            objects: [
                ...layerState.objects.filter(
                    (object: CanvasImage) => !(object.image?.url === imageData.path && object.x === boundingBox.x && object.y === boundingBox.y)),                    
                {
                    kind: 'image',
                    layer: 'base',
                    ...boundingBox,
                    image,
                },
            ],
        });
        setFutureLayerStates([]);
    }

    const mainImageIndex = { selectedIndex: 0 };
    const reducer = (state: { selectedIndex: number; }, action: { type: any; payload: any; }) => {
        switch (action.type) {
        case 'arrowLeft':
            return {
                selectedIndex:
              state.selectedIndex !== 0
                  ? state.selectedIndex - 1
                  : latestImages.length - 1
            };
        case 'arrowRight':
            return {
                selectedIndex:
              state.selectedIndex !== latestImages.length - 1
                  ? state.selectedIndex + 1
                  : 0
            };
        case 'select':
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

    const prevSelectedIndex = useRef(0);
    useEffect(() => {
        if (latestImages.length > 0 && prevSelectedIndex.current !== state.selectedIndex){
            addToCanvas(latestImages[state?.selectedIndex]);
            prevSelectedIndex.current = state.selectedIndex;
        }
        },[state?.selectedIndex]
    );

    return (
        <Box
            width="100%">
            <VStack
                align="center"
                spacing={4}>
                <Box
                    className="paint-output">
                    <UnifiedCanvas />
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
                        {latestImages.map((imageData, index) => (<ChakraImage
                            h="5vh"
                            key={index}
                            src={imageData.b64}
                            onClick={() => {
                                addToCanvas(imageData)
                            }}
                        />))}
                    </SimpleGrid>
                </Box>
                {cloudMode
                    ? <Button
                        className="run-button"
                        ml={2}
                        onClick={handleRunInpainting}
                        variant="outline"
                        width="200px">
                        <Text pr={2}>
                            Run
                        </Text>

                        <ChakraImage
                            src={Shards}
                            width="12px" />

                        <Text pl={1}>
                            {computeShardCost()}
                        </Text>
                    </Button>
                    : <Button
                        className="run-button"
                        ml={2}
                        onClick={handleRunInpainting}
                        width="200px"> 
                        Run
                    </Button>}
                <Box width="80%">
                    <Prompt setFocused={setFocused} />
                </Box>
            </VStack>
        </Box>
    );
}
export default Paint;
