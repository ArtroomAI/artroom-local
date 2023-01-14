import React, {useEffect, useState, useReducer, useRef} from 'react';
import { useRecoilState, useRecoilValue } from 'recoil';
import * as atom from '../atoms/atoms';
import { boundingBoxCoordinatesAtom, boundingBoxDimensionsAtom, layerStateAtom, maxHistoryAtom, pastLayerStatesAtom, futureLayerStatesAtom, setIsMaskEnabledAction, stageScaleAtom, stageDimensionsAtom } from './UnifiedCanvas/atoms/canvas.atoms';
import axios from 'axios';
import { UnifiedCanvas } from './UnifiedCanvas/UnifiedCanvas';
import {
    Text,
    Box,
    VStack,
    SimpleGrid,
    Image as ChakraImage,
    Button,
    useToast
} from '@chakra-ui/react';
import Prompt from './Prompt';
import { useInterval } from './Reusable/useInterval/useInterval';
import Shards from '../images/shards.png';
import _ from 'lodash';
import { v4 as uuidv4 } from 'uuid';
import { generateMask, getCanvasBaseLayer, getScaledBoundingBoxDimensions } from './UnifiedCanvas/util';
import { isCanvasMaskLine } from './UnifiedCanvas/atoms/canvasTypes';

const loadImage = async (b64: string) => {
    const image = new Image();
    image.src = b64;
    return new Promise((resolve) => {
      image.onload = () => {
        resolve({width: image.width, height: image.height});
      }
    });
  }


window.addEventListener('keydown', function(e) {
    if (e.code === 'Space') {
    e.preventDefault();
    }
});
  
function Paint () {
    const LOCAL_URL = process.env.REACT_APP_LOCAL_URL;
    const ARTROOM_URL = process.env.REACT_APP_ARTROOM_URL;
    const baseURL = LOCAL_URL;

    const toast = useToast({});

    const [mainImage, setMainImage] = useRecoilState(atom.mainImageState);
    const [latestImages, setLatestImages] = useRecoilState(atom.latestImageState);
    const [latestImagesID, setLatestImagesID] = useRecoilState(atom.latestImagesIDState);

    const [progress, setProgress] = useState(-1);
    const [focused, setFocused] = useState(false);
    const [cloudMode, setCloudMode] = useRecoilState(atom.cloudModeState);

    const boundingBoxCoordinates = useRecoilValue(boundingBoxCoordinatesAtom);  
    const boundingBoxDimensions = useRecoilValue(boundingBoxDimensionsAtom);  
    const [layerState, setLayerState] = useRecoilState(layerStateAtom);  
    const [maxHistory, setMaxHistory] = useRecoilState(maxHistoryAtom);  
    const [pastLayerStates, setPastLayerStates] = useRecoilState(pastLayerStatesAtom);  
    const [futureLayerStates, setFutureLayerStates] = useRecoilState(futureLayerStatesAtom);  
    const [imageSettings, setImageSettings] = useRecoilState(atom.imageSettingsState)
    const [isMaskEnabled, setIsMaskEnabled] = useRecoilState(setIsMaskEnabledAction);
    const stageScale = useRecoilValue(stageScaleAtom);   
    const stageDimensions = useRecoilValue(stageDimensionsAtom);

    const handleRunInpainting = () => {
        const canvasBaseLayer = getCanvasBaseLayer();
        const boundingBox = {
          ...boundingBoxCoordinates,
          ...boundingBoxDimensions,
        };
        const maskDataURL = generateMask(
          isMaskEnabled ? layerState.objects.filter(isCanvasMaskLine) : [],
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
        const body = {
          ...imageSettings,
          init_image: imageDataURL,
          mask_image: maskDataURL
        };
        axios.post(
          `${baseURL}/add_to_queue`,
          body,
          {
            headers: { 'Content-Type': 'application/json' }
          }
          ).then(result =>{
            console.log(result);
          })
      };

    // useEffect(() =>{
    //     console.log({...boundingBoxCoordinates, ...boundingBoxDimensions});
    // },[boundingBoxCoordinates, boundingBoxDimensions])

    const computeShardCost = () => {
        //estimated_price = (width * height) / (512 * 512) * (steps / 50) * num_images * 10
        let estimated_price = Math.round((imageSettings.width * imageSettings.height) / (512 * 512) * (imageSettings.steps / 50) * imageSettings.n_iter * 10);
        return estimated_price;
    }

    function addToCanvas(imageData: { b64: string, path: string; }){

    
        const scaledDimensions = getScaledBoundingBoxDimensions(
            boundingBoxDimensions
        );         
        
        const boundingBox = {
            ...boundingBoxCoordinates,
            ...scaledDimensions,
        };
        console.log(boundingBox)

        const image = {
        ...scaledDimensions,
        category: "user",
        mtime: 1673399421.3987432,
        url: imageData.path,
        uuid: uuidv4(),
        kind: "image",
        layer: "base",
        x: 0,
        y: 0
    }

        if (!boundingBox || !image) return;

        setPastLayerStates([
        ...pastLayerStates,
        _.cloneDeep(layerState),
        ]);

        if (pastLayerStates.length > maxHistory) {
        setPastLayerStates(pastLayerStates.slice(1));
        }
        //Filters so that the same image isn't used in the same spot, saving memory
        setLayerState({
        ...layerState,
        objects: [
            ...layerState.objects.filter(
                object => !(object.image?.url === imageData.path && object.x === boundingBox.x && object.y === boundingBox.y)
                ),                    
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
                        setMainImage(result.data.content.latest_images[result.data.content.latest_images.length - 1]);
                    }
                } else if (result.data.status === 'Failure') {
                    setMainImage('');
                }
            });
        },
        3000
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
                    <UnifiedCanvas></UnifiedCanvas>
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
