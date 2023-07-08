import React, {useEffect, useRef, useContext, useCallback, useState} from 'react';
import { useRecoilState, useRecoilValue, useSetRecoilState } from 'recoil';
import * as atom from '../atoms/atoms';
import { boundingBoxCoordinatesAtom, boundingBoxDimensionsAtom, layerStateAtom, maxHistoryAtom, pastLayerStatesAtom, futureLayerStatesAtom, stageScaleAtom, shouldPreserveMaskedAreaAtom } from './UnifiedCanvas/atoms/canvas.atoms';
import { UnifiedCanvas } from './UnifiedCanvas/UnifiedCanvas';
import {
    Box,
    SimpleGrid,
    useToast,
    Image as ChakraImage,
    Button,
    HStack,
    IconButton,
    VStack,
    Text
} from '@chakra-ui/react';
import _ from 'lodash';
import { v4 } from 'uuid';
import { generateMask, getCanvasBaseLayer, getScaledBoundingBoxDimensions } from './UnifiedCanvas/util';
import { CanvasImage, isCanvasMaskLine } from './UnifiedCanvas/atoms/canvasTypes';
import { SocketContext } from '../socket';
import { queueSettingsSelector, randomSeedState, seedState } from '../SettingsManager';
import { addToQueueState, cloudModeState } from '../atoms/atoms';
import { parseSettings } from './Utils/utils';
import { GenerationProgressBars } from './Reusable/GenerationProgressBars';
import { latestImagesDispatcher, useKeyPress } from './latestImagesDispatcher';
import { loadImage } from './Utils/image';
import { ImageState } from '../atoms/atoms.types';
import { FaStop } from 'react-icons/fa';
import { computeShardCost } from './Utils/shardCost';
import Prompt from './Prompt';
import Shards from '../images/shards.png';

const addOutpaintingLayer = async (imageDataURL: string, maskDataURL: string, width?: number, height?: number) => {
    // Create a new canvas element
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d')!;

    // Load the transparent PNG and the mask PNG
    const regular = await loadImage(imageDataURL);
    const mask = await loadImage(maskDataURL);

    canvas.width = width || regular.width;
    canvas.height = height || regular.height;

    ctx.drawImage(regular, 0, 0);

    // Get the image data from the canvas
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageData.data;

    // Iterate through the image data and set the alpha value to the color value
    for (let i = 0; i < data.length; i += 4) {
        const alpha = data[i + 3];
        data[i] = alpha;
        data[i + 1] = alpha;
        data[i + 2] = alpha;
    }

    // Put the image data back on the canvas
    ctx.putImageData(imageData, 0, 0);
    ctx.globalCompositeOperation = "source-in";
    ctx.drawImage(mask, 0, 0);
    return canvas.toDataURL();
}

const PaintLatestImages = ({ focused } : { focused: boolean }) => {
    const boundingBoxCoordinates = useRecoilValue(boundingBoxCoordinatesAtom);  
    const boundingBoxDimensions = useRecoilValue(boundingBoxDimensionsAtom);
    const [layerState, setLayerState] = useRecoilState(layerStateAtom);
    const latestImages = useRecoilValue(atom.latestImageState);
    const maxHistory = useRecoilValue(maxHistoryAtom);  
    const [pastLayerStates, setPastLayerStates] = useRecoilState(pastLayerStatesAtom);  
    const setFutureLayerStates = useSetRecoilState(futureLayerStatesAtom);

    const addToCanvas = useCallback(
        (imageData: Partial<ImageState>) => {
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
    }, [maxHistory, pastLayerStates])

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

    const prevSelectedIndex = useRef(0);
    useEffect(() => {
        if (latestImages.length > 0 && prevSelectedIndex.current !== state.selectedIndex) {
            addToCanvas(latestImages[state.selectedIndex]);
            prevSelectedIndex.current = state.selectedIndex;
        }
    }, [state.selectedIndex]);

    return (
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
    )
}

const PaintQueueButtons = () => {
    const toast = useToast({});
    const [queue, setQueue] = useRecoilState(atom.queueState);
    const setAddToQueue = useSetRecoilState(addToQueueState);

    const boundingBoxCoordinates = useRecoilValue(boundingBoxCoordinatesAtom);  
    const boundingBoxDimensions = useRecoilValue(boundingBoxDimensionsAtom);
    const layerState = useRecoilValue(layerStateAtom);  

    const imageSettings = useRecoilValue(queueSettingsSelector);
    const shouldPreserveMaskedArea = useRecoilValue(shouldPreserveMaskedAreaAtom)
    const stageScale = useRecoilValue(stageScaleAtom);   
    const useRandomSeed = useRecoilValue(randomSeedState);
    const setSeed = useSetRecoilState(seedState);

    const cloudMode = useRecoilValue(cloudModeState);

    const socket = useContext(SocketContext);
    const stopQueue = useCallback(() => {
        socket.emit('stop_queue');
    }, [socket]);

    const handleRunInpainting = useCallback(() => {
        const canvasBaseLayer = getCanvasBaseLayer();
        if(!canvasBaseLayer) {
            return;
        }

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
            toast({
                title: 'Added to Queue!',
                description: `Currently ${queue.length + 1} elements in queue`,
                status: 'success',
                position: 'top',
                duration: 2000,
                isClosable: false,
                containerStyle: {
                    pointerEvents: 'none'
                }
            });
            setAddToQueue(true);
            const settings = parseSettings(
                {...imageSettings,  
                    width: boundingBox.width,
                    height: boundingBox.height,
                    init_image: imageDataURL,
                    mask_image: combinedMask,
                    invert: shouldPreserveMaskedArea, 
                    id: `${Math.floor(Math.random() * Number.MAX_SAFE_INTEGER)}`},
                useRandomSeed
            );
    
            if(useRandomSeed) setSeed(settings.seed);
            setQueue((queue) => [...queue, settings]);
        }).catch(err =>{
           console.log(err);
        })        

    }, [boundingBoxCoordinates, boundingBoxDimensions, layerState.objects, stageScale, imageSettings, socket, shouldPreserveMaskedArea]);


    return <>
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
                    {computeShardCost(imageSettings)}
                </Text>
            </Button>
            : 
            <HStack>
                <Button
                    className="run-button"
                    ml={2}
                    onClick={handleRunInpainting}
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
    </>
}

function Paint () {
    const [focused, setFocused] = useState(false);

    return (
        <Box width="100%">
            <VStack align="center" spacing={4}>
                <Box className="paint-output">
                    <UnifiedCanvas />
                    <GenerationProgressBars />
                </Box>
                <PaintLatestImages focused={focused} />
                <PaintQueueButtons />

                <Box width="80%">
                    <Prompt setFocused={setFocused} />
                </Box>
            </VStack>
        </Box>
    );
}
export default Paint;
