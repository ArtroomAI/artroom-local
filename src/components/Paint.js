import 'tui-image-editor/dist/tui-image-editor.css';
import 'tui-color-picker/dist/tui-color-picker.css';

import React from 'react';
import { useState, useEffect, createRef } from 'react';
import { useInterval } from './Reusable/useInterval/useInterval';
import { useRecoilState } from 'recoil';
import * as atom from '../atoms/atoms';
import axios from 'axios';

import ImageEditor from '@toast-ui/react-image-editor';
import theme from '../themes/theme';
import {
    Flex,
    Box,
    VStack,
    HStack,
    Button,
    FormControl,
    Tooltip,
    Stack,
    Text,
    Image,
    Progress,
    Select,
    createStandaloneToast,
    SimpleGrid
} from '@chakra-ui/react';
import { FaQuestionCircle } from 'react-icons/fa';
import Prompt from './Prompt';
function Paint () {
    const imageEditor = createRef();

    const { ToastContainer, toast } = createStandaloneToast();

    const [paintType, setPaintType] = useRecoilState(atom.paintTypeState);
    const [init_image, setInitImage] = useRecoilState(atom.initImageState);

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
    const [strength, setStrength] = useRecoilState(atom.strengthState);

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
    const [delay, setDelay] = useRecoilState(atom.delayState);

    const [progress, setProgress] = useState(-1);
    const [stage, setStage] = useState('');
    const [running, setRunning] = useState(false);
    const [open_pictures, setOpenPictures] = useRecoilState(atom.openPicturesState);
    const [keep_warm, setKeepWarm] = useRecoilState(atom.keepWarmState);

    const [mainImage, setMainImage] = useRecoilState(atom.mainImageState);
    const [latestImages, setLatestImages] = useRecoilState(atom.latestImageState);
    const [latestImagesID, setLatestImagesID] = useRecoilState(atom.latestImagesIDState);

    const [paintHistory, setPaintHistory] = useRecoilState(atom.paintHistoryState);

    useEffect(
        () => {
            const interval = setInterval(
                () => axios.get(
                    'http://127.0.0.1:5300/get_progress',
                    { headers: { 'Content-Type': 'application/json' } }
                ).then((result) => {
                    if (result.data.status === 'Success' && result.data.content.percentage) {
                        setProgress(result.data.content.percentage);
                        setRunning(result.data.content.running);
                        setStage(result.data.content.stage);
                    } else {
                        setProgress(-1);
                        setRunning(false);
                        setStage('');
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

    /*
     * Used to get history. Maybe some kind of fix for painting?
     * useInterval(() => {
     *   console.log(imageEditor.current.getInstance()._invoker._undoStack);
     * }, 3000);
     */

    const submitEvent = (event) => {
        const instance = imageEditor.current.getInstance();
        instance._graphics.getCanvas().on({
            'selection:updated': (options) => options.target.sendToBack(),
            'selection:created': (options) => options.target.sendToBack()
        });
        instance.addShape(
            'rect',
            {
                fill: 'white',
                stroke: 'white',
                strokeWidth: 1,
                width: instance.getCanvasSize().width,
                height: instance.getCanvasSize().height,
                isRegular: false
            }
        ).then((objectProps) => {
            const dataURL = instance.toDataURL();
            instance.removeObject(objectProps.id);
            const output = {
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
                reverse_mask: paintType === 'Use Reverse Mask',
                mask: dataURL,
                delay
            };
            axios.post(
                'http://127.0.0.1:5300/add_to_queue',
                output,
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
        });
    };

    useEffect(
        () => {
            if (init_image.length > 0) {
                imageEditor.current.getInstance().loadImageFromURL(
                    init_image,
                    'output'
                );
            } else {
                imageEditor.current.getInstance().loadImageFromURL(
                    'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7',
                    'blank'
                ).
                    then((result) => {
                        imageEditor.current.getInstance().ui.resizeEditor({
                            imageSize: { oldWidth: result.oldWidth,
                                oldHeight: result.oldHeight,
                                newWidth: result.newWidth,
                                newHeight: result.newHeight }
                        });
                    }).
                    catch((err) => {
                        console.error(
                            'Something went wrong:',
                            err
                        );
                    });
            }
        },
        [init_image]
    );

    return (
        <Box
            align="center"
            width="100%">
            <VStack
                align="center"
                spacing={4}>
                <Box
                    className="paint-output"
                    ratio={16 / 9}
                    width="75%">
                    <ImageEditor
                        cssMaxHeight={500}
                        cssMaxWidth={700}
                        includeUI={{
                            loadImage: {
                                path: init_image,
                                name: 'ArtroomLogo'
                            },
                            menu: ['draw', 'shape', 'crop', 'flip', 'rotate', 'filter'],
                            initMenu: 'draw',
                            uiSize: {
                                // Width: '1000px',
                                height: '700px'
                            },
                            theme,
                            menuBarPosition: 'bottom'
                        }}
                        ref={imageEditor}
                        selectionStyle={{
                            cornerSize: 20,
                            rotatingPointOffset: 70
                        }}
                        usageStatistics={false}
                    />

                    {
                        progress > 0
                            ? <Progress
                                align="left"
                                hasStripe
                                value={progress} />
                            : <></>
                    }

                    <Box
                        maxHeight="120px"
                        overflowY="auto"
                        width="50%">
                        <SimpleGrid
                            minChildWidth="100px"
                            spacing="10px">
                            {latestImages?.map((image, index) => (
                                <Image
                                    fit="scale-down"
                                    h="5vh"
                                    key={index}
                                    onClick={() => setMainImage(image)}
                                    src={image}
                                />
                            ))}
                        </SimpleGrid>
                    </Box>
                </Box>

                <HStack>
                    <Tooltip
                        fontSize="md"
                        label={<Stack>
                            {/* <Text>
                                Use Painted Image: Sends your colored image into Img2Img and runs it on the whole image
                            </Text> */}

                            <Text>
                                Use Mask: Paint over a region and do generate art ONLY on that region
                            </Text>

                            <Text>
                                Use Reverse Mask: Paint over a region and do everything EXCEPT that region
                            </Text>

                            <Text>
                                Please do NOT use white as your mask color (you can use white in Painted Image mode, just not Mask Mode). All other colors will be treated the same.
                            </Text>
                        </Stack>}
                        placement="top"
                        shouldWrapChildren>
                        <FaQuestionCircle color="#777" />
                    </Tooltip>

                    <FormControl className="paint-type">
                        <Select
                            id="paintType"
                            name="paintType"
                            onChange={(event) => setPaintType(event.target.value)}
                            value={paintType}
                            variant="outline"
                        >
                            {/* <option style={{ backgroundColor: '#080B16' }} value='Use Painted Image'>Use Painted Image</option> */}

                            <option
                                style={{ backgroundColor: '#080B16' }}
                                value="Use Mask">
                                Use Mask
                            </option>

                            <option
                                style={{ backgroundColor: '#080B16' }}
                                value="Use Reverse Mask">
                                Use Reverse Mask
                            </option>
                        </Select>
                    </FormControl>

                    <Button
                        ml={2}
                        onClick={submitEvent}
                        width="250px">
                        Run
                    </Button>
                </HStack>

                <Box width="80%">
                    <Prompt />
                </Box>
            </VStack>
        </Box>
    );
}
export default Paint;
